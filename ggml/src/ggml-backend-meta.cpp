#شامل "ggml.h"
#شامل "ggml-impl.h"
#شامل "ggml-backend.h"
#شامل "ggml-backend-impl.h"
#شامل "ggml-alloc.h"
#شامل "ggml-cpp.h"

#شامل <algorithm>
#شامل <cassert>
#شامل <cmath>
#شامل <cstddef>
#شامل <cstdint>
#شامل <cstring>
#شامل <map>
#شامل <memory>
#شامل <set>
#شامل <string>
#شامل <tuple>
#شامل <utility>
#شامل <vector>

ساخت ggml_backend_meta_device;
ساخت ggml_backend_meta_بفر_type;
ساخت ggml_backend_meta_بفر;
ساخت ggml_backend_meta;

ثابت حرف * ggml_backend_meta_بانٹو_axis_name(شمار ggml_backend_meta_بانٹو_axis بانٹو_axis) {
    انتخاب (بانٹو_axis) {
        حالت GGML_BACKEND_بانٹو_AXIS_0:
            واپس "0";
        حالت GGML_BACKEND_بانٹو_AXIS_1:
            واپس "1";
        حالت GGML_BACKEND_بانٹو_AXIS_2:
            واپس "2";
        حالت GGML_BACKEND_بانٹو_AXIS_3:
            واپس "3";
        حالت GGML_BACKEND_بانٹو_AXIS_MIRRORED:
            واپس "MIRRORED";
        حالت GGML_BACKEND_بانٹو_AXIS_PARTIAL:
            واپس "PARTIAL";
        حالت GGML_BACKEND_بانٹو_AXIS_NONE:
            واپس "NONE";
        حالت GGML_BACKEND_بانٹو_AXIS_UNKNOWN:
            واپس "UNKNOWN";
        طےشدہ:
            GGML_ABORT("fatal error");
    }
}

//
// meta backend device
//

ساخت ggml_backend_meta_device_context {
    std::vector<ggml_backend_dev_t>     simple_devs;
    ggml_backend_meta_get_بانٹو_state_t get_بانٹو_state;
    خالی *                              get_بانٹو_state_ud;

    std::string name;
    std::string description;

    ggml_backend_meta_device_context(
            std::vector<ggml_backend_dev_t> simple_devs, ggml_backend_meta_get_بانٹو_state_t get_بانٹو_state, خالی * get_بانٹو_state_ud) :
            simple_devs(std::move(simple_devs)), get_بانٹو_state(get_بانٹو_state), get_بانٹو_state_ud(get_بانٹو_state_ud) {
        name        = std::string("Meta(");
        description = std::string("Meta(");
        برائے (size_t i = 0; i < simple_devs.size(); i++) {
            اگر (i > 0) {
                name        += ",";
                description += ",";
            }
            name        += ggml_backend_dev_name       (simple_devs[i]);
            description += ggml_backend_dev_description(simple_devs[i]);
        }
        name        += ")";
        description += ")";
    }

    منطقی عامل<(ثابت ggml_backend_meta_device_context & other) ثابت {
        واپس std::tie(simple_devs, get_بانٹو_state, get_بانٹو_state_ud)
            < std::tie(other.simple_devs, other.get_بانٹو_state, other.get_بانٹو_state_ud);
    }
};

ساکن منطقی ggml_backend_dev_is_meta(ggml_backend_dev_t dev);

ساکن ثابت حرف * ggml_backend_meta_device_get_name(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ثابت ggml_backend_meta_device_context * meta_dev_ctx = (ثابت ggml_backend_meta_device_context *) dev->context;
    واپس meta_dev_ctx->name.c_str();
}

ساکن ثابت حرف * ggml_backend_meta_device_get_description(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ثابت ggml_backend_meta_device_context * meta_dev_ctx = (ثابت ggml_backend_meta_device_context *) dev->context;
    واپس meta_dev_ctx->description.c_str();
}

ساکن خالی ggml_backend_meta_device_get_حافظہ(ggml_backend_dev_t dev, size_t * آزاد, size_t * کل) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ثابت ggml_backend_meta_device_context * meta_dev_ctx = (ثابت ggml_backend_meta_device_context *) dev->context;
    *آزاد  = 0;
    *کل = 0;
    برائے (ggml_backend_dev_t dev : meta_dev_ctx->simple_devs) {
        size_t tmp_free, tmp_کل;
        ggml_backend_dev_حافظہ(dev, &tmp_free, &tmp_کل);
        *آزاد  += tmp_free;
        *کل += tmp_کل;
    }
}

ساکن شمار ggml_backend_dev_type ggml_backend_meta_device_get_type(ggml_backend_dev_t dev) {
    واپس GGML_BACKEND_DEVICE_TYPE_META;

    GGML_غیر_مستعمل(dev);
}

ساکن خالی ggml_backend_meta_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ثابت ggml_backend_meta_device_context * meta_dev_ctx = (ثابت ggml_backend_meta_device_context *) dev->context;

    // TODO replace placeholders
    props->name        = ggml_backend_meta_device_get_name(dev);
    props->description = ggml_backend_meta_device_get_description(dev);
    props->type        = ggml_backend_meta_device_get_type(dev);
    props->device_id   = 0;

    ggml_backend_meta_device_get_حافظہ(dev, &props->حافظہ_free, &props->حافظہ_کل);

    props->caps = {
        /* .async                 = */ سچ,
        /* .host_buffer           = */ جھوٹ, // Not implemented.
        /* .buffer_from_host_ptr  = */ جھوٹ, // Not implemented.
        /* .events                = */ جھوٹ, // Not implemented.
        /* .mmap_support          = */ سچ,
    };
    برائے (ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
        ggml_backend_dev_props tmp_props;
        ggml_backend_dev_get_props(simple_dev, &tmp_props);
        props->caps.async                = props->caps.async                && tmp_props.caps.async;
        props->caps.host_بفر          = props->caps.host_بفر          && tmp_props.caps.host_بفر;
        props->caps.بفر_from_host_ptr = props->caps.بفر_from_host_ptr && tmp_props.caps.بفر_from_host_ptr;
        props->caps.events               = props->caps.events               && tmp_props.caps.events;
        props->caps.mmap_support         = props->caps.mmap_support         && tmp_props.caps.mmap_support;
    }
}

ساکن ggml_backend_t ggml_backend_meta_device_init_backend(ggml_backend_dev_t dev, ثابت حرف * params);

ساکن ggml_backend_بفر_type_t ggml_backend_meta_device_get_بفر_type(ggml_backend_dev_t dev);

ساکن ggml_backend_بفر_type_t ggml_backend_meta_device_get_host_بفر_type(ggml_backend_dev_t dev);

ساکن منطقی ggml_backend_meta_device_supports_op(ggml_backend_dev_t dev, ثابت ggml_tensor * op) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ثابت ggml_backend_meta_device_context * meta_dev_ctx = (ثابت ggml_backend_meta_device_context *) dev->context;
    واپس std::all_of(meta_dev_ctx->simple_devs.begin(), meta_dev_ctx->simple_devs.end(),
        [op](ggml_backend_dev_t simple_dev) { واپس ggml_backend_dev_supports_op(simple_dev, op); });
}

ساکن منطقی ggml_backend_meta_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_بفر_type_t buft) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ggml_backend_dev_t dev_buft = ggml_backend_buft_get_device(buft);
    اگر (!ggml_backend_dev_is_meta(dev_buft)) {
        واپس جھوٹ;
    }
    ثابت ggml_backend_meta_device_context * meta_dev_ctx      = (ثابت ggml_backend_meta_device_context *) dev->context;
    ثابت ggml_backend_meta_device_context * meta_buft_dev_ctx = (ثابت ggml_backend_meta_device_context *) dev_buft->context;
    اگر (meta_dev_ctx->simple_devs.size() != meta_buft_dev_ctx->simple_devs.size()) {
        واپس جھوٹ;
    }
    برائے (size_t i = 0; i < meta_dev_ctx->simple_devs.size(); i++) {
        اگر (meta_dev_ctx->simple_devs[i] != meta_buft_dev_ctx->simple_devs[i]) {
            واپس جھوٹ;
        }
    }
    واپس سچ;
}

ساکن ثابت ggml_backend_device_i ggml_backend_meta_device_iface = {
    /* .get_name             = */ ggml_backend_meta_device_get_name,
    /* .get_description      = */ ggml_backend_meta_device_get_description,
    /* .get_memory           = */ ggml_backend_meta_device_get_حافظہ,
    /* .get_type             = */ ggml_backend_meta_device_get_type,
    /* .get_props            = */ ggml_backend_meta_device_get_props,
    /* .init_backend         = */ ggml_backend_meta_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_meta_device_get_بفر_type,
    /* .get_host_buffer_type = */ ggml_backend_meta_device_get_host_بفر_type,
    /* .buffer_from_host_ptr = */ خالی_اشاریہ,
    /* .supports_op          = */ ggml_backend_meta_device_supports_op,
    /* .supports_buft        = */ ggml_backend_meta_device_supports_buft,
    /* .offload_op           = */ خالی_اشاریہ,
    /* .event_new            = */ خالی_اشاریہ,
    /* .event_free           = */ خالی_اشاریہ,
    /* .event_synchronize    = */ خالی_اشاریہ,
};

ساکن منطقی ggml_backend_dev_is_meta(ggml_backend_dev_t dev) {
    واپس dev != خالی_اشاریہ && dev->iface.get_name == ggml_backend_meta_device_iface.get_name;
}

ساکن size_t ggml_backend_meta_dev_n_devs(ggml_backend_dev_t meta_dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(meta_dev));
    ثابت ggml_backend_meta_device_context * meta_dev_ctx = (ثابت ggml_backend_meta_device_context *) meta_dev->context;
    واپس meta_dev_ctx->simple_devs.size();
}

ساکن ggml_backend_dev_t ggml_backend_meta_dev_simple_dev(ggml_backend_dev_t meta_dev, size_t اشاریہ) {
    GGML_ASSERT(ggml_backend_dev_is_meta(meta_dev));
    ثابت ggml_backend_meta_device_context * meta_dev_ctx = (ثابت ggml_backend_meta_device_context *) meta_dev->context;
    GGML_ASSERT(اشاریہ < meta_dev_ctx->simple_devs.size());
    واپس meta_dev_ctx->simple_devs[اشاریہ];
}

ggml_backend_dev_t ggml_backend_meta_device(
        ggml_backend_dev_t * devs, size_t n_devs, ggml_backend_meta_get_بانٹو_state_t get_بانٹو_state, خالی * get_بانٹو_state_ud) {
    GGML_ASSERT(n_devs <= GGML_BACKEND_META_MAX_DEVICES);
    // TODO: this is not thread-safe - needs to be fixed
    ساکن std::vector<std::unique_ptr<ggml_backend_meta_device_context>>         ctxs;
    ساکن std::map<ggml_backend_meta_device_context, ساخت ggml_backend_device> meta_devs;

    std::vector<ggml_backend_dev_t> simple_devs;
    simple_devs.reserve(n_devs);
    برائے (size_t i = 0; i < n_devs; i++) {
        simple_devs.push_back(devs[i]);
    }
    ggml_backend_meta_device_context ctx(simple_devs, get_بانٹو_state, get_بانٹو_state_ud);

    {
        خودکار it = meta_devs.find(ctx);
        اگر (it != meta_devs.end()) {
            واپس &it->second;
        }
    }
    ctxs.push_back(std::make_unique<ggml_backend_meta_device_context>(ctx));

    ساخت ggml_backend_device meta_dev = {
        /*iface  =*/ ggml_backend_meta_device_iface,
        /*reg    =*/ خالی_اشاریہ,
        /*ctx    =*/ ctxs.back().get(),
    };

    خودکار نتیجہ = meta_devs.emplace(*ctxs.back(), meta_dev);
    واپس &نتیجہ.پہلا->second;
}

//
// meta backend buffer type
//

ساخت ggml_backend_meta_بفر_type_context {
    std::vector<ggml_backend_بفر_type_t> simple_bufts;

    std::string name;

    ggml_backend_meta_بفر_type_context(std::vector<ggml_backend_بفر_type_t> simple_bufts) : simple_bufts(std::move(simple_bufts)) {
        name = "Meta(";
        برائے (size_t i = 0; i < simple_bufts.size(); i++) {
            اگر (i > 0) {
                name += ",";
            }
            name += ggml_backend_buft_name(simple_bufts[i]);
        }
        name += ")";
    }

    منطقی عامل<(ثابت ggml_backend_meta_بفر_type_context & other) ثابت {
        واپس simple_bufts < other.simple_bufts;
    }
};

ساکن size_t ggml_backend_meta_buft_n_bufts(ggml_backend_بفر_type_t meta_buft) {
    GGML_ASSERT(ggml_backend_buft_is_meta(meta_buft));
    ثابت ggml_backend_meta_بفر_type_context * meta_buft_ctx = (ثابت ggml_backend_meta_بفر_type_context *) meta_buft->context;
    واپس meta_buft_ctx->simple_bufts.size();
}

ساکن ثابت حرف * ggml_backend_meta_بفر_type_get_name(ggml_backend_بفر_type_t buft) {
    GGML_ASSERT(ggml_backend_buft_is_meta(buft));
    ثابت ggml_backend_meta_بفر_type_context * meta_buft_ctx = (ثابت ggml_backend_meta_بفر_type_context *) buft->context;
    واپس meta_buft_ctx->name.c_str();
}

ساکن ggml_backend_بفر_type_t ggml_backend_meta_buft_simple_buft(ggml_backend_بفر_type_t meta_buft, size_t اشاریہ) {
    GGML_ASSERT(ggml_backend_buft_is_meta(meta_buft));
    ثابت ggml_backend_meta_بفر_type_context * meta_buft_ctx = (ثابت ggml_backend_meta_بفر_type_context *) meta_buft->context;
    GGML_ASSERT(اشاریہ < meta_buft_ctx->simple_bufts.size());
    واپس meta_buft_ctx->simple_bufts[اشاریہ];
}

ساکن ggml_backend_بفر_t ggml_backend_meta_بفر_type_alloc_بفر(ggml_backend_بفر_type_t buft, size_t size);

ساکن ggml_backend_بفر_t ggml_backend_meta_بفر_type_alloc_بفر_n(ggml_backend_بفر_type_t buft, ggml_tensor ** tensors, صحیح n_tensors);

ساکن size_t ggml_backend_meta_بفر_type_get_alignment(ggml_backend_بفر_type_t buft) {
    ثابت size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_alignment = 1;
    برائے (size_t i = 0; i < n_simple_bufts; i++) {
        ثابت size_t alignment = ggml_backend_buft_get_alignment(ggml_backend_meta_buft_simple_buft(buft, i));
        max_alignment = std::max(max_alignment, alignment);
        GGML_ASSERT(max_alignment % alignment == 0);
    }
    واپس max_alignment;
}

ساکن size_t ggml_backend_meta_بفر_type_get_max_سائز(ggml_backend_بفر_type_t buft) {
    ثابت size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_size = SIZE_MAX;
    برائے (size_t i = 0; i < n_simple_bufts; i++) {
        max_size = std::min(max_size, ggml_backend_buft_get_max_سائز(ggml_backend_meta_buft_simple_buft(buft, i)));
    }
    واپس max_size;
}

ساکن size_t ggml_backend_meta_بفر_type_get_alloc_سائز(ggml_backend_بفر_type_t buft, ثابت ggml_tensor * tensor) {
    ثابت size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_alloc_سائز = 0;
    برائے (size_t i = 0; i < n_simple_bufts; i++) {
        ثابت size_t alloc_سائز = ggml_backend_buft_get_alloc_سائز(ggml_backend_meta_buft_simple_buft(buft, i), tensor);
        max_alloc_سائز = std::max(max_alloc_سائز, alloc_سائز);
    }
    واپس max_alloc_سائز;
}

ساکن منطقی ggml_backend_meta_بفر_type_is_host(ggml_backend_بفر_type_t buft) {
    ثابت size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    برائے (size_t i = 0; i < n_simple_bufts; i++) {
        اگر (!ggml_backend_buft_is_host(ggml_backend_meta_buft_simple_buft(buft, i))) {
            واپس جھوٹ;
        }
    }
    واپس سچ;
}

ساکن ثابت ساخت ggml_backend_بفر_type_i ggml_backend_meta_بفر_type_iface = {
    /* .get_name            = */ ggml_backend_meta_بفر_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_meta_بفر_type_alloc_بفر,
    /* .alloc_buffer_n      = */ ggml_backend_meta_بفر_type_alloc_بفر_n,
    /* .get_alignment       = */ ggml_backend_meta_بفر_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_meta_بفر_type_get_max_سائز,
    /* .get_alloc_size      = */ ggml_backend_meta_بفر_type_get_alloc_سائز,
    /* .get_alloc_size_n    = */ تہی,
    /* .is_host             = */ ggml_backend_meta_بفر_type_is_host,
};

منطقی ggml_backend_buft_is_meta(ggml_backend_بفر_type_t buft) {
    واپس buft != خالی_اشاریہ && buft->iface.get_name == ggml_backend_meta_بفر_type_iface.get_name;
}

ساکن ggml_backend_بفر_type_t ggml_backend_meta_device_get_بفر_type(ggml_backend_dev_t dev) {
    ساکن std::map<ggml_backend_dev_t, ساخت ggml_backend_بفر_type> meta_bufts;
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    {
        خودکار it = meta_bufts.find(dev);
        اگر (it != meta_bufts.end()) {
            واپس &it->second;
        }
    }

    ثابت size_t n_devs = ggml_backend_meta_dev_n_devs(dev);
    std::vector<ggml_backend_بفر_type_t> simple_bufts;
    simple_bufts.reserve(n_devs);
    برائے (size_t i = 0; i < n_devs; i++) {
        simple_bufts.push_back(ggml_backend_dev_بفر_type(ggml_backend_meta_dev_simple_dev(dev, i)));
    }
    ggml_backend_meta_بفر_type_context * buft_ctx = نیا ggml_backend_meta_بفر_type_context(simple_bufts);

    ساخت ggml_backend_بفر_type meta_buft = {
        /*iface  =*/ ggml_backend_meta_بفر_type_iface,
        /*device =*/ dev,
        /*ctx    =*/ buft_ctx,
    };
    خودکار نتیجہ = meta_bufts.emplace(dev, meta_buft);
    واپس &نتیجہ.پہلا->second;
}

ساکن ggml_backend_بفر_type_t ggml_backend_meta_device_get_host_بفر_type(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ثابت ggml_backend_meta_device_context * meta_dev_ctx = (ثابت ggml_backend_meta_device_context *) dev->context;

    ggml_backend_بفر_type_t host_buft = خالی_اشاریہ;
    برائے (ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
        ggml_backend_بفر_type_t simple_host_buft = ggml_backend_dev_host_بفر_type(simple_dev);
        اگر (simple_host_buft == خالی_اشاریہ) {
            واپس خالی_اشاریہ;
        }
        اگر (host_buft == خالی_اشاریہ) {
            host_buft = simple_host_buft;
        } ورنہ اگر (host_buft != simple_host_buft) {
            // if different simple devices have different host buffer types,
            // we cannot provide a single host buffer type for the meta device
            واپس خالی_اشاریہ;
        }
    }
    واپس host_buft;
}

//
// meta backend buffer
//

// Container to hold the tensor slices per simple ggml backend buffer.
ساخت ggml_backend_meta_simple_tensor_container {
    std::vector<ggml_context_ptr> ctxs;
    std::map<ثابت ggml_tensor *, std::vector<ggml_tensor *>> simple_tensors;

    ggml_backend_meta_simple_tensor_container(ثابت ggml_init_params & params, ثابت صحیح n_simple) {
        ctxs.reserve(n_simple);
        برائے (صحیح i = 0; i < n_simple; i++) {
            ctxs.emplace_back(ggml_init(params));
        }
    }
    ggml_backend_meta_simple_tensor_container() {}
};

ساخت ggml_backend_meta_بفر_context {
    // FIXME
    // Most tensors can simply be stored statically in their own buffer.
    // Externally created views however also need a mapping to simple tensors but they use the buffer of the view source.
    // If external views are simply using that buffer they will slowly deplete its memory.
    // Current solution: rotating set of 2 "compute" containers to hold external views, works correctly for llama.cpp.
    // Long-term: tie the lifetime of external views to the meta backend executing the graph instead,
    //     currently not possible due to graph-external operations in the backend scheduler.
    ggml_backend_meta_simple_tensor_container stc_static;
    ggml_backend_meta_simple_tensor_container stc_compute[2];
    صحیح stc_compute_اشاریہ      = 0;
    صحیح stc_compute_اشاریہ_اگلا = 0;
    std::vector<ggml_backend_بفر_ptr> bufs;

    // FIXME
    // The size of the split state cache is unbounded and can theoretically grow infinitely large.
    // However, it is also expensive to build and clearing it on every rebuild in ggml_backend_meta_graph_compute is too expensive.
    ساکن constexpr size_t nbtc = GGML_TENSOR_سائز - حجم(ggml_tensor::padding);
    std::map<std::pair<ثابت ggml_tensor *, منطقی>, std::pair<ggml_backend_meta_بانٹو_state, حرف[nbtc]>> بانٹو_state_cache;

    صحیح debug;

    ggml_backend_meta_بفر_context(
            ggml_backend_meta_simple_tensor_container & stc_static,
            ggml_backend_meta_simple_tensor_container & stc_compute_0,
            ggml_backend_meta_simple_tensor_container & stc_compute_1,
            ثابت std::vector<ggml_backend_بفر_t> & bufs)
            : stc_static(std::move(stc_static)), stc_compute{std::move(stc_compute_0), std::move(stc_compute_1)} {
        یہ->bufs.reserve(bufs.size());
        برائے (ggml_backend_بفر_t buf : bufs) {
            یہ->bufs.emplace_back(buf);
        }
        ثابت حرف * GGML_META_DEBUG = getenv("GGML_META_DEBUG");
        debug = GGML_META_DEBUG ? atoi(GGML_META_DEBUG) : 0;
    }

    ggml_backend_meta_simple_tensor_container & get_simple_tensor_container(ثابت ggml_tensor * tensor) {
        اگر (stc_static.simple_tensors.find(tensor) != stc_static.simple_tensors.end()) {
            واپس stc_static;
        }
        واپس stc_compute[stc_compute_اشاریہ];
    }
};

ساکن خالی ggml_backend_meta_بفر_free_بفر(ggml_backend_بفر_t بفر) {
    GGML_ASSERT(ggml_backend_بفر_is_meta(بفر));
    ggml_backend_meta_بفر_context * buf_ctx = (ggml_backend_meta_بفر_context *) بفر->context;
    حذف buf_ctx;
}

ساکن size_t ggml_backend_meta_بفر_n_bufs(ggml_backend_بفر_t meta_buf) {
    GGML_ASSERT(ggml_backend_بفر_is_meta(meta_buf));
    ggml_backend_meta_بفر_context * buf_ctx = (ggml_backend_meta_بفر_context *) meta_buf->context;
    واپس buf_ctx->bufs.size();
}

ساکن ggml_backend_بفر_t ggml_backend_meta_بفر_simple_بفر(ggml_backend_بفر_t meta_buf, size_t اشاریہ) {
    GGML_ASSERT(ggml_backend_بفر_is_meta(meta_buf));
    ggml_backend_meta_بفر_context * buf_ctx = (ggml_backend_meta_بفر_context *) meta_buf->context;
    GGML_ASSERT(اشاریہ < buf_ctx->bufs.size());
    واپس buf_ctx->bufs[اشاریہ].get();
}

ساکن ساخت ggml_tensor * ggml_backend_meta_بفر_simple_tensor(ثابت ساخت ggml_tensor * tensor, size_t اشاریہ) {
    GGML_ASSERT(ggml_backend_بفر_is_meta(tensor->بفر));
    ggml_backend_meta_بفر_context * buf_ctx = (ggml_backend_meta_بفر_context *) tensor->بفر->context;
    GGML_ASSERT(اشاریہ < buf_ctx->bufs.size());

    ggml_backend_meta_simple_tensor_container & stc = buf_ctx->get_simple_tensor_container(tensor);
    خودکار it = stc.simple_tensors.find(tensor);
    اگر (it == stc.simple_tensors.end()) {
        واپس خالی_اشاریہ;
    }
    واپس it->second[اشاریہ];
}

ساکن ساخت ggml_backend_meta_بانٹو_state ggml_backend_meta_get_بانٹو_state(ثابت ساخت ggml_tensor * tensor, منطقی assume_sync);

ساکن ساخت ggml_backend_meta_بانٹو_state ggml_backend_meta_get_بانٹو_state(
        ggml_backend_meta_simple_tensor_container & stc, ثابت ساخت ggml_tensor * tensor, منطقی assume_sync) {
    // FIXME Currently this function preserves/erases the information in n_segments and nr in an inconsistent way.
    // Since the operations in question are developed specifically for llama.cpp this currently does not manifest as a bug there.
    // However, in a broader ggml context with arbitrary ggml graphs this can lead to unexpected results.
    ثابت size_t n_bufs = ggml_backend_meta_بفر_n_bufs(tensor->بفر);
    ggml_backend_meta_بفر_context * buf_ctx = (ggml_backend_meta_بفر_context *) tensor->بفر->context;

    خودکار بانٹو_حالتیں_equal = [&](ثابت ggml_backend_meta_بانٹو_state & a, ثابت ggml_backend_meta_بانٹو_state & b) -> منطقی {
        اگر (a.axis != b.axis) {
            واپس جھوٹ;
        }
        برائے (size_t j = 0; j < n_bufs; j++) {
            int64_t مجموع_a = 0;
            برائے (size_t s = 0; s < a.n_segments; s++) {
                مجموع_a += a.ne[s*n_bufs + j] * a.nr[s];
            }
            int64_t مجموع_b = 0;
            برائے (size_t s = 0; s < b.n_segments; s++) {
                مجموع_b += b.ne[s*n_bufs + j] * b.nr[s];
            }
            اگر (مجموع_a != مجموع_b) {
                واپس جھوٹ;
            }
        }
        واپس سچ;
    };

    خودکار handle_generic = [&](ثابت std::vector<ggml_backend_meta_بانٹو_state> & src_ss, منطقی scalar_only) -> ggml_backend_meta_بانٹو_state {
        ggml_backend_meta_بانٹو_state ret = {GGML_BACKEND_بانٹو_AXIS_NONE, {0}, {1}, 1};
        برائے (size_t i = 0; i < GGML_MAX_SRC; i++) {
            اگر (tensor->src[i] == خالی_اشاریہ || tensor->src[i] == tensor) {
                جاری;
            }
            اگر (ret.axis == GGML_BACKEND_بانٹو_AXIS_NONE) {
                ret = src_ss[i];
            } ورنہ اگر (!بانٹو_حالتیں_equal(src_ss[i], ret)) {
                ret = {GGML_BACKEND_بانٹو_AXIS_UNKNOWN, {0}, {1}, 1};
                توڑو;
            }
        }
        اگر (ret.axis == GGML_BACKEND_بانٹو_AXIS_NONE) {
            ret = {GGML_BACKEND_بانٹو_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        اگر (scalar_only && ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
            ret = {GGML_BACKEND_بانٹو_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        GGML_ASSERT(ret.axis != GGML_BACKEND_بانٹو_AXIS_UNKNOWN);
        واپس ret;
    };

    // Some ops process data on a per-row bases:
    خودکار handle_فی_صف = [&](ثابت std::vector<ggml_backend_meta_بانٹو_state> & src_ss) -> ggml_backend_meta_بانٹو_state {
        GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_بانٹو_AXIS_0);
        واپس src_ss[0];
    };

    // Some ops broadcast the src1 data across src0:
    خودکار handle_bin_bcast = [&](ثابت std::vector<ggml_backend_meta_بانٹو_state> & src_ss) -> ggml_backend_meta_بانٹو_state {
        اگر (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS &&
                tensor->src[1]->ne[src_ss[0].axis] == 1 && src_ss[1].axis == GGML_BACKEND_بانٹو_AXIS_MIRRORED) {
            واپس src_ss[0];
        }
        اگر (src_ss[2].axis == GGML_BACKEND_بانٹو_AXIS_MIRRORED && (src_ss[0].axis == src_ss[1].axis ||
           (src_ss[0].axis == GGML_BACKEND_بانٹو_AXIS_MIRRORED && (src_ss[1].axis == GGML_BACKEND_بانٹو_AXIS_PARTIAL)))) {
            واپس src_ss[0]; // GGML_OP_ADD_ID
        }
        GGML_ASSERT(tensor->src[2] == خالی_اشاریہ || src_ss[2].axis == GGML_BACKEND_بانٹو_AXIS_MIRRORED);
        واپس handle_generic(src_ss, /*scalar_only =*/ جھوٹ);
    };

    خودکار handle_concat = [&](ثابت std::vector<ggml_backend_meta_بانٹو_state> & src_ss) -> ggml_backend_meta_بانٹو_state {
        ثابت ggml_backend_meta_بانٹو_axis concat_axis = ggml_backend_meta_بانٹو_axis(ggml_get_op_params_i32(tensor, 0));
        اگر (src_ss[0].axis == GGML_BACKEND_بانٹو_AXIS_MIRRORED && src_ss[1].axis >= 0 && src_ss[1].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(concat_axis != src_ss[1].axis);
            واپس src_ss[1];
        }
        اگر (src_ss[1].axis == GGML_BACKEND_بانٹو_AXIS_MIRRORED && src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(concat_axis != src_ss[0].axis);
            واپس src_ss[0];
        }
        اگر (src_ss[0].axis == src_ss[1].axis && src_ss[0].axis != concat_axis) {
            واپس src_ss[0];
        }
        واپس handle_generic(src_ss, /*scalar_only =*/ سچ);
    };

    خودکار handle_mul_mat = [&](ثابت std::vector<ggml_backend_meta_بانٹو_state> & src_ss) -> ggml_backend_meta_بانٹو_state {
        اگر (src_ss[0].axis == GGML_BACKEND_بانٹو_AXIS_MIRRORED && src_ss[1].axis == GGML_BACKEND_بانٹو_AXIS_MIRRORED) {
            واپس {GGML_BACKEND_بانٹو_AXIS_MIRRORED, {0}, {1}, 1};
        }
        اگر (src_ss[0].axis == GGML_BACKEND_بانٹو_AXIS_1 && src_ss[1].axis == GGML_BACKEND_بانٹو_AXIS_MIRRORED) {
            ggml_backend_meta_بانٹو_state ret = src_ss[0];
            ret.axis = GGML_BACKEND_بانٹو_AXIS_0;
            ret.nr[0] = 1;
            ret.n_segments = 1;
            واپس ret;
        }
        اگر (src_ss[1].axis == GGML_BACKEND_بانٹو_AXIS_1 && src_ss[0].axis == GGML_BACKEND_بانٹو_AXIS_MIRRORED) {
            واپس src_ss[1];
        }
        اگر (src_ss[0].axis == GGML_BACKEND_بانٹو_AXIS_0 && src_ss[1].axis == GGML_BACKEND_بانٹو_AXIS_0) {
            GGML_ASSERT(بانٹو_حالتیں_equal(src_ss[0], src_ss[1]));
            واپس {assume_sync ? GGML_BACKEND_بانٹو_AXIS_MIRRORED : GGML_BACKEND_بانٹو_AXIS_PARTIAL, {0}, {1}, 1};
        }
        اگر (src_ss[0].axis == src_ss[1].axis && src_ss[0].axis >= GGML_BACKEND_بانٹو_AXIS_2 &&
                src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(بانٹو_حالتیں_equal(src_ss[0], src_ss[1]));
            واپس src_ss[0];
        }
        // batched matmul with the batches split across devices and a replicated activation
        اگر (src_ss[0].axis >= GGML_BACKEND_بانٹو_AXIS_2 && src_ss[0].axis < GGML_MAX_DIMS &&
                src_ss[1].axis == GGML_BACKEND_بانٹو_AXIS_MIRRORED) {
            واپس src_ss[0];
        }
        GGML_ABORT("unsupported mul_mat split states: node=%s src0=%s axis=%d src1=%s axis=%d",
            tensor->name, tensor->src[0]->name, (صحیح) src_ss[0].axis, tensor->src[1]->name, (صحیح) src_ss[1].axis);
        //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
    };

    خودکار handle_reshape = [&](ثابت std::vector<ggml_backend_meta_بانٹو_state> & src_ss) -> ggml_backend_meta_بانٹو_state {
        انتخاب (src_ss[0].axis) {
            حالت GGML_BACKEND_بانٹو_AXIS_0:
            حالت GGML_BACKEND_بانٹو_AXIS_1:
            حالت GGML_BACKEND_بانٹو_AXIS_2:
            حالت GGML_BACKEND_بانٹو_AXIS_3: {
                int64_t بنیاد_ne_اندر = 1;
                برائے (صحیح dim = 0; dim <= src_ss[0].axis; dim++) {
                    بنیاد_ne_اندر *= tensor->src[0]->ne[dim];
                }
                اگر (src_ss[0].n_segments == 1) {
                    بنیاد_ne_اندر /= src_ss[0].nr[0];
                    اگر (src_ss[0].axis == ggml_n_dims(tensor->src[0]) - 1 && src_ss[0].nr[0] == 1) {
                        واپس {ggml_backend_meta_بانٹو_axis(ggml_n_dims(tensor) - 1), {0}, {1}, 1};
                    }
                    اگر (src_ss[0].axis == GGML_BACKEND_بانٹو_AXIS_0 && tensor->ne[0] == tensor->src[0]->ne[0] &&
                            tensor->ne[1] == 1 && src_ss[0].nr[0] == 1) {
                        منطقی complete_صفیں = سچ;
                        برائے (size_t j = 0; j < n_bufs; j++) {
                            ثابت int64_t ne = src_ss[0].ne[j];
                            complete_صفیں = complete_صفیں && (ne == 0 || ne == tensor->src[0]->ne[0]);
                        }
                        اگر (complete_صفیں) {
                            // Move a complete dim-0 split to the following singleton dimension.
                            واپس {GGML_BACKEND_بانٹو_AXIS_1, {0}, {1}, 1};
                        }
                    }
                }
                // Reshape outputs use one segment; split-state propagation merges source segments.
                int64_t بنیاد_ne_برآمد = 1;
                برائے (صحیح dim = 0; dim < GGML_MAX_DIMS; dim++) {
                    بنیاد_ne_برآمد *= tensor->ne[dim];
                    اگر (بنیاد_ne_برآمد % بنیاد_ne_اندر == 0) {
                        واپس {ggml_backend_meta_بانٹو_axis(dim), {0}, {uint32_t(بنیاد_ne_برآمد/بنیاد_ne_اندر)}, 1};
                    }
                    اگر (بنیاد_ne_برآمد > بنیاد_ne_اندر) {
                        GGML_ASSERT(src_ss[0].n_segments == 1);
                        GGML_ASSERT(src_ss[0].nr[0]      == 1);
                        واپس {ggml_backend_meta_بانٹو_axis(dim), {0}, {1}, 1};
                    }
                }
                GGML_ABORT("shape mismatch for %s", ggml_op_name(tensor->op));
            }
            حالت GGML_BACKEND_بانٹو_AXIS_MIRRORED:
            حالت GGML_BACKEND_بانٹو_AXIS_PARTIAL: {
                واپس src_ss[0];
            }
            طےشدہ: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    خودکار handle_cpy = [&](ثابت std::vector<ggml_backend_meta_بانٹو_state> & src_ss) -> ggml_backend_meta_بانٹو_state {
        اگر (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            واپس handle_reshape(src_ss);
        }
        واپس handle_generic(src_ss, /*scalar_only =*/ جھوٹ);
    };

    خودکار handle_view = [&](ثابت std::vector<ggml_backend_meta_بانٹو_state> & src_ss) -> ggml_backend_meta_بانٹو_state {
        اگر (ggml_is_contiguous(tensor) && ggml_is_contiguous(tensor->src[0])) {
            واپس handle_reshape(src_ss);
        }
        ثابت صحیح axis = src_ss[0].axis;
        {
            منطقی all_strides_the_same = سچ;
            برائے (صحیح dim = 0; dim < GGML_MAX_DIMS; dim++) {
                اگر (tensor->ne[dim] == 1 && tensor->src[0]->ne[dim] == 1) {
                    جاری;
                }
                اگر (tensor->nb[dim] != tensor->src[0]->nb[dim]) {
                    all_strides_the_same = جھوٹ;
                    توڑو;
                }
            }
            اگر (all_strides_the_same) {
                واپس src_ss[0];
            }
        }
        اگر (!ggml_is_permuted(tensor) && !ggml_is_permuted(tensor->src[0]) && axis >= 0 && axis < GGML_MAX_DIMS-1) {
            برائے (صحیح dim = 0; dim < GGML_MAX_DIMS-1; dim++) {
                اگر (tensor->nb[dim+1] == tensor->src[0]->nb[axis+1]) {
                    واپس {ggml_backend_meta_بانٹو_axis(dim), {0}, {1}, 1};
                }
            }
            GGML_ABORT("fatal error");
        }
        اگر (src_ss[0].axis == GGML_BACKEND_بانٹو_AXIS_MIRRORED || src_ss[0].axis == GGML_BACKEND_بانٹو_AXIS_PARTIAL) {
            واپس src_ss[0];
        }
        GGML_ABORT("view of permuted tensor not implemented");
        //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
    };

    خودکار handle_ترتیب_بدلو = [&](ثابت std::vector<ggml_backend_meta_بانٹو_state> & src_ss) -> ggml_backend_meta_بانٹو_state {
        انتخاب (src_ss[0].axis) {
            حالت GGML_BACKEND_بانٹو_AXIS_0:
            حالت GGML_BACKEND_بانٹو_AXIS_1:
            حالت GGML_BACKEND_بانٹو_AXIS_2:
            حالت GGML_BACKEND_بانٹو_AXIS_3: {
                GGML_ASSERT(src_ss[0].n_segments == 1 || src_ss[0].nr[0] == 1);
                واپس {ggml_backend_meta_بانٹو_axis(tensor->op_params[src_ss[0].axis]), {0}, {src_ss[0].nr[0]}, 1};
            }
            حالت GGML_BACKEND_بانٹو_AXIS_MIRRORED:
            حالت GGML_BACKEND_بانٹو_AXIS_PARTIAL: {
                واپس src_ss[0];
            }
            طےشدہ: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    خودکار handle_transpose = [&](ثابت std::vector<ggml_backend_meta_بانٹو_state> & src_ss) -> ggml_backend_meta_بانٹو_state {
        انتخاب (src_ss[0].axis) {
            حالت GGML_BACKEND_بانٹو_AXIS_0:
            حالت GGML_BACKEND_بانٹو_AXIS_1: {
                GGML_ASSERT(src_ss[0].n_segments == 1 || src_ss[0].nr[0] == 1);
                واپس {ggml_backend_meta_بانٹو_axis(صحیح(src_ss[0].axis) ^ 1), {0}, {src_ss[0].nr[0]}, 1};
            }
            حالت GGML_BACKEND_بانٹو_AXIS_2:
            حالت GGML_BACKEND_بانٹو_AXIS_3:
            حالت GGML_BACKEND_بانٹو_AXIS_MIRRORED:
            حالت GGML_BACKEND_بانٹو_AXIS_PARTIAL: {
                واپس src_ss[0];
            }
            طےشدہ: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    خودکار handle_get_صفیں = [&](ثابت std::vector<ggml_backend_meta_بانٹو_state> & src_ss) -> ggml_backend_meta_بانٹو_state {
        اگر (src_ss[0].axis == GGML_BACKEND_بانٹو_AXIS_0 && src_ss[1].axis == GGML_BACKEND_بانٹو_AXIS_MIRRORED) {
            واپس src_ss[0];
        }
        واپس handle_generic(src_ss, /*scalar_only =*/ سچ);
    };

    خودکار handle_سیٹ_صفیں = [&](ثابت std::vector<ggml_backend_meta_بانٹو_state> & src_ss) -> ggml_backend_meta_بانٹو_state {
        GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_بانٹو_AXIS_1);
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_بانٹو_AXIS_MIRRORED);
        GGML_ASSERT(بانٹو_حالتیں_equal(src_ss[0], src_ss[2]));
        واپس src_ss[0];
    };

    خودکار handle_rope = [&](ثابت std::vector<ggml_backend_meta_بانٹو_state> & src_ss) -> ggml_backend_meta_بانٹو_state {
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_بانٹو_AXIS_MIRRORED);
        واپس src_ss[0];
    };

    خودکار handle_pad = [&](ثابت std::vector<ggml_backend_meta_بانٹو_state> & src_ss) -> ggml_backend_meta_بانٹو_state {
        اگر (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(tensor->op_params[2*src_ss[0].axis + 0] == 0);
            GGML_ASSERT(tensor->op_params[2*src_ss[0].axis + 1] == 0);
        }
        واپس src_ss[0];
    };

    خودکار handle_pool_1d = [&](ثابت std::vector<ggml_backend_meta_بانٹو_state> & src_ss) -> ggml_backend_meta_بانٹو_state {
        اگر (src_ss[0].axis == GGML_BACKEND_بانٹو_AXIS_0) {
            واپس {GGML_BACKEND_بانٹو_AXIS_MIRRORED, {0}, {1}, 1};
        }
        واپس src_ss[0];
    };

    خودکار handle_flash_attn_ext = [&](ثابت std::vector<ggml_backend_meta_بانٹو_state> & src_ss) -> ggml_backend_meta_بانٹو_state {
        GGML_ASSERT(tensor->src[3] == خالی_اشاریہ || src_ss[3].axis == GGML_BACKEND_بانٹو_AXIS_MIRRORED);

        اگر (src_ss[0].axis == GGML_BACKEND_بانٹو_AXIS_MIRRORED) {
            GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_بانٹو_AXIS_MIRRORED);
            GGML_ASSERT(src_ss[2].axis == GGML_BACKEND_بانٹو_AXIS_MIRRORED);
            GGML_ASSERT(tensor->src[4] == خالی_اشاریہ || src_ss[4].axis == GGML_BACKEND_بانٹو_AXIS_MIRRORED);
            واپس {GGML_BACKEND_بانٹو_AXIS_MIRRORED, {0}, {1}, 1};
        }

        GGML_ASSERT(src_ss[0].axis == GGML_BACKEND_بانٹو_AXIS_2);
        ثابت منطقی kv_بانٹو = src_ss[1].axis == GGML_BACKEND_بانٹو_AXIS_2 &&
                src_ss[2].axis == GGML_BACKEND_بانٹو_AXIS_2;
        ثابت منطقی kv_mirrored = src_ss[1].axis == GGML_BACKEND_بانٹو_AXIS_MIRRORED &&
                src_ss[2].axis == GGML_BACKEND_بانٹو_AXIS_MIRRORED;
        GGML_ASSERT(kv_بانٹو || kv_mirrored);
        GGML_ASSERT(tensor->src[4] == خالی_اشاریہ || src_ss[4].axis == GGML_BACKEND_بانٹو_AXIS_0);
        واپس {GGML_BACKEND_بانٹو_AXIS_1, {0}, {1}, 1};
    };

    خودکار handle_lightning_indexer = [&](
            ثابت std::vector<ggml_backend_meta_بانٹو_state> & src_ss) -> ggml_backend_meta_بانٹو_state {
        برائے (size_t i = 0; i < 4; i++) {
            GGML_ASSERT(src_ss[i].axis == GGML_BACKEND_بانٹو_AXIS_MIRRORED);
        }
        واپس {GGML_BACKEND_بانٹو_AXIS_MIRRORED, {0}, {1}, 1};
    };

    خودکار handle_ssm_conv = [&](ثابت std::vector<ggml_backend_meta_بانٹو_state> & src_ss) -> ggml_backend_meta_بانٹو_state {
        اگر (src_ss[0].axis == src_ss[1].axis) {
            اگر (src_ss[0].axis == GGML_BACKEND_بانٹو_AXIS_0) {
                واپس {GGML_BACKEND_بانٹو_AXIS_1, {0}, {1}, 1};
            }
            اگر (src_ss[0].axis == GGML_BACKEND_بانٹو_AXIS_1) {
                واپس {GGML_BACKEND_بانٹو_AXIS_0, {0}, {1}, 1};
            }
        }
        واپس handle_generic(src_ss, /*scalar_only =*/ جھوٹ);
    };

    خودکار handle_gated_delta_net = [&](ثابت std::vector<ggml_backend_meta_بانٹو_state> & src_ss) -> ggml_backend_meta_بانٹو_state {
        اگر (src_ss[0].axis == GGML_BACKEND_بانٹو_AXIS_MIRRORED && src_ss[1].axis == GGML_BACKEND_بانٹو_AXIS_MIRRORED &&
                src_ss[2].axis == GGML_BACKEND_بانٹو_AXIS_MIRRORED && src_ss[3].axis == GGML_BACKEND_بانٹو_AXIS_MIRRORED &&
                src_ss[4].axis == GGML_BACKEND_بانٹو_AXIS_MIRRORED && src_ss[5].axis == GGML_BACKEND_بانٹو_AXIS_MIRRORED) {
            واپس src_ss[0];
        }
        GGML_ASSERT(src_ss[0].axis == GGML_BACKEND_بانٹو_AXIS_1);
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_بانٹو_AXIS_1);
        GGML_ASSERT(src_ss[2].axis == GGML_BACKEND_بانٹو_AXIS_1);
        GGML_ASSERT(src_ss[3].axis == GGML_BACKEND_بانٹو_AXIS_1);
        GGML_ASSERT(src_ss[4].axis == GGML_BACKEND_بانٹو_AXIS_1);
        // state shape is [S_v, S_v, H_v, n_seqs] (s0 only); the heads dim is its own axis 2,
        // so a head-aligned split on the input cache lands on axis 2 here.
        GGML_ASSERT(src_ss[5].axis == GGML_BACKEND_بانٹو_AXIS_2 || src_ss[5].axis == GGML_BACKEND_بانٹو_AXIS_1 || src_ss[5].axis == GGML_BACKEND_بانٹو_AXIS_0);
        واپس {GGML_BACKEND_بانٹو_AXIS_0, {0}, {1}, 1};
    };

    خودکار calculate_بانٹو_state = [&]() -> ggml_backend_meta_بانٹو_state {
        اگر (ggml_nelements(tensor) == 0) {
            واپس {GGML_BACKEND_بانٹو_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        اگر (ggml_backend_بفر_get_usage(tensor->بفر) != GGML_BACKEND_بفر_USAGE_COMPUTE && tensor->view_src == خالی_اشاریہ) {
            ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_بفر_get_type(tensor->بفر));
            ثابت ggml_backend_meta_device_context * dev_ctx = (ثابت ggml_backend_meta_device_context *) dev->context;
            ggml_backend_meta_بانٹو_state ret = dev_ctx->get_بانٹو_state(tensor, dev_ctx->get_بانٹو_state_ud);
            اگر (ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
                ثابت int64_t granularity = ret.axis == GGML_BACKEND_بانٹو_AXIS_0 ? ggml_blck_سائز(tensor->type) : 1;
                int64_t ne_مجموع = 0;
                برائے (size_t s = 0; s < ret.n_segments; s++) {
                    برائے (size_t j = 0; j < n_bufs; j++) {
                        GGML_ASSERT(ret.ne[s*n_bufs + j] % granularity == 0);
                        ne_مجموع += ret.ne[s*n_bufs + j] * ret.nr[s];
                    }
                }
                GGML_ASSERT(ne_مجموع == tensor->ne[ret.axis]);
            } ورنہ اگر (ret.axis == GGML_BACKEND_بانٹو_AXIS_PARTIAL) {
                GGML_ASSERT(ret.n_segments == 1);
                GGML_ASSERT(ret.nr[0] == 1);
            }
            واپس ret;
        }

        std::vector<ggml_backend_meta_بانٹو_state> src_ss(GGML_MAX_SRC, {GGML_BACKEND_بانٹو_AXIS_NONE, {0}, {1}, 1});
        برائے (size_t i = 0; i < GGML_MAX_SRC; i++) {
            اگر (tensor->src[i] == خالی_اشاریہ || tensor->src[i] == tensor) {
                src_ss[i] = {GGML_BACKEND_بانٹو_AXIS_UNKNOWN, {0}, {1}, 1};
                جاری;
            }
            src_ss[i] = ggml_backend_meta_get_بانٹو_state(stc, tensor->src[i], /*assume_sync =*/ سچ);
            GGML_ASSERT(src_ss[i].axis != GGML_BACKEND_بانٹو_AXIS_UNKNOWN);
        }

        ggml_backend_meta_بانٹو_state بانٹو_state;
        انتخاب (tensor->op) {
            حالت GGML_OP_NONE: {
                اگر (tensor->view_src != خالی_اشاریہ) {
                    // full-tensor view created with ggml_view_tensor, transparent for the split state
                    بانٹو_state = ggml_backend_meta_get_بانٹو_state(stc, tensor->view_src, assume_sync);
                } ورنہ {
                    بانٹو_state = {GGML_BACKEND_بانٹو_AXIS_MIRRORED, {0}, {1}, 1};
                }
            } توڑو;
            حالت GGML_OP_DUP: {
                بانٹو_state = handle_generic(src_ss, /*scalar_only =*/ سچ);
            } توڑو;
            حالت GGML_OP_جوڑو:
            حالت GGML_OP_جوڑو_ID: {
                بانٹو_state = handle_bin_bcast(src_ss);
            } توڑو;
            حالت GGML_OP_ADD1:
            حالت GGML_OP_ACC: {
                بانٹو_state = handle_generic(src_ss, /*scalar_only =*/ سچ);
            } توڑو;
            حالت GGML_OP_SUB:
            حالت GGML_OP_MUL:
            حالت GGML_OP_DIV: {
                بانٹو_state = handle_bin_bcast(src_ss);
            } توڑو;
            حالت GGML_OP_SQR:
            حالت GGML_OP_SQRT:
            حالت GGML_OP_LOG:
            حالت GGML_OP_SIN:
            حالت GGML_OP_COS: {
                بانٹو_state = handle_generic(src_ss, /*scalar_only =*/ جھوٹ);
            } توڑو;
            حالت GGML_OP_مجموع: {
                بانٹو_state = handle_generic(src_ss, /*scalar_only =*/ سچ);
            } توڑو;
            حالت GGML_OP_مجموع_صفیں:
            حالت GGML_OP_CUMSUM:
            حالت GGML_OP_اوسط:
            حالت GGML_OP_ARGMAX:
            حالت GGML_OP_گنتی_EQUAL: {
                بانٹو_state = handle_فی_صف(src_ss);
            } توڑو;
            حالت GGML_OP_REPEAT:
            حالت GGML_OP_REPEAT_عقب: {
                بانٹو_state = handle_generic(src_ss, /*scalar_only =*/ جھوٹ);
            } توڑو;
            حالت GGML_OP_CONCAT: {
                بانٹو_state = handle_concat(src_ss);
            } توڑو;
            حالت GGML_OP_SILU_عقب: {
                بانٹو_state = handle_generic(src_ss, /*scalar_only =*/ جھوٹ);
            } توڑو;
            حالت GGML_OP_NORM:
            حالت GGML_OP_RMS_NORM:
            حالت GGML_OP_RMS_NORM_عقب:
            حالت GGML_OP_گروہ_NORM:
            حالت GGML_OP_L2_NORM: {
                بانٹو_state = handle_فی_صف(src_ss);
            } توڑو;
            حالت GGML_OP_MUL_MAT:
            حالت GGML_OP_MUL_MAT_ID: {
                بانٹو_state = handle_mul_mat(src_ss);
            } توڑو;
            حالت GGML_OP_برآمد_PROD: {
                بانٹو_state = handle_generic(src_ss, /*scalar_only =*/ سچ);
            } توڑو;
            حالت GGML_OP_SCALE: {
                بانٹو_state = handle_generic(src_ss, /*scalar_only =*/ جھوٹ);
            } توڑو;
            حالت GGML_OP_سیٹ: {
                بانٹو_state = handle_generic(src_ss, /*scalar_only =*/ سچ);
            } توڑو;
            حالت GGML_OP_CPY: {
                بانٹو_state = handle_cpy(src_ss);
            } توڑو;
            حالت GGML_OP_CONT:
            حالت GGML_OP_RESHAPE: {
                بانٹو_state = handle_reshape(src_ss);
            } توڑو;
            حالت GGML_OP_VIEW: {
                بانٹو_state = handle_view(src_ss);
            } توڑو;
            حالت GGML_OP_ترتیب_بدلو: {
                بانٹو_state = handle_ترتیب_بدلو(src_ss);
            } توڑو;
            حالت GGML_OP_TRANSPOSE: {
                بانٹو_state = handle_transpose(src_ss);
            } توڑو;
            حالت GGML_OP_GET_صفیں: {
                بانٹو_state = handle_get_صفیں(src_ss);
            } توڑو;
            حالت GGML_OP_GET_صفیں_عقب: {
                بانٹو_state = handle_generic(src_ss, /*scalar_only =*/ سچ);
            } توڑو;
            حالت GGML_OP_سیٹ_صفیں: {
                بانٹو_state = handle_سیٹ_صفیں(src_ss);
            } توڑو;
            حالت GGML_OP_DIAG:
            حالت GGML_OP_DIAG_نقاب_INF:
            حالت GGML_OP_DIAG_نقاب_ZERO: {
                بانٹو_state = handle_generic(src_ss, /*scalar_only =*/ سچ);
            } توڑو;
            حالت GGML_OP_SOFT_MAX:
            حالت GGML_OP_SOFT_MAX_عقب: {
                بانٹو_state = handle_generic(src_ss, /*scalar_only =*/ جھوٹ);
            } توڑو;
            حالت GGML_OP_ROPE: {
                بانٹو_state = handle_rope(src_ss);
            } توڑو;
            حالت GGML_OP_ROPE_عقب: {
                بانٹو_state = handle_rope(src_ss);
            } توڑو;
            حالت GGML_OP_CLAMP: {
                بانٹو_state = handle_generic(src_ss, /*scalar_only =*/ جھوٹ);
            } توڑو;
            حالت GGML_OP_CONV_TRANSPOSE_1D:
            حالت GGML_OP_IM2_ستون:
            حالت GGML_OP_IM2_ستون_عقب:
            حالت GGML_OP_IM2_ستون_3_D:
            حالت GGML_OP_CONV_2D:
            حالت GGML_OP_CONV_3D:
            حالت GGML_OP_CONV_2D_DW:
            حالت GGML_OP_CONV_TRANSPOSE_2D:
            حالت GGML_OP_POOL_2D:
            حالت GGML_OP_POOL_2_D_عقب:
            حالت GGML_OP_UPSCALE: {
                بانٹو_state = handle_generic(src_ss, /*scalar_only =*/ سچ);
            } توڑو;
            حالت GGML_OP_POOL_1D: {
                بانٹو_state = handle_pool_1d(src_ss);
            } توڑو;
            حالت GGML_OP_PAD: {
                بانٹو_state = handle_pad(src_ss);
            } توڑو;
            حالت GGML_OP_PAD_REFLECT_1D:
            حالت GGML_OP_ROLL:
            حالت GGML_OP_ARANGE:
            حالت GGML_OP_TIMESTEP_EMBEDDING: {
                بانٹو_state = handle_generic(src_ss, /*scalar_only =*/ سچ);
            } توڑو;
            حالت GGML_OP_ARGSORT:
            حالت GGML_OP_چوٹی_K: {
                بانٹو_state = handle_فی_صف(src_ss);
            } توڑو;
            حالت GGML_OP_LEAKY_RELU: {
                بانٹو_state = handle_generic(src_ss, /*scalar_only =*/ جھوٹ);
            } توڑو;
            حالت GGML_OP_TRI: {
                بانٹو_state = handle_generic(src_ss, /*scalar_only =*/ سچ);
            } توڑو;
            حالت GGML_OP_FILL: {
                بانٹو_state = handle_generic(src_ss, /*scalar_only =*/ جھوٹ);
            } توڑو;
            حالت GGML_OP_FLASH_ATTN_EXT: {
                بانٹو_state = handle_flash_attn_ext(src_ss);
            } توڑو;
            حالت GGML_OP_FLASH_ATTN_عقب: {
                بانٹو_state = handle_generic(src_ss, /*scalar_only =*/ سچ);
            } توڑو;
            حالت GGML_OP_SSM_CONV: {
                بانٹو_state = handle_ssm_conv(src_ss);
            } توڑو;
            حالت GGML_OP_SSM_جائزہ:
            حالت GGML_OP_WIN_PART:
            حالت GGML_OP_WIN_UNPART:
            حالت GGML_OP_GET_REL_POS:
            حالت GGML_OP_جوڑو_REL_POS:
            حالت GGML_OP_RWKV_WKV6:
            حالت GGML_OP_GATED_خطی_ATTN:
            حالت GGML_OP_RWKV_WKV7:
            حالت GGML_OP_SOLVE_TRI: {
                بانٹو_state = handle_generic(src_ss, /*scalar_only =*/ سچ);
            } توڑو;
            حالت GGML_OP_GATED_DELTA_NET: {
                بانٹو_state = handle_gated_delta_net(src_ss);
            } توڑو;
            حالت GGML_OP_LIGHTNING_INDEXER: {
                بانٹو_state = handle_lightning_indexer(src_ss);
            } توڑو;
            حالت GGML_OP_DSV4_HC_COMB:
            حالت GGML_OP_DSV4_HC_PRE:
            حالت GGML_OP_DSV4_HC_POST:
            حالت GGML_OP_چوٹی_K_نقاب: {
                بانٹو_state = handle_generic(src_ss, /*scalar_only =*/ سچ);
            } توڑو;
            حالت GGML_OP_UNARY: {
                بانٹو_state = handle_generic(src_ss, /*scalar_only =*/ جھوٹ);
            } توڑو;
            حالت GGML_OP_MAP_CUSTOM1:
            حالت GGML_OP_MAP_CUSTOM2:
            حالت GGML_OP_MAP_CUSTOM3:
            حالت GGML_OP_CUSTOM: {
                بانٹو_state = handle_generic(src_ss, /*scalar_only =*/ سچ);
            } توڑو;
            حالت GGML_OP_CROSS_ENTROPY_LOSS:
            حالت GGML_OP_CROSS_ENTROPY_LOSS_عقب: {
                بانٹو_state = handle_فی_صف(src_ss);
            } توڑو;
            حالت GGML_OP_OPT_قدم_ADAMW:
            حالت GGML_OP_OPT_قدم_SGD:
            حالت GGML_OP_GLU: {
                بانٹو_state = handle_generic(src_ss, /*scalar_only =*/ جھوٹ);
            } توڑو;
            طےشدہ: {
                GGML_ABORT("ggml op not implemented: %s", ggml_op_name(tensor->op));
                بانٹو_state = {GGML_BACKEND_بانٹو_AXIS_UNKNOWN, {0}, {1}, 1};
            } توڑو;
        }
        اگر (بانٹو_state.axis >= 0 && بانٹو_state.axis < GGML_MAX_DIMS) {
            منطقی پہلا_src_بانٹو_by_axis = سچ;
            ثابت size_t n_bufs = ggml_backend_meta_بفر_n_bufs(tensor->بفر);

            برائے (size_t i = 0; i < GGML_MAX_SRC; i++) {
                اگر (tensor->src[i] == خالی_اشاریہ || src_ss[i].axis < 0 || src_ss[i].axis >= GGML_MAX_DIMS) {
                    جاری;
                }
                اگر (پہلا_src_بانٹو_by_axis) {
                    برائے (size_t j = 0; j < n_bufs; j++) {
                        // Take over ratio from src:
                        برائے (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            بانٹو_state.ne[s*n_bufs + j] = 0;
                        }
                        برائے (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            بانٹو_state.ne[j] += src_ss[i].ne[s*n_bufs + j] * src_ss[i].nr[s];
                        }
                        بانٹو_state.ne[j] *= tensor->ne[بانٹو_state.axis];
                        اگر (بانٹو_state.ne[j] != 0 || tensor->src[i]->ne[src_ss[i].axis] != 0) {
                            ثابت int64_t div = tensor->src[i]->ne[src_ss[i].axis] * بانٹو_state.nr[0];
                            GGML_ASSERT(بانٹو_state.ne[j] % div == 0);
                            بانٹو_state.ne[j] /= div;
                        }
                    }
                } ورنہ {
                    GGML_ASSERT(بانٹو_state.n_segments == 1);
                    برائے (size_t j = 0; j < n_bufs; j++) {
                        // Assert that ratio is consistent:
                        int64_t مجموع = 0;
                        برائے (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            مجموع += src_ss[i].ne[s*n_bufs + j] * src_ss[i].nr[s];
                        }
                        GGML_ASSERT(بانٹو_state.ne[j]*بانٹو_state.nr[0] * tensor->src[i]->ne[src_ss[i].axis]
                                                                 == مجموع * tensor->ne[بانٹو_state.axis]);
                    }
                }
                پہلا_src_بانٹو_by_axis = جھوٹ;
            }
            GGML_ASSERT(!پہلا_src_بانٹو_by_axis);
        }
        واپس بانٹو_state;
    };

    ثابت std::pair کلید = std::make_pair(tensor, assume_sync);
    خودکار it = buf_ctx->بانٹو_state_cache.find(کلید);
    اگر (it != buf_ctx->بانٹو_state_cache.end() && memcmp(it->second.second, (ثابت حرف *) tensor, حجم(it->second.second)) != 0) {
        buf_ctx->بانٹو_state_cache.clear();
        it = buf_ctx->بانٹو_state_cache.end();
    }

    اگر (it == buf_ctx->بانٹو_state_cache.end()) {
        buf_ctx->بانٹو_state_cache[کلید].پہلا = calculate_بانٹو_state();
        حافظہ_نقل(buf_ctx->بانٹو_state_cache[کلید].second, tensor, حجم(buf_ctx->بانٹو_state_cache[کلید].second));
        اگر (buf_ctx->debug > 0) {
            std::string srcs_info;
            برائے (size_t i = 0; i < GGML_MAX_SRC; i++) {
                اگر (tensor->src[i] == خالی_اشاریہ || tensor->src[i] == tensor) {
                    جاری;
                }
                اگر (!srcs_info.خالی_خانے()) {
                    srcs_info += ", ";
                }
                ثابت ggml_backend_meta_بانٹو_state بانٹو_state =
                        ggml_backend_meta_get_بانٹو_state(tensor->src[i], سچ);
                GGML_ASSERT(بانٹو_state.n_segments == 1);
                ثابت حرف * axis_name = ggml_backend_meta_بانٹو_axis_name(بانٹو_state.axis);
                std::string ne_info;
                برائے (size_t j = 0; j < n_bufs; j++) {
                    اگر (!ne_info.خالی_خانے()) {
                        ne_info += ", ";
                    }
                    ne_info += std::to_string(بانٹو_state.ne[j]) + "x" + std::to_string(بانٹو_state.nr[0]);
                }
                srcs_info += std::string(tensor->src[i]->name) + "[" + ggml_op_name(tensor->src[i]->op) + ", " + axis_name + ", {" + ne_info + "}]";
            }
            std::string ne_info;
            برائے (size_t j = 0; j < n_bufs; j++) {
                اگر (!ne_info.خالی_خانے()) {
                    ne_info += ", ";
                }
                ثابت ggml_backend_meta_بانٹو_state & ss = buf_ctx->بانٹو_state_cache[کلید].پہلا;
                ne_info += std::to_string(ss.ne[j]) + "x" + std::to_string(ss.nr[0]);
            }
            GGML_LOG_DEBUG("SPLIT_STATE: {%s} -> %s[%s, %s, {%s}]\n", srcs_info.c_str(), tensor->name, ggml_op_name(tensor->op),
                ggml_backend_meta_بانٹو_axis_name(buf_ctx->بانٹو_state_cache[کلید].پہلا.axis), ne_info.c_str());
        }
    }

    ggml_backend_meta_بانٹو_state ret = buf_ctx->بانٹو_state_cache[کلید].پہلا;
    GGML_ASSERT(ret.axis != GGML_BACKEND_بانٹو_AXIS_NONE);
#اگر_غیر_تعریف_شدہ NDEBUG
    اگر (ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
        int64_t ne_ret = 0;
        برائے (size_t s = 0; s < ret.n_segments; s++) {
            برائے (size_t j = 0; j < n_bufs; j++) {
                ne_ret += ret.ne[s*n_bufs + j] * ret.nr[s];
            }
        }
        أكد(ne_ret == tensor->ne[صحیح(ret.axis)]);
    }
#اگر_ختم // NDEBUG
    واپس ret;
}

ساکن منطقی ggml_backend_meta_is_host_view(ثابت ساخت ggml_tensor * tensor) {
    واپس ggml_is_view(tensor) && ggml_backend_بفر_is_host(tensor->view_src->بفر);
}

ساکن ساخت ggml_backend_meta_بانٹو_state ggml_backend_meta_get_بانٹو_state(ثابت ساخت ggml_tensor * tensor, منطقی assume_sync) {
    // [TAG_META_HOST_VIEWS]
    // TODO: technically, this check should not be needed if the backend scheduler correctly prevents assigning
    //       such host-buffer views to the meta backend. figure out how to update the scheduler logic to achieve that
    //       ref: https://github.com/ggml-org/llama.cpp/pull/30217
    اگر (!ggml_backend_بفر_is_meta(tensor->بفر)) {
        GGML_ASSERT(ggml_backend_meta_is_host_view(tensor));

        // the view is not allocated in the meta buffer, it is not split across the sub-devices
        واپس { GGML_BACKEND_بانٹو_AXIS_NONE, {0}, {1}, 1 };
    }

    ggml_backend_meta_بفر_context * buf_ctx = (ggml_backend_meta_بفر_context *) tensor->بفر->context;
    واپس ggml_backend_meta_get_بانٹو_state(buf_ctx->get_simple_tensor_container(tensor), tensor, assume_sync);
}

ساکن خالی * ggml_backend_meta_بفر_get_بنیاد(ggml_backend_بفر_t بفر) {
    GGML_غیر_مستعمل(بفر);
    واپس (خالی *) 0x1000000000000000; // FIXME
}

ساکن شمار ggml_حالت ggml_backend_meta_بفر_init_tensor_impl(ggml_backend_meta_simple_tensor_container & stc, ggml_tensor * tensor) {
    GGML_ASSERT(ggml_backend_بفر_is_meta(tensor->بفر));
    ggml_backend_meta_بفر_context * buf_ctx = (ggml_backend_meta_بفر_context *) tensor->بفر->context;
    ثابت size_t n_simple_bufs = ggml_backend_meta_بفر_n_bufs(tensor->بفر);

    ثابت ggml_backend_meta_بانٹو_state بانٹو_state = ggml_backend_meta_get_بانٹو_state(stc, tensor, /*assume_sync =*/ سچ);
    GGML_ASSERT(ggml_nelements(tensor) == 0 || بانٹو_state.axis != GGML_BACKEND_بانٹو_AXIS_UNKNOWN);
    GGML_ASSERT(بانٹو_state.n_segments <= 16);

    صحیح بانٹو_dim = بانٹو_state.axis;
    int64_t ne[GGML_MAX_DIMS];
    size_t  nb[GGML_MAX_DIMS];
    برائے (size_t k = 0; k < GGML_MAX_DIMS; k++) {
        ne[k] = tensor->ne[k];
        nb[k] = tensor->nb[k];
    }

    std::vector<ggml_tensor *> simple_tensors;
    simple_tensors.reserve(n_simple_bufs);
    برائے (size_t j = 0; j < n_simple_bufs; j++) {
        ggml_context          * simple_ctx = stc.ctxs[j].get();
        ggml_backend_بفر_t   simple_buf = buf_ctx->bufs[j].get();

        اگر (بانٹو_dim >= 0 && بانٹو_dim < GGML_MAX_DIMS) {
            // TODO: the following assert fails for llama-parallel even though the results are correct:
            // GGML_ASSERT(ggml_is_contiguously_allocated(tensor));
            ne[بانٹو_dim] = 0;
            برائے (size_t s = 0; s < بانٹو_state.n_segments; s++) {
                ne[بانٹو_dim] += بانٹو_state.ne[s*n_simple_bufs + j] * بانٹو_state.nr[s];
            }
            برائے (صحیح i = 0; i < GGML_MAX_DIMS; i++) {
                اگر (tensor->nb[i] > tensor->nb[بانٹو_dim]) {
                    nb[i] = tensor->nb[i] * ne[بانٹو_dim]/tensor->ne[بانٹو_dim];
                }
            }
        }

        ggml_tensor * t_ij = ggml_new_tensor(simple_ctx, tensor->type, GGML_MAX_DIMS, ne);
        t_ij->op = tensor->op;
        برائے (صحیح i = 0; i < GGML_MAX_DIMS; i++) {
            t_ij->nb[i] = nb[i];
        }
        t_ij->flags = tensor->flags;
        حافظہ_نقل(t_ij->op_params, tensor->op_params, حجم(tensor->op_params));
        ggml_سیٹ_name(t_ij, tensor->name);
        t_ij->بفر = simple_buf;
        t_ij->view_src = tensor->view_src;
        t_ij->view_offs = tensor->view_offs;
        اگر (t_ij->view_src != خالی_اشاریہ && ggml_backend_بفر_is_meta(t_ij->view_src->بفر)) {
            t_ij->view_src = ggml_backend_meta_بفر_simple_tensor(tensor->view_src, j);
            اگر (t_ij->view_offs > 0 && بانٹو_dim >= 0 && بانٹو_dim < GGML_MAX_DIMS) {
                GGML_ASSERT(tensor->ne[بانٹو_dim] != 0);
                ثابت صحیح بانٹو_dim_view_src = ggml_backend_meta_get_بانٹو_state(tensor->view_src, /*assume_sync =*/ سچ).axis;
                GGML_ASSERT(بانٹو_dim_view_src >= 0 && بانٹو_dim_view_src < GGML_MAX_DIMS);

                // The offset can be internal to the data split, in those cases the view offset should not be scaled.
                // If however, the offset is larger than the data split then it needs to be scaled proportionally.
                منطقی بانٹو_اندرونی_انحراف = t_ij->view_offs <= tensor->view_src->nb[بانٹو_dim_view_src];
                برائے (صحیح i = 0; i < GGML_MAX_DIMS; i++) {
                    ثابت size_t dim_سائز = tensor->ne[i] * tensor->nb[i];
                    اگر (tensor->view_offs <= dim_سائز && dim_سائز < tensor->nb[بانٹو_dim]) {
                        بانٹو_اندرونی_انحراف = سچ;
                        توڑو;
                    }
                }
                اگر (!بانٹو_اندرونی_انحراف) {
                    t_ij->view_offs = t_ij->view_offs * ne[بانٹو_dim]/tensor->ne[بانٹو_dim];
                }
            }
        }
        // TODO: revisit once the graph allocator has been refactored, see https://github.com/ggml-org/llama.cpp/pull/25051#issuecomment-4842873396
        ggml_backend_بفر_t init_buf = simple_buf;
        اگر (t_ij->view_src != خالی_اشاریہ) {
            t_ij->ڈیٹا = (حرف *) t_ij->view_src->ڈیٹا + t_ij->view_offs;
            // views inherit the source slice's concrete sub-buffer (issue 22197)
            اگر (tensor->view_src != خالی_اشاریہ && ggml_backend_بفر_is_meta(tensor->view_src->بفر)
                    && t_ij->view_src->بفر != خالی_اشاریہ) {
                t_ij->بفر = t_ij->view_src->بفر;
                init_buf     = t_ij->view_src->بفر;
            }
        } ورنہ اگر (simple_buf != خالی_اشاریہ) {
            اگر (ggml_backend_بفر_is_multi_بفر(simple_buf)) {
                GGML_ABORT("multi buffers are not supported by the meta backend");
            }
            t_ij->ڈیٹا = (حرف *) ggml_backend_بفر_get_بنیاد(simple_buf)
                + size_t(tensor->ڈیٹا) - size_t(ggml_backend_بفر_get_بنیاد(tensor->بفر));
        }

        اگر (init_buf) {
            // the backend that owns the buffer will set .extra
            ggml_backend_بفر_init_tensor(init_buf, t_ij);
        } ورنہ {
            t_ij->extra = tensor->extra;
        }

        برائے (صحیح i = 0; i < GGML_MAX_SRC; i++) {
            t_ij->src[i] = tensor->src[i];
            اگر (tensor->src[i] == tensor) {
                t_ij->src[i] = t_ij;
            } ورنہ اگر (t_ij->src[i] != خالی_اشاریہ && ggml_backend_بفر_is_meta(t_ij->src[i]->بفر)) {
                t_ij->src[i] = ggml_backend_meta_بفر_simple_tensor(tensor->src[i], j);
            }
        }

        simple_tensors.push_back(t_ij);
    }

    // If one of the sources has a zero-sized slice, disable the computation:
    برائے (صحیح i = 0; i < GGML_MAX_SRC; i++) {
        اگر (tensor->src[i] == خالی_اشاریہ || !ggml_backend_بفر_is_meta(tensor->src[i]->بفر)) {
            جاری;
        }

        ثابت ggml_backend_meta_بانٹو_state بانٹو_state_src = ggml_backend_meta_get_بانٹو_state(tensor->src[i], /*assume_sync =*/ سچ);
        اگر (بانٹو_state_src.axis < 0 || بانٹو_state_src.axis >= GGML_MAX_DIMS) {
            جاری;
        }
        برائے (size_t j = 0; j < n_simple_bufs; j++) {
            int64_t ne_مجموع = 0;
            برائے (size_t s = 0; s < بانٹو_state_src.n_segments; s++) {
                ne_مجموع += بانٹو_state_src.ne[s*n_simple_bufs + j] * بانٹو_state_src.nr[s];
            }
            اگر (ne_مجموع == 0) {
                simple_tensors[j]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
            }
        }
    }

    stc.simple_tensors[tensor] = simple_tensors;

    واپس GGML_حالت_SUCCESS;
}

ساکن شمار ggml_حالت ggml_backend_meta_بفر_init_tensor(ggml_backend_بفر_t بفر, ggml_tensor * tensor) {
    GGML_ASSERT(ggml_backend_بفر_is_meta(بفر));
    ggml_backend_meta_بفر_context * buf_ctx = (ggml_backend_meta_بفر_context *) بفر->context;
    buf_ctx->stc_compute_اشاریہ = buf_ctx->stc_compute_اشاریہ_اگلا;
    واپس ggml_backend_meta_بفر_init_tensor_impl(buf_ctx->get_simple_tensor_container(tensor), tensor);
}

ساکن خالی ggml_backend_meta_بفر_memset_tensor(
        ggml_backend_بفر_t بفر, ggml_tensor * tensor, uint8_t value, size_t انحراف, size_t size) {
    ثابت size_t n_bufs = ggml_backend_meta_بفر_n_bufs(بفر);
    ثابت ggml_backend_meta_بانٹو_state بانٹو_state =
            ggml_backend_meta_get_بانٹو_state(tensor, /*assume_sync =*/ جھوٹ);
    GGML_ASSERT(ggml_is_contiguous(tensor) || بانٹو_state.axis == GGML_BACKEND_بانٹو_AXIS_MIRRORED);

    اگر (بانٹو_state.n_segments != 1 || بانٹو_state.nr[0] != 1) {
        GGML_ASSERT(بانٹو_state.axis >= 0 && بانٹو_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(بانٹو_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        std::vector<size_t> simple_offsets(n_bufs, 0);
        اگر (بانٹو_state.axis == GGML_BACKEND_بانٹو_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            ثابت size_t صف_stride = tensor->nb[1];
            GGML_ASSERT(انحراف % صف_stride == 0);
            GGML_ASSERT(size   % صف_stride == 0);
            ثابت int64_t صف_آغاز = انحراف / صف_stride;
            ثابت int64_t صف_گنتی = size   / صف_stride;
            GGML_ASSERT(صف_آغاز + صف_گنتی <= tensor->ne[1]);

            ثابت int64_t blck_سائز = ggml_blck_سائز(tensor->type);
            برائے (size_t s = 0; s < بانٹو_state.n_segments; s++) {
                برائے (size_t r = 0; r < بانٹو_state.nr[s]; r++) {
                    برائے (size_t j = 0; j < n_bufs; j++) {
                        ggml_tensor * simple_tensor = ggml_backend_meta_بفر_simple_tensor(tensor, j);
                        GGML_ASSERT(بانٹو_state.ne[s*n_bufs + j] % blck_سائز == 0);
                        ثابت size_t nbytes = بانٹو_state.ne[s*n_bufs + j]/blck_سائز * tensor->nb[0];
                        برائے (int64_t صف = 0; صف < صف_گنتی; صف++) {
                            ggml_backend_tensor_memset(simple_tensor, value,
                                    simple_offsets[j] + (صف_آغاز + صف)*simple_tensor->nb[1], nbytes);
                        }
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            واپس;
        }

        GGML_ASSERT(بانٹو_state.axis == GGML_BACKEND_بانٹو_AXIS_1);

        ثابت size_t صف_stride = tensor->nb[2];
        GGML_ASSERT(انحراف % صف_stride == 0);
        GGML_ASSERT(size   % صف_stride == 0);
        ثابت int64_t صف_آغاز = انحراف / صف_stride;
        ثابت int64_t صف_گنتی = size   / صف_stride;
        GGML_ASSERT(صف_آغاز + صف_گنتی <= tensor->ne[2]);

        برائے (size_t s = 0; s < بانٹو_state.n_segments; s++) {
            برائے (size_t r = 0; r < بانٹو_state.nr[s]; r++) {
                برائے (size_t j = 0; j < n_bufs; j++) {
                    ggml_tensor * simple_tensor = ggml_backend_meta_بفر_simple_tensor(tensor, j);
                    ثابت size_t nbytes = بانٹو_state.ne[s*n_bufs + j] * tensor->nb[1];
                    برائے (int64_t صف = 0; صف < صف_گنتی; صف++) {
                        ggml_backend_tensor_memset(simple_tensor, value,
                                simple_offsets[j] + (صف_آغاز + صف)*simple_tensor->nb[2], nbytes);
                    }
                    simple_offsets[j] += nbytes;
                }
            }
        }
        واپس;
    }

    انتخاب (بانٹو_state.axis) {
        حالت GGML_BACKEND_بانٹو_AXIS_0:
        حالت GGML_BACKEND_بانٹو_AXIS_1:
        حالت GGML_BACKEND_بانٹو_AXIS_2: {
            ثابت size_t chunk_سائز_بھرا = tensor->nb[بانٹو_state.axis + 1];
            GGML_ASSERT(انحراف % chunk_سائز_بھرا == 0);
            GGML_ASSERT(size   % chunk_سائز_بھرا == 0);
            ثابت int64_t i_آغاز =  انحراف        / chunk_سائز_بھرا;
            ثابت int64_t i_stop  = (انحراف + size) / chunk_سائز_بھرا;
            برائے (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_بفر_simple_tensor(tensor, j);
                ثابت size_t chunk_سائز = simple_tensor->nb[بانٹو_state.axis + 1];
                اگر (chunk_سائز == 0) {
                    جاری;
                }
                برائے (int64_t i = i_آغاز; i < i_stop; i++) {
                    ggml_backend_tensor_memset(simple_tensor, value, i*chunk_سائز, chunk_سائز);
                }
            }
        } توڑو;
        حالت GGML_BACKEND_بانٹو_AXIS_PARTIAL: {
            GGML_ASSERT(value == 0);
            [[fallthrough]];
        }
        حالت GGML_BACKEND_بانٹو_AXIS_MIRRORED: {
            برائے (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_بفر_simple_tensor(tensor, j);
                ggml_backend_tensor_memset(simple_tensor, value, انحراف, size);
            }
        } توڑو;
        طےشدہ: {
            GGML_ABORT("fatal error");
        }
    }
}

ساکن خالی ggml_backend_meta_بفر_سیٹ_tensor(ggml_backend_بفر_t بفر, ggml_tensor * tensor, ثابت خالی * ڈیٹا, size_t انحراف, size_t size) {
    ثابت size_t n_bufs = ggml_backend_meta_بفر_n_bufs(بفر);
    ثابت ggml_backend_meta_بانٹو_state بانٹو_state = ggml_backend_meta_get_بانٹو_state(tensor, /*assume_sync =*/ جھوٹ);
    GGML_ASSERT(ggml_is_contiguous(tensor) || بانٹو_state.axis == GGML_BACKEND_بانٹو_AXIS_MIRRORED);

    اگر (بانٹو_state.n_segments != 1 || بانٹو_state.nr[0] != 1) {
        GGML_ASSERT(بانٹو_state.axis >= 0 && بانٹو_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(بانٹو_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        size_t انحراف_ڈیٹا = 0;
        std::vector<size_t> simple_offsets(n_bufs, 0);
        اگر (بانٹو_state.axis == GGML_BACKEND_بانٹو_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            ثابت size_t صف_stride = tensor->nb[1];
            GGML_ASSERT(انحراف % صف_stride == 0);
            GGML_ASSERT(size   % صف_stride == 0);
            ثابت int64_t صف_آغاز = انحراف / صف_stride;
            ثابت int64_t صف_گنتی = size   / صف_stride;
            GGML_ASSERT(صف_آغاز + صف_گنتی <= tensor->ne[1]);

            ثابت int64_t blck_سائز = ggml_blck_سائز(tensor->type);
            برائے (size_t s = 0; s < بانٹو_state.n_segments; s++) {
                برائے (size_t r = 0; r < بانٹو_state.nr[s]; r++) {
                    برائے (size_t j = 0; j < n_bufs; j++) {
                        ggml_tensor * simple_tensor = ggml_backend_meta_بفر_simple_tensor(tensor, j);
                        GGML_ASSERT(بانٹو_state.ne[s*n_bufs + j] % blck_سائز == 0);
                        ثابت size_t nbytes = بانٹو_state.ne[s*n_bufs + j]/blck_سائز * tensor->nb[0];
                        ggml_backend_tensor_سیٹ_2d(simple_tensor, (ثابت حرف *) ڈیٹا + انحراف_ڈیٹا,
                            simple_offsets[j] + صف_آغاز * simple_tensor->nb[1], nbytes,
                            صف_گنتی, simple_tensor->nb[1], tensor->nb[1]);
                        انحراف_ڈیٹا       += nbytes;
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            GGML_ASSERT(انحراف_ڈیٹا*صف_گنتی == size);
            واپس;
        }
        GGML_ASSERT(بانٹو_state.axis == GGML_BACKEND_بانٹو_AXIS_1);

        ثابت size_t صف_stride = tensor->nb[2];
        GGML_ASSERT(انحراف % صف_stride == 0);
        GGML_ASSERT(size   % صف_stride == 0);
        ثابت int64_t صف_آغاز = انحراف / صف_stride;
        ثابت int64_t صف_گنتی = size   / صف_stride;
        GGML_ASSERT(صف_آغاز + صف_گنتی <= tensor->ne[2]);

        برائے (size_t s = 0; s < بانٹو_state.n_segments; s++) {
            برائے (size_t r = 0; r < بانٹو_state.nr[s]; r++) {
                برائے (size_t j = 0; j < n_bufs; j++) {
                    ggml_tensor * simple_tensor = ggml_backend_meta_بفر_simple_tensor(tensor, j);
                    ثابت size_t nbytes = بانٹو_state.ne[s*n_bufs + j] * tensor->nb[1];
                    ggml_backend_tensor_سیٹ_2d(simple_tensor, (ثابت حرف *) ڈیٹا + انحراف_ڈیٹا,
                        simple_offsets[j] + صف_آغاز * simple_tensor->nb[2], nbytes,
                        صف_گنتی, simple_tensor->nb[2], tensor->nb[2]);
                    انحراف_ڈیٹا       += nbytes;
                    simple_offsets[j] += nbytes;
                }
            }
        }
        GGML_ASSERT(انحراف_ڈیٹا*صف_گنتی == size);
        واپس;
    }

    انتخاب (بانٹو_state.axis) {
        حالت GGML_BACKEND_بانٹو_AXIS_0:
        حالت GGML_BACKEND_بانٹو_AXIS_1:
        حالت GGML_BACKEND_بانٹو_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            ثابت size_t chunk_سائز_بھرا = tensor->nb[بانٹو_state.axis + 1];
            GGML_ASSERT(انحراف % chunk_سائز_بھرا == 0);
            GGML_ASSERT(size   % chunk_سائز_بھرا == 0);
            ثابت int64_t i_آغاز =  انحراف        /chunk_سائز_بھرا;
            ثابت int64_t i_stop  = (انحراف + size)/chunk_سائز_بھرا;
            size_t انحراف_j = 0;
            برائے (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_بفر_simple_tensor(tensor, j);
                ثابت size_t chunk_سائز_j = simple_tensor->nb[بانٹو_state.axis + 1];
                اگر (chunk_سائز_j == 0) {
                    جاری;
                }
                ثابت size_t simple_انحراف = i_آغاز * chunk_سائز_j;
                ggml_backend_tensor_سیٹ_2d(simple_tensor, (ثابت حرف *) ڈیٹا + انحراف_j, simple_انحراف, chunk_سائز_j, i_stop - i_آغاز, chunk_سائز_j, chunk_سائز_بھرا);
                انحراف_j += chunk_سائز_j;
            }
            GGML_ASSERT(انحراف_j == chunk_سائز_بھرا);
        } توڑو;
        حالت GGML_BACKEND_بانٹو_AXIS_MIRRORED: {
            برائے (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_بفر_simple_tensor(tensor, j);
                ggml_backend_tensor_سیٹ(simple_tensor, ڈیٹا, انحراف, size);
            }
        } توڑو;
        حالت GGML_BACKEND_بانٹو_AXIS_PARTIAL: {
            GGML_ASSERT(tensor->type == GGML_TYPE_F32);
            GGML_ASSERT(انحراف % حجم(اعشاری) == 0);
            GGML_ASSERT(size   % حجم(اعشاری) == 0);
            ثابت size_t n_اقدار = size / حجم(اعشاری);
            size_t n_contributors = 0;
            برائے (size_t j = 0; j < n_bufs; j++) {
                n_contributors += بانٹو_state.ne[j] != 0;
            }
            ثابت منطقی has_contributor_نقاب = n_contributors != 0;
            اگر (!has_contributor_نقاب) {
                n_contributors = n_bufs;
            }
            std::vector<اعشاری> tmp(n_اقدار);
            برائے (size_t i = 0; i < n_اقدار; i++) {
                tmp[i] = ((ثابت اعشاری *) ڈیٹا)[i] / n_contributors;
            }
            std::vector<اعشاری> zero;
            اگر (has_contributor_نقاب) {
                zero.resize(n_اقدار, 0.0f);
            }
            برائے (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_بفر_simple_tensor(tensor, j);
                ثابت اعشاری * partial = has_contributor_نقاب && بانٹو_state.ne[j] == 0 ? zero.ڈیٹا() : tmp.ڈیٹا();
                ggml_backend_tensor_سیٹ(simple_tensor, partial, انحراف, size);
            }
        } توڑو;
        طےشدہ: {
            GGML_ABORT("fatal error");
        }
    }
}

ساکن خالی ggml_backend_meta_بفر_get_tensor(ggml_backend_بفر_t بفر, ثابت ggml_tensor * tensor, خالی * ڈیٹا, size_t انحراف, size_t size) {
    ثابت size_t n_bufs = ggml_backend_meta_بفر_n_bufs(بفر);
    ثابت ggml_backend_meta_بانٹو_state بانٹو_state = ggml_backend_meta_get_بانٹو_state(tensor, /*assume_sync =*/ جھوٹ);
    GGML_ASSERT(ggml_is_contiguous(tensor) || بانٹو_state.axis == GGML_BACKEND_بانٹو_AXIS_MIRRORED);

    اگر (بانٹو_state.n_segments != 1 || بانٹو_state.nr[0] != 1) {
        GGML_ASSERT(بانٹو_state.axis >= 0 && بانٹو_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(بانٹو_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        size_t انحراف_ڈیٹا = 0;
        std::vector<size_t> simple_offsets(n_bufs, 0);
        اگر (بانٹو_state.axis == GGML_BACKEND_بانٹو_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            ثابت size_t صف_stride = tensor->nb[1];
            GGML_ASSERT(انحراف % صف_stride == 0);
            GGML_ASSERT(size   % صف_stride == 0);
            ثابت int64_t صف_آغاز = انحراف / صف_stride;
            ثابت int64_t صف_گنتی = size   / صف_stride;
            GGML_ASSERT(صف_آغاز + صف_گنتی <= tensor->ne[1]);

            ثابت int64_t blck_سائز = ggml_blck_سائز(tensor->type);
            برائے (size_t s = 0; s < بانٹو_state.n_segments; s++) {
                برائے (size_t r = 0; r < بانٹو_state.nr[s]; r++) {
                    برائے (size_t j = 0; j < n_bufs; j++) {
                        ثابت ggml_tensor * simple_tensor = ggml_backend_meta_بفر_simple_tensor(tensor, j);
                        GGML_ASSERT(بانٹو_state.ne[s*n_bufs + j] % blck_سائز == 0);
                        ثابت size_t nbytes = بانٹو_state.ne[s*n_bufs + j]/blck_سائز * tensor->nb[0];
                        ggml_backend_tensor_get_2d(simple_tensor, (حرف *) ڈیٹا + انحراف_ڈیٹا,
                            simple_offsets[j] + صف_آغاز * simple_tensor->nb[1], nbytes,
                            صف_گنتی, simple_tensor->nb[1], tensor->nb[1]);
                        انحراف_ڈیٹا       += nbytes;
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            GGML_ASSERT(انحراف_ڈیٹا*صف_گنتی == size);
            واپس;
        }
        GGML_ASSERT(بانٹو_state.axis == GGML_BACKEND_بانٹو_AXIS_1);

        ثابت size_t صف_stride = tensor->nb[2];
        GGML_ASSERT(انحراف % صف_stride == 0);
        GGML_ASSERT(size   % صف_stride == 0);
        ثابت int64_t صف_آغاز = انحراف / صف_stride;
        ثابت int64_t صف_گنتی = size   / صف_stride;
        GGML_ASSERT(صف_آغاز + صف_گنتی <= tensor->ne[2]);

        برائے (size_t s = 0; s < بانٹو_state.n_segments; s++) {
            برائے (size_t r = 0; r < بانٹو_state.nr[s]; r++) {
                برائے (size_t j = 0; j < n_bufs; j++) {
                    ثابت ggml_tensor * simple_tensor = ggml_backend_meta_بفر_simple_tensor(tensor, j);
                    ثابت size_t nbytes = بانٹو_state.ne[s*n_bufs + j] * tensor->nb[1];
                    ggml_backend_tensor_get_2d(simple_tensor, (حرف *) ڈیٹا + انحراف_ڈیٹا,
                        simple_offsets[j] + صف_آغاز * simple_tensor->nb[2], nbytes,
                        صف_گنتی, simple_tensor->nb[2], tensor->nb[2]);
                    انحراف_ڈیٹا       += nbytes;
                    simple_offsets[j] += nbytes;
                }
            }
        }
        GGML_ASSERT(انحراف_ڈیٹا*صف_گنتی == size);
        واپس;
    }

    انتخاب (بانٹو_state.axis) {
        حالت GGML_BACKEND_بانٹو_AXIS_0:
        حالت GGML_BACKEND_بانٹو_AXIS_1:
        حالت GGML_BACKEND_بانٹو_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            ثابت size_t chunk_سائز_بھرا = tensor->nb[بانٹو_state.axis + 1];
            GGML_ASSERT(انحراف % chunk_سائز_بھرا == 0);
            GGML_ASSERT(size   % chunk_سائز_بھرا == 0);
            ثابت int64_t i_آغاز =  انحراف        /chunk_سائز_بھرا;
            ثابت int64_t i_stop  = (انحراف + size)/chunk_سائز_بھرا;
            size_t انحراف_j = 0;
            برائے (size_t j = 0; j < n_bufs; j++){
                ثابت ggml_tensor * simple_tensor = ggml_backend_meta_بفر_simple_tensor(tensor, j);
                ثابت size_t chunk_سائز_j = simple_tensor->nb[بانٹو_state.axis + 1];
                اگر (chunk_سائز_j == 0) {
                    جاری;
                }
                ثابت size_t simple_انحراف = i_آغاز * chunk_سائز_j;
                ggml_backend_tensor_get_2d(simple_tensor, (حرف *) ڈیٹا + انحراف_j, simple_انحراف, chunk_سائز_j, i_stop - i_آغاز, chunk_سائز_j, chunk_سائز_بھرا);
                انحراف_j += chunk_سائز_j;
            }
            GGML_ASSERT(انحراف_j == chunk_سائز_بھرا);
        } توڑو;
        حالت GGML_BACKEND_بانٹو_AXIS_MIRRORED: {
            // TODO other simple backend may be better
            ثابت ggml_tensor * simple_tensor = ggml_backend_meta_بفر_simple_tensor(tensor, 0);
            ggml_backend_tensor_get(simple_tensor, ڈیٹا, انحراف, size);
        } توڑو;
        طےشدہ: {
            GGML_ABORT("fatal error");
        }
    }
}

ساکن خالی ggml_backend_meta_بفر_clear(ggml_backend_بفر_t بفر, uint8_t value) {
    ثابت size_t n_buffers = ggml_backend_meta_بفر_n_bufs(بفر);
    برائے (size_t i = 0; i < n_buffers; i++) {
        ggml_backend_بفر_clear(ggml_backend_meta_بفر_simple_بفر(بفر, i), value);
    }
}

ساکن خالی ggml_backend_meta_بفر_reset(ggml_backend_بفر_t بفر) {
    GGML_ASSERT(ggml_backend_بفر_is_meta(بفر));
    ggml_backend_meta_بفر_context * buf_ctx = (ggml_backend_meta_بفر_context *) بفر->context;
    برائے (size_t i = 0; i < buf_ctx->bufs.size(); i++) {
        ggml_backend_بفر_reset(ggml_backend_meta_بفر_simple_بفر(بفر, i));
    }
}

ساکن ثابت ggml_backend_بفر_i ggml_backend_meta_بفر_iface = {
    /* .free_buffer     = */ ggml_backend_meta_بفر_free_بفر,
    /* .get_base        = */ ggml_backend_meta_بفر_get_بنیاد,
    /* .init_tensor     = */ ggml_backend_meta_بفر_init_tensor,
    /* .memset_tensor   = */ ggml_backend_meta_بفر_memset_tensor,
    /* .set_tensor      = */ ggml_backend_meta_بفر_سیٹ_tensor,
    /* .get_tensor      = */ ggml_backend_meta_بفر_get_tensor,
    /* .set_tensor_2d   = */ خالی_اشاریہ,
    /* .get_tensor_2d   = */ خالی_اشاریہ,
    /* .cpy_tensor      = */ خالی_اشاریہ,
    /* .clear           = */ ggml_backend_meta_بفر_clear,
    /* .reset           = */ ggml_backend_meta_بفر_reset,
};

منطقی ggml_backend_بفر_is_meta(ggml_backend_بفر_t buf) {
    واپس buf != خالی_اشاریہ && buf->iface.free_بفر == ggml_backend_meta_بفر_iface.free_بفر;
}

خالی ggml_backend_meta_بفر_سیٹ_usage(ggml_backend_بفر_t بفر, شمار ggml_backend_بفر_usage usage) {
    GGML_ASSERT(ggml_backend_بفر_is_meta(بفر));
    ggml_backend_meta_بفر_context * buf_ctx = (ggml_backend_meta_بفر_context *) بفر->context;
    برائے (size_t i = 0; i < buf_ctx->bufs.size(); i++) {
        اگر (buf_ctx->bufs[i]) {
            ggml_backend_بفر_سیٹ_usage(buf_ctx->bufs[i].get(), usage);
        }
    }
}

ساکن ggml_backend_بفر_t ggml_backend_meta_بفر_type_alloc_بفر(ggml_backend_بفر_type_t buft, size_t size) {
    ثابت size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);

    ثابت ggml_init_params params = {
        /*.mem_size   =*/ 1024*1024*ggml_tensor_overhead(), // FIXME
        /*.mem_buffer =*/ خالی_اشاریہ,
        /*.no_alloc   =*/ سچ,
    };
    ggml_backend_meta_simple_tensor_container stc_static;
    ggml_backend_meta_simple_tensor_container stc_compute_0(params, n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_1(params, n_simple_bufts);

    size_t max_size = 0;
    std::vector<ggml_backend_بفر_t> bufs;
    bufs.reserve(n_simple_bufts);
    برائے (size_t i = 0; i < n_simple_bufts; i++) {
        bufs.push_back(ggml_backend_buft_alloc_بفر(ggml_backend_meta_buft_simple_buft(buft, i), size));
        GGML_ASSERT(bufs.back() != خالی_اشاریہ);
        max_size = std::max(max_size, ggml_backend_بفر_get_سائز(bufs.back()));
    }
    ggml_backend_meta_بفر_context * buf_ctx = نیا ggml_backend_meta_بفر_context(stc_static, stc_compute_0, stc_compute_1, bufs);

    واپس ggml_backend_بفر_init(buft, ggml_backend_meta_بفر_iface, buf_ctx, max_size);
}

ساکن ggml_backend_بفر_t ggml_backend_meta_بفر_type_alloc_بفر_n(ggml_backend_بفر_type_t buft, ggml_tensor ** tensors, صحیح n_tensors) {
    ثابت size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);

    constexpr size_t compute_headroom = 16; // Maximum number of views per statically allocated tensor that can be created between evals.
    ثابت ggml_init_params params_static = {
        /*.mem_size   =*/ n_tensors * ggml_tensor_overhead(),
        /*.mem_buffer =*/ خالی_اشاریہ,
        /*.no_alloc   =*/ سچ,
    };
    ثابت ggml_init_params params_compute = {
        /*.mem_size   =*/ compute_headroom * n_tensors * ggml_tensor_overhead(),
        /*.mem_buffer =*/ خالی_اشاریہ,
        /*.no_alloc   =*/ سچ,
    };
    ggml_backend_meta_simple_tensor_container stc_static   (params_static,  n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_0(params_compute, n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_1(params_compute, n_simple_bufts);

    std::vector<ggml_backend_بفر_t> bufs(n_simple_bufts, خالی_اشاریہ);
    ggml_backend_meta_بفر_context * meta_buf_ctx = نیا ggml_backend_meta_بفر_context(stc_static, stc_compute_0, stc_compute_1, bufs);

    ggml_backend_بفر_t meta_buf = ggml_backend_بفر_init(buft, ggml_backend_meta_بفر_iface, meta_buf_ctx, 0);
    برائے (صحیح i = 0; i < n_tensors; i++) {
        ggml_tensor * t = tensors[i];
        t->بفر = meta_buf;
        ggml_backend_meta_بفر_init_tensor_impl(meta_buf_ctx->stc_static, t);
        t->ڈیٹا = (خالی *) 0x2000000000000000; // FIXME
    }
    برائے (size_t i = 0; i < n_simple_bufts; i++) {
        ggml_context * ctx = meta_buf_ctx->stc_static.ctxs[i].get();
        ggml_backend_بفر_type_t simple_buft = ggml_backend_meta_buft_simple_buft(buft, i);

        // If a ggml_context only has zero-sized tensors, ggml_backend_alloc_ctx_tensors_from_buft returns NULL.
        // For those edge cases, allocate a dummy buffer instead.
        منطقی any_nonzero_قاش = جھوٹ;
        برائے (ggml_tensor * t = ggml_get_پہلا_tensor(ctx); t != خالی_اشاریہ; t = ggml_get_اگلا_tensor(ctx, t)) {
            اگر (ggml_nelements(t) != 0) {
                any_nonzero_قاش = سچ;
                توڑو;
            }
        }
        اگر (any_nonzero_قاش) {
            meta_buf_ctx->bufs[i].reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx, simple_buft));
        } ورنہ {
            meta_buf_ctx->bufs[i].reset(ggml_backend_buft_alloc_بفر(simple_buft, 0));
            برائے (ggml_tensor * t = ggml_get_پہلا_tensor(ctx); t != خالی_اشاریہ; t = ggml_get_اگلا_tensor(ctx, t)) {
                t->بفر = meta_buf_ctx->bufs[i].get();
            }
        }
        GGML_ASSERT(meta_buf_ctx->bufs[i]);
        meta_buf->size = std::max(meta_buf->size, ggml_backend_بفر_get_سائز(meta_buf_ctx->bufs[i].get()));
    }
    واپس meta_buf;
}

//
// meta backend
//

ساکن ggml_guid_t ggml_backend_meta_guid() {
    ساکن ggml_guid guid = {0xf1, 0x0e, 0x34, 0xcf, 0x9c, 0x6f, 0x43, 0xcb, 0x96, 0x92, 0xbe, 0x8e, 0xbb, 0x71, 0x3f, 0xda};
    واپس &guid;
}

ساخت ggml_backend_meta_context {
    ساخت cgraph_config {
        ggml_cgraph * cgraph_main = خالی_اشاریہ;
        صحیح           انحراف      = 0; // Node offset vs. original graph

        std::vector<ggml_cgraph *> cgraphs_aux;
    };
    ساخت backend_config {
        ggml_backend_t backend;

        std::vector<cgraph_config>           cgraphs;
        std::vector<ggml_tensor *>           nodes;
        std::vector<ggml_backend_بفر_ptr> bufs;

        backend_config(ggml_backend_t backend, ثابت size_t n_reduce_اقدام) : backend(backend) {
            bufs.resize(n_reduce_اقدام);
        }
    };
    std::string                 name;
    std::vector<backend_config> backend_configs;
    ggml_context_ptr            ctx;
    std::vector<ggml_cgraph *>  cgraphs_aux;
    std::vector<ggml_tensor *>  nodes_aux;
    size_t                      n_reduce_اقدام;
    صحیح                         max_nnodes    = 0;
    size_t                      max_tmp_سائز  = 0;
    size_t                      max_subgraphs = 0;
    size_t                      n_subgraphs   = 0;
    uint64_t                    uid           = 0;

    خالی *                               comm_ctx       = خالی_اشاریہ;
    ggml_backend_comm_allreduce_tensor_t comm_allreduce = خالی_اشاریہ;

    ggml_backend_meta_context(ggml_backend_dev_t meta_dev, ثابت حرف * params) {
        ثابت size_t n_devs = ggml_backend_meta_dev_n_devs(meta_dev);
        n_reduce_اقدام = std::ceil(std::log2(n_devs));
        name = "Meta(";
        std::vector<ggml_backend_t> simple_backends;
        backend_configs.reserve(n_devs);
        simple_backends.reserve(n_devs);
        برائے (size_t i = 0; i < n_devs; i++) {
            ggml_backend_dev_t simple_dev = ggml_backend_meta_dev_simple_dev(meta_dev, i);
            اگر (i > 0) {
                name += ",";
            }
            name += ggml_backend_dev_name(simple_dev);
            simple_backends.push_back(ggml_backend_dev_init(simple_dev, params));
            backend_configs.emplace_back(simple_backends.back(), n_reduce_اقدام);
        }
        name += ")";

        اگر (n_devs > 1) {
            ggml_backend_comm_init_t comm_init = (ggml_backend_comm_init_t) ggml_backend_reg_get_proc_پتہ(
                ggml_backend_dev_backend_reg(ggml_backend_get_device(simple_backends[0])), "ggml_backend_comm_init");
            اگر (comm_init != خالی_اشاریہ) {
                comm_ctx = comm_init(simple_backends.ڈیٹا(), simple_backends.size());
            }
        }
        اگر (comm_ctx != خالی_اشاریہ) {
            comm_allreduce = (ggml_backend_comm_allreduce_tensor_t)
                ggml_backend_reg_get_proc_پتہ(ggml_backend_dev_backend_reg(
                    ggml_backend_get_device(simple_backends[0])), "ggml_backend_comm_allreduce_tensor");
            GGML_ASSERT(comm_allreduce != خالی_اشاریہ);
        }
    }

    ~ggml_backend_meta_context() {
        اگر (comm_ctx != خالی_اشاریہ) {
            ggml_backend_comm_free_t comm_free = (ggml_backend_comm_free_t) ggml_backend_reg_get_proc_پتہ(
                ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_configs[0].backend)), "ggml_backend_comm_free");
            GGML_ASSERT(comm_free != خالی_اشاریہ);
            comm_free(comm_ctx);
        }
        برائے (خودکار & bc : backend_configs) {
            ggml_backend_free(bc.backend);
        }
    }
};

ساکن ثابت حرف * ggml_backend_meta_get_name(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    ثابت ggml_backend_meta_context * backend_ctx = (ثابت ggml_backend_meta_context *) backend->context;
    واپس backend_ctx->name.c_str();
}

ساکن خالی ggml_backend_meta_free(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;
    حذف backend_ctx;
    حذف backend;
}

ساکن خالی ggml_backend_meta_سیٹ_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, ثابت خالی * ڈیٹا, size_t انحراف, size_t size) {
    ثابت size_t n_backends = ggml_backend_meta_n_backends(backend);
    GGML_ASSERT(انحراف == 0);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    ثابت ggml_backend_meta_بانٹو_state بانٹو_state = ggml_backend_meta_get_بانٹو_state(tensor, /*assume_sync =*/ جھوٹ);
    GGML_ASSERT(بانٹو_state.n_segments == 1);
    GGML_ASSERT(بانٹو_state.nr[0]      == 1);

    انتخاب (بانٹو_state.axis) {
        حالت GGML_BACKEND_بانٹو_AXIS_0:
        حالت GGML_BACKEND_بانٹو_AXIS_1:
        حالت GGML_BACKEND_بانٹو_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            ثابت size_t chunk_سائز_بھرا = tensor->nb[بانٹو_state.axis + 1];
            GGML_ASSERT(انحراف % chunk_سائز_بھرا == 0);
            GGML_ASSERT(size   % chunk_سائز_بھرا == 0);
            ثابت int64_t i_آغاز =  انحراف        /chunk_سائز_بھرا;
            ثابت int64_t i_stop  = (انحراف + size)/chunk_سائز_بھرا;
            size_t انحراف_j = 0;
            برائے (size_t j = 0; j < n_backends; j++){
                ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
                ggml_tensor * simple_tensor = ggml_backend_meta_بفر_simple_tensor(tensor, j);
                ثابت size_t chunk_سائز_j = simple_tensor->nb[بانٹو_state.axis + 1];
                اگر (chunk_سائز_j == 0) {
                    جاری;
                }
                ggml_backend_tensor_سیٹ_2d_async(simple_backend, simple_tensor, (ثابت حرف *) ڈیٹا + انحراف_j, انحراف, chunk_سائز_j,
                    i_stop - i_آغاز, chunk_سائز_j, chunk_سائز_بھرا);
                انحراف_j += chunk_سائز_j;
            }
            GGML_ASSERT(انحراف_j == chunk_سائز_بھرا);
        } توڑو;
        حالت GGML_BACKEND_بانٹو_AXIS_MIRRORED: {
            برائے (size_t j = 0; j < n_backends; j++) {
                ggml_backend_tensor_سیٹ_async(
                    ggml_backend_meta_simple_backend(backend, j), ggml_backend_meta_بفر_simple_tensor(tensor, j), ڈیٹا, انحراف, size);
            }
        } توڑو;
        طےشدہ: {
            GGML_ABORT("fatal error");
        }
    }
}

ساکن خالی ggml_backend_meta_get_tensor_async(ggml_backend_t backend, ثابت ggml_tensor * tensor, خالی * ڈیٹا, size_t انحراف, size_t size) {
    ثابت size_t n_backends = ggml_backend_meta_n_backends(backend);
    GGML_ASSERT(انحراف == 0);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    ثابت ggml_backend_meta_بانٹو_state بانٹو_state = ggml_backend_meta_get_بانٹو_state(tensor, /*assume_sync =*/ جھوٹ);
    GGML_ASSERT(بانٹو_state.n_segments == 1);
    GGML_ASSERT(بانٹو_state.nr[0]      == 1);

    انتخاب (بانٹو_state.axis) {
        حالت GGML_BACKEND_بانٹو_AXIS_0:
        حالت GGML_BACKEND_بانٹو_AXIS_1:
        حالت GGML_BACKEND_بانٹو_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            ثابت size_t chunk_سائز_بھرا = tensor->nb[بانٹو_state.axis + 1];
            GGML_ASSERT(انحراف % chunk_سائز_بھرا == 0);
            GGML_ASSERT(size   % chunk_سائز_بھرا == 0);
            ثابت int64_t i_آغاز =  انحراف        /chunk_سائز_بھرا;
            ثابت int64_t i_stop  = (انحراف + size)/chunk_سائز_بھرا;
            size_t انحراف_j = 0;
            برائے (size_t j = 0; j < n_backends; j++){
                ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
                ثابت ggml_tensor * simple_tensor = ggml_backend_meta_بفر_simple_tensor(tensor, j);
                ثابت size_t chunk_سائز_j = simple_tensor->nb[بانٹو_state.axis + 1];
                اگر (chunk_سائز_j == 0) {
                    جاری;
                }
                ggml_backend_tensor_get_2d_async(simple_backend, simple_tensor, (حرف *) ڈیٹا + انحراف_j, انحراف, chunk_سائز_j,
                    i_stop - i_آغاز, chunk_سائز_j, chunk_سائز_بھرا);
                انحراف_j += chunk_سائز_j;
            }
            GGML_ASSERT(انحراف_j == chunk_سائز_بھرا);
        } توڑو;
        حالت GGML_BACKEND_بانٹو_AXIS_MIRRORED: {
            // TODO other simple backend may be better
            ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, 0);
            ثابت ggml_tensor * simple_tensor = ggml_backend_meta_بفر_simple_tensor(tensor, 0);
            ggml_backend_tensor_get_async(simple_backend, simple_tensor, ڈیٹا, انحراف, size);
        } توڑو;
        طےشدہ: {
            GGML_ABORT("fatal error");
        }
    }
}

ساکن خالی ggml_backend_meta_synchronize(ggml_backend_t backend) {
    ثابت size_t n_backends = ggml_backend_meta_n_backends(backend);
    برائے (size_t i = 0; i < n_backends; i++) {
        ggml_backend_synchronize(ggml_backend_meta_simple_backend(backend, i));
    }
}

ساکن شمار ggml_حالت ggml_backend_meta_گراف_compute(ggml_backend_t backend, ساخت ggml_cgraph * cgraph) {
    GGML_ASSERT(cgraph->grads == خالی_اشاریہ);
    ثابت size_t n_backends = ggml_backend_meta_n_backends(backend);
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;

    // If the previous cgraph had a defined UID it can be used to skip rebuilding the subgraphs per simple backend.
    ثابت منطقی needs_rebuild = (cgraph->uid == 0) || (cgraph->uid != backend_ctx->uid);

    منطقی max_nnodes_raised = جھوٹ;
    اگر (cgraph->n_nodes > backend_ctx->max_nnodes) {
        برائے (size_t j = 0; j < n_backends; j++) {
            خودکار & bcj = backend_ctx->backend_configs[j];
            bcj.nodes.resize(cgraph->n_nodes);
            bcj.cgraphs.resize(cgraph->n_nodes);
        }
        backend_ctx->max_nnodes = cgraph->n_nodes;
        max_nnodes_raised = سچ;
        أكد(needs_rebuild);
    }

    اگر (needs_rebuild) {
        std::set<ggml_backend_بفر_t> مستعمل_buffers;
        برائے (صحیح i = 0; i < cgraph->n_leafs; i++) {
            اگر (ggml_backend_بفر_is_meta(cgraph->leafs[i]->بفر)) {
                مستعمل_buffers.emplace(cgraph->leafs[i]->بفر);
            }
        }
        برائے (صحیح i = 0; i < cgraph->n_nodes; i++) {
            اگر (ggml_backend_بفر_is_meta(cgraph->nodes[i]->بفر)) {
                مستعمل_buffers.emplace(cgraph->nodes[i]->بفر);
            }
        }
        برائے (ggml_backend_بفر_t buf : مستعمل_buffers) {
            ggml_backend_meta_بفر_context * buf_ctx = (ggml_backend_meta_بفر_context *) buf->context;
            buf_ctx->stc_compute_اشاریہ_اگلا = buf_ctx->stc_compute_اشاریہ ^ 1;
            ggml_backend_meta_simple_tensor_container & stc = buf_ctx->stc_compute[buf_ctx->stc_compute_اشاریہ_اگلا];
            برائے (ggml_context_ptr & ctx : stc.ctxs) {
                ggml_reset(ctx.get());
            }
            stc.simple_tensors.clear();
        }
        size_t n_subgraphs  = 0;
        size_t max_tmp_سائز = 0;

        برائے (size_t j = 0; j < n_backends; j++) {
            خودکار & bcj = backend_ctx->backend_configs[j];

            برائے (صحیح i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * گرہ = cgraph->nodes[i];
                اگر (!ggml_backend_بفر_is_meta(گرہ->بفر)) {
                    // [TAG_META_HOST_VIEWS]
                    GGML_ASSERT(ggml_backend_meta_is_host_view(گرہ));

                    // keep the node as is, mapping it to a simple tensor is not possible
                    bcj.nodes[i] = گرہ;
                    جاری;
                }
                bcj.nodes[i] = ggml_backend_meta_بفر_simple_tensor(گرہ, j);
                GGML_ASSERT(bcj.nodes[i]);
            }
        }

        {
            // For MoE models it may make sense to delay the AllReduce in order to reduce I/O:
            خودکار get_i_delayed_branch = [&](ثابت صحیح i) -> صحیح {
                صحیح id = i; // i_delayed
                صحیح idr = i; // i_delayed return, last safe return value

                ggml_tensor * گرہ = cgraph->nodes[id];
                int32_t n_مستعمل = ggml_گرہ_get_use_گنتی(cgraph, id);

                // Skip MIRRORED nodes that don't consume node
                خودکار skip_unrelated = [&]() {
                    جب (id + 1 < cgraph->n_nodes) {
                        ggml_tensor * اگلا = cgraph->nodes[id+1];
                        اگر (ggml_backend_meta_get_بانٹو_state(اگلا, جھوٹ).axis != GGML_BACKEND_بانٹو_AXIS_MIRRORED) {
                            توڑو;
                        }
                        منطقی محفوظ_2 = سچ;
                        برائے (صحیح s = 0; s < GGML_MAX_SRC; s++) {
                            اگر (اگلا->src[s] == خالی_اشاریہ) {
                                جاری;
                            }
                            اگر (اگلا->src[s] == گرہ) {
                                محفوظ_2 = جھوٹ;
                                توڑو;
                            }
                            اگر (ggml_backend_meta_get_بانٹو_state(اگلا->src[s], جھوٹ).axis != GGML_BACKEND_بانٹو_AXIS_MIRRORED) {
                                محفوظ_2 = جھوٹ;
                                توڑو;
                            }
                        }
                        اگر (!محفوظ_2) {
                            توڑو;
                        }
                        id++;
                    }
                };

                skip_unrelated();
                اگر (id + 1 >= cgraph->n_nodes) {
                    واپس idr;
                }
                {
                    ggml_tensor * اگلا = cgraph->nodes[id+1];
                    اگر (اگلا->op == GGML_OP_جوڑو_ID && اگلا->src[0] == گرہ &&
                            ggml_backend_meta_get_بانٹو_state(اگلا->src[1], جھوٹ).axis == GGML_BACKEND_بانٹو_AXIS_PARTIAL &&
                            ggml_backend_meta_get_بانٹو_state(اگلا->src[2], جھوٹ).axis == GGML_BACKEND_بانٹو_AXIS_MIRRORED) {
                        گرہ = اگلا;
                        id++;
                        idr = id;
                        n_مستعمل = ggml_گرہ_get_use_گنتی(cgraph, id);
                    }
                }
                // Chain of MULs with MIRRORED src[1]
                جب (سچ) {
                    skip_unrelated();
                    اگر (id + 1 >= cgraph->n_nodes) {
                        واپس idr;
                    }
                    ggml_tensor * اگلا = cgraph->nodes[id+1];
                    اگر (اگلا->op == GGML_OP_MUL && اگلا->src[0] == گرہ &&
                            ggml_backend_meta_get_بانٹو_state(اگلا->src[1], جھوٹ).axis == GGML_BACKEND_بانٹو_AXIS_MIRRORED) {
                        گرہ = اگلا;
                        id++;
                        idr = id;
                        n_مستعمل = ggml_گرہ_get_use_گنتی(cgraph, id);
                    } ورنہ {
                        توڑو;
                    }
                }

                اگر (n_مستعمل != گرہ->ne[1] || id + 2*n_مستعمل-1 >= cgraph->n_nodes) {
                    واپس idr;
                }
                برائے (int32_t k = 0; k < n_مستعمل; k++) {
                    ggml_tensor * اگلا = cgraph->nodes[id+1];
                    اگر (اگلا->op != GGML_OP_VIEW || اگلا->view_src != گرہ || اگلا->view_offs != k*گرہ->nb[1] ||
                            اگلا->ne[0] != گرہ->ne[0] || اگلا->ne[1] != گرہ->ne[2] || اگلا->nb[1] != گرہ->nb[2] ||
                            ggml_گرہ_get_use_گنتی(cgraph, id+1) != 1) {
                        واپس idr;
                    }
                    id++;
                }
                {
                    ggml_tensor * اگلا = cgraph->nodes[id+1];
                    اگر (اگلا->op != GGML_OP_جوڑو || اگلا->src[0] != cgraph->nodes[id - (n_مستعمل-1)] ||
                            اگلا->src[1] != cgraph->nodes[id - (n_مستعمل-2)] || ggml_گرہ_get_use_گنتی(cgraph, id+1) != 1) {
                        واپس idr;
                    }
                    id++;
                }
                برائے (int32_t k = 0; k < n_مستعمل - 2; k++) {
                    ggml_tensor * اگلا = cgraph->nodes[id+1];
                    اگر (اگلا->op != GGML_OP_جوڑو || اگلا->src[0] != cgraph->nodes[id] ||
                            اگلا->src[1] != cgraph->nodes[id - (n_مستعمل-2)] || ggml_گرہ_get_use_گنتی(cgraph, id+1) != 1) {
                        واپس idr;
                    }
                    id++;
                }
                idr = id;
                واپس idr;
            };

            // AllReduce(a) + AllReduce(b) == AllReduce(a + b) for independent partial branches.
            خودکار get_i_delayed = [&](ثابت صحیح i) -> صحیح {
                ثابت صحیح i_delayed = get_i_delayed_branch(i);
                ggml_tensor * گرہ = cgraph->nodes[i_delayed];

                اگر (ggml_گرہ_get_use_گنتی(cgraph, i_delayed) != 1) {
                    واپس i_delayed;
                }

                برائے (صحیح id = i_delayed + 1; id < cgraph->n_nodes; id++) {
                    ggml_tensor * اگلا = cgraph->nodes[id];
                    اگر (اگلا->view_src == گرہ) {
                        واپس i_delayed;
                    }
                    برائے (صحیح s = 0; s < GGML_MAX_SRC; s++) {
                        اگر (اگلا->src[s] == گرہ) {
                            واپس i_delayed;
                        }
                    }

                    اگر (اگلا->view_src != خالی_اشاریہ && اگلا->view_src->op == GGML_OP_NONE && ggml_backend_بفر_is_host(اگلا->view_src->بفر)) {
                        جاری;
                    }
                    اگر (ggml_backend_meta_get_بانٹو_state(اگلا, جھوٹ).axis != GGML_BACKEND_بانٹو_AXIS_PARTIAL) {
                        جاری;
                    }

                    ثابت صحیح i_other = id;
                    ثابت صحیح i_other_delayed = get_i_delayed_branch(i_other);
                    ggml_tensor * other = cgraph->nodes[i_other_delayed];
                    اگر (ggml_گرہ_get_use_گنتی(cgraph, i_other_delayed) != 1 || i_other_delayed + 1 >= cgraph->n_nodes) {
                        واپس i_delayed;
                    }

                    ggml_tensor * مجموع = cgraph->nodes[i_other_delayed + 1];
                    اگر (مجموع->op != GGML_OP_جوڑو ||
                            !ggml_are_same_shape(گرہ, other) || گرہ->type != other->type || مجموع->type != گرہ->type ||
                            !((مجموع->src[0] == گرہ && مجموع->src[1] == other) ||
                              (مجموع->src[0] == other && مجموع->src[1] == گرہ)) ||
                            ggml_backend_meta_get_بانٹو_state(مجموع, جھوٹ).axis != GGML_BACKEND_بانٹو_AXIS_MIRRORED) {
                        واپس i_delayed;
                    }

                    برائے (size_t j = 0; j < n_backends; j++) {
                        خودکار & bcj = backend_ctx->backend_configs[j];
                        ثابت منطقی compute       = bcj.nodes[i]->flags       & GGML_TENSOR_FLAG_COMPUTE;
                        ثابت منطقی compute_other = bcj.nodes[i_other]->flags & GGML_TENSOR_FLAG_COMPUTE;
                        اگر (compute != compute_other) {
                            واپس i_delayed;
                        }
                    }
                    واپس i_other_delayed + 1;
                }
                واپس i_delayed;
            };

            صحیح i_آغاز = 0;
            برائے (صحیح i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * گرہ = cgraph->nodes[i];
                اگر (گرہ->view_src != خالی_اشاریہ && گرہ->view_src->op == GGML_OP_NONE && ggml_backend_بفر_is_host(گرہ->view_src->بفر)) {
                    جاری;
                }
                ثابت ggml_backend_meta_بانٹو_state بانٹو_state = ggml_backend_meta_get_بانٹو_state(گرہ, /*assume_sync =*/ جھوٹ);
                اگر (بانٹو_state.axis == GGML_BACKEND_بانٹو_AXIS_PARTIAL) {
                    max_tmp_سائز = std::max(max_tmp_سائز, ggml_nbytes(گرہ));
                }
                ثابت منطقی new_subgraph = i + 1 == cgraph->n_nodes || بانٹو_state.axis == GGML_BACKEND_بانٹو_AXIS_PARTIAL;
                اگر (!new_subgraph) {
                    جاری;
                }

                ثابت صحیح i_delayed = get_i_delayed(i);

                // If we can delay the AllReduce we need to consider the interaction with zero-sized tensor slices.
                // A backend with such a slice would normally have valid data after participating in the AllReduce with a node that has
                //     its compute flag disabled and thus gets its data zeroed out.
                // If the AllReduce is delayed then the nodes until that point also need to have their compute flag disabled.
                اگر (i_delayed > i) {
                    برائے (size_t j = 0; j < n_backends; j++) {
                        خودکار & bcj = backend_ctx->backend_configs[j];
                        اگر ((bcj.nodes[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                            برائے (صحیح ii = i + 1; ii <= i_delayed; ii++) {
                                bcj.nodes[ii]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
                            }
                        }
                    }
                }

                i = i_delayed;

                برائے (size_t j = 0; j < n_backends; j++) {
                    خودکار & bcj = backend_ctx->backend_configs[j];
                    bcj.cgraphs[n_subgraphs].انحراف = i_آغاز;
                }
                n_subgraphs++;
                i_آغاز = i + 1;
            }
            GGML_ASSERT(i_آغاز == cgraph->n_nodes);
        }

        backend_ctx->uid         = cgraph->uid;
        backend_ctx->n_subgraphs = n_subgraphs;

        اگر (max_tmp_سائز > backend_ctx->max_tmp_سائز) {
            برائے (size_t j = 0; j < n_backends; j++) {
                خودکار & bcj = backend_ctx->backend_configs[j];
                برائے (size_t i = 0; i < backend_ctx->n_reduce_اقدام; i++) {
                    bcj.bufs[i].reset(ggml_backend_alloc_بفر(bcj.backend, max_tmp_سائز));
                }
            }
            backend_ctx->max_tmp_سائز = max_tmp_سائز;
        }

        اگر (max_nnodes_raised || n_subgraphs > backend_ctx->max_subgraphs) {
            backend_ctx->max_subgraphs = std::max(backend_ctx->max_subgraphs, n_subgraphs);
            ثابت size_t n_nodes_فی_device = 3 * backend_ctx->n_reduce_اقدام; // tmp + ADD (+zeroing) graph per step and device
            ثابت size_t n_cgraphs_فی_device = 2 * backend_ctx->n_reduce_اقدام; // ADD ( + zeroing) graph per step and device
            ثابت size_t mem_فی_device_graphs_main = backend_ctx->max_subgraphs*ggml_گراف_overhead_custom(backend_ctx->max_nnodes, cgraph->grads);
            ثابت size_t mem_فی_device_graphs_aux = n_cgraphs_فی_device*backend_ctx->max_subgraphs*ggml_گراف_overhead_custom(1, cgraph->grads);
            ثابت size_t mem_فی_device_nodes_aux = n_nodes_فی_device*backend_ctx->max_subgraphs*ggml_tensor_overhead();
            ثابت ggml_init_params params = {
                /*.mem_size   =*/ n_backends * (mem_فی_device_graphs_main + mem_فی_device_graphs_aux + mem_فی_device_nodes_aux),
                /*.mem_buffer =*/ خالی_اشاریہ,
                /*.no_alloc   =*/ سچ,
            };
            backend_ctx->ctx.reset(ggml_init(params));
            برائے (size_t j = 0; j < n_backends; j++) {
                خودکار & bcj = backend_ctx->backend_configs[j];
                برائے (size_t i = 0; i < backend_ctx->max_subgraphs; i++) {
                    bcj.cgraphs[i].cgraph_main = ggml_new_گراف_custom(backend_ctx->ctx.get(), backend_ctx->max_nnodes, /*grads =*/ جھوٹ);
                }
            }
            backend_ctx->cgraphs_aux.resize(n_backends*n_cgraphs_فی_device*backend_ctx->max_subgraphs);
            برائے (size_t k = 0; k < backend_ctx->cgraphs_aux.size(); k++) {
                backend_ctx->cgraphs_aux[k] = ggml_new_گراف_custom(backend_ctx->ctx.get(), 1, cgraph->grads);
            }
            backend_ctx->nodes_aux.resize(n_backends*n_nodes_فی_device*backend_ctx->max_subgraphs);
            برائے (size_t k = 0; k < backend_ctx->nodes_aux.size(); k++) {
                backend_ctx->nodes_aux[k] = ggml_new_tensor_1d(backend_ctx->ctx.get(), GGML_TYPE_F32, 1);
            }
        }

        برائے (size_t j = 0; j < n_backends; j++) {
            خودکار & bcj = backend_ctx->backend_configs[j];
            برائے (size_t i_گراف = 0; i_گراف < n_subgraphs; i_گراف++) {
                ggml_cgraph * cgraph_ij = bcj.cgraphs[i_گراف].cgraph_main;
                ثابت size_t i_گرہ_آغاز = bcj.cgraphs[i_گراف].انحراف;
                ثابت size_t i_گرہ_stop = i_گراف + 1 < n_subgraphs ? bcj.cgraphs[i_گراف + 1].انحراف : cgraph->n_nodes;
                cgraph_ij->n_nodes = i_گرہ_stop - i_گرہ_آغاز;
                ggml_hash_سیٹ_reset(&cgraph_ij->دیکھا_hash_سیٹ);
                برائے (size_t i_گرہ = i_گرہ_آغاز; i_گرہ < i_گرہ_stop; i_گرہ++) {
                    ggml_tensor * گرہ_ij = bcj.nodes[i_گرہ];
                    cgraph_ij->nodes[i_گرہ - i_گرہ_آغاز] = گرہ_ij;
                    ثابت size_t hash_pos_orig = ggml_hash_ڈھونڈو(&cgraph->دیکھا_hash_سیٹ, cgraph->nodes[i_گرہ]);
                    ثابت size_t hash_pos_ij = ggml_hash_داخل_کرو(&cgraph_ij->دیکھا_hash_سیٹ, گرہ_ij);
                    cgraph_ij->use_counts[hash_pos_ij] = cgraph->use_counts[hash_pos_orig];
                }
                cgraph_ij->uid = ggml_گراف_اگلا_uid();
            }
        }
    }

    size_t iga = 0; // i graph aux
    size_t ina = 0; // i node aux

    خودکار get_گرہ_aux = [&](ggml_tensor * t) -> ggml_tensor * {
        ggml_tensor * ret = backend_ctx->nodes_aux[ina++];
        memset(ret, 0, حجم(ggml_tensor));
        ret->op   = GGML_OP_NONE;
        ret->type = t->type;
        برائے (size_t k = 0; k < GGML_MAX_DIMS; k++) {
            ret->ne[k] = t->ne[k];
            ret->nb[k] = t->nb[k];
        }
        واپس ret;
    };
    خودکار سیٹ_tmp_ڈیٹا = [&](ggml_tensor * tensor, ثابت size_t j, ثابت size_t i_buf) {
        خودکار & bcj = backend_ctx->backend_configs[j];
        ggml_backend_بفر_ptr & buf_ptr = bcj.bufs[i_buf];
        اگر (!buf_ptr || ggml_backend_بفر_get_سائز(buf_ptr.get()) < backend_ctx->max_tmp_سائز) {
            buf_ptr.reset(ggml_backend_alloc_بفر(bcj.backend, backend_ctx->max_tmp_سائز));
        }
        tensor->بفر = buf_ptr.get();
        tensor->ڈیٹا   = ggml_backend_بفر_get_بنیاد(buf_ptr.get());
    };
    // FIXME usage_counts
    خودکار get_cgraph_aux = [&]() -> ggml_cgraph * {
        ggml_cgraph * ret = backend_ctx->cgraphs_aux[iga++];
        واپس ret;
    };

    // Preferentially use backend-specific allreduce_tensor_async (e.g. NCCL for CUDA), use a generic fallback if unavailable:
    خودکار allreduce_fallback = [&](size_t i) -> ggml_حالت {
        std::vector<ggml_cgraph *> قدم_cgraphs(n_backends, خالی_اشاریہ);

        // Zero out nodes that were disabled due to having a zero-sized slice:
        برائے (size_t j = 0; j < n_backends; j++) {
            خودکار & bcj = backend_ctx->backend_configs[j];
            ggml_tensor * گرہ = bcj.cgraphs[i].cgraph_main->nodes[bcj.cgraphs[i].cgraph_main->n_nodes - 1];
            اگر (گرہ->flags & GGML_TENSOR_FLAG_COMPUTE) {
                جاری;
            }
            اگر (ggml_nelements(گرہ) > 0) {
                ggml_tensor * گرہ_zero = get_گرہ_aux(گرہ);
                گرہ_zero->op = GGML_OP_FILL;
                گرہ_zero->src[0] = گرہ; // only used for the shape, the data is not read
                ggml_سیٹ_op_params_f32(گرہ_zero, 0, 0.0f);
                گرہ_zero->ڈیٹا = گرہ->ڈیٹا;
                گرہ_zero->بفر = گرہ->بفر;
                گرہ_zero->flags |= GGML_TENSOR_FLAG_COMPUTE;

                قدم_cgraphs[j] = get_cgraph_aux();
                قدم_cgraphs[j]->nodes[0] = گرہ_zero;
                قدم_cgraphs[j]->n_nodes = 1;
                ثابت ggml_حالت حالت_2 = ggml_backend_گراف_compute_async(bcj.backend, قدم_cgraphs[j]);
                اگر (حالت_2 != GGML_حالت_SUCCESS) {
                    واپس حالت_2;
                }
            }
        }
        std::fill(قدم_cgraphs.begin(), قدم_cgraphs.end(), خالی_اشاریہ);

        خودکار ڈالو_ڈیٹا = [&](ثابت size_t j_src, ثابت size_t j_dst, ثابت size_t i_buf) {
            أكد(قدم_cgraphs[j_dst] == خالی_اشاریہ);
            خودکار & bcj_src = backend_ctx->backend_configs[j_src];
            خودکار & bcj_dst = backend_ctx->backend_configs[j_dst];

            ggml_tensor * گرہ_src = bcj_src.cgraphs[i].cgraph_main->nodes[bcj_src.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_tensor * گرہ_dst = bcj_dst.cgraphs[i].cgraph_main->nodes[bcj_dst.cgraphs[i].cgraph_main->n_nodes - 1];
            GGML_ASSERT(ggml_is_contiguous(گرہ_src));
            GGML_ASSERT(ggml_is_contiguous(گرہ_dst));

            ggml_tensor * گرہ_tmp = get_گرہ_aux(گرہ_dst);
            سیٹ_tmp_ڈیٹا(گرہ_tmp, j_dst, i_buf);

            ggml_backend_tensor_copy_async(bcj_src.backend, bcj_dst.backend, گرہ_src, گرہ_tmp);

            ggml_tensor * گرہ_red = get_گرہ_aux(گرہ_dst);
            گرہ_red->view_src = گرہ_dst->view_src == خالی_اشاریہ ? گرہ_dst : گرہ_dst->view_src;
            گرہ_red->view_offs = گرہ_dst->view_offs;
            گرہ_red->op = GGML_OP_جوڑو;
            گرہ_red->src[0] = گرہ_dst;
            گرہ_red->src[1] = گرہ_tmp;
            گرہ_red->flags |= GGML_TENSOR_FLAG_COMPUTE;
            ggml_backend_view_init(گرہ_red);

            ggml_cgraph * cgraph_aux = get_cgraph_aux();
            cgraph_aux->nodes[0] = گرہ_red;
            cgraph_aux->n_nodes = 1;
            قدم_cgraphs[j_dst] = cgraph_aux;
        };

        size_t انحراف_j = n_backends/2;
        جب ((انحراف_j & (انحراف_j - 1)) != 0) {
            انحراف_j--;
        }
        ثابت size_t انحراف_j_max = انحراف_j;
        size_t i_buf = 0;

        // If n_backends is not a power of 2, fold in the excess prior to butterfly reduction:
        برائے (size_t j_src = 2*انحراف_j_max; j_src < n_backends; j_src++) {
            ثابت size_t j_dst = j_src - 2*انحراف_j_max;
            ڈالو_ڈیٹا(j_src, j_dst, i_buf);
            ثابت ggml_حالت حالت_2 = ggml_backend_گراف_compute_async(backend_ctx->backend_configs[j_dst].backend, قدم_cgraphs[j_dst]);
            اگر (حالت_2 != GGML_حالت_SUCCESS) {
                واپس حالت_2;
            }
            i_buf = 1;
        }

        // Butterfly reduction:
        برائے (; انحراف_j >= 1; انحراف_j /= 2) {
            std::fill(قدم_cgraphs.begin(), قدم_cgraphs.end(), خالی_اشاریہ);

            برائے (size_t j = 0; j < 2*انحراف_j_max; j++) {
                ثابت size_t j_other = j ^ انحراف_j;
                اگر (j_other >= n_backends) {
                    جاری;
                }
                ڈالو_ڈیٹا(j, j_other, i_buf);
            }

            برائے (size_t j = 0; j < 2*انحراف_j_max; j++) {
                اگر (قدم_cgraphs[j] == خالی_اشاریہ) {
                    جاری;
                }
                خودکار & bcj = backend_ctx->backend_configs[j];
                ثابت ggml_حالت حالت_2 = ggml_backend_گراف_compute_async(bcj.backend, قدم_cgraphs[j]);
                اگر (حالت_2 != GGML_حالت_SUCCESS) {
                    واپس حالت_2;
                }
            }
            i_buf++;
        }
        أكد(i_buf == backend_ctx->n_reduce_اقدام);

        // If n_backends is not a power of 2, copy back the reduced tensors to the excess:
        برائے (size_t j = 2*انحراف_j_max; j < n_backends; j++) {
            خودکار & bcj_src = backend_ctx->backend_configs[j - 2*انحراف_j_max];
            خودکار & bcj_dst = backend_ctx->backend_configs[j];

            ggml_tensor * گرہ_src = bcj_src.cgraphs[i].cgraph_main->nodes[bcj_src.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_tensor * گرہ_dst = bcj_dst.cgraphs[i].cgraph_main->nodes[bcj_dst.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_backend_tensor_copy_async(bcj_src.backend, bcj_dst.backend, گرہ_src, گرہ_dst);
        }

        واپس GGML_حالت_SUCCESS;
    };


    برائے (size_t i = 0; i < backend_ctx->n_subgraphs; i++) {
        برائے (size_t j = 0; j < n_backends; j++) {
            خودکار & bcj = backend_ctx->backend_configs[j];
            ثابت ggml_حالت حالت_2 = ggml_backend_گراف_compute_async(bcj.backend, bcj.cgraphs[i].cgraph_main);
            اگر (حالت_2 != GGML_حالت_SUCCESS) {
                واپس حالت_2;
            }
        }

        اگر (n_backends > 1 && i < backend_ctx->n_subgraphs - 1) {
            منطقی backend_allreduce_success = جھوٹ;
            اگر (backend_ctx->comm_ctx) {
                std::vector<ggml_tensor *> nodes;
                nodes.reserve(n_backends);
                برائے (size_t j = 0; j < n_backends; j++) {
                    خودکار & bcj = backend_ctx->backend_configs[j];
                    ggml_cgraph * cgraph_ij = bcj.cgraphs[i].cgraph_main;
                    nodes.push_back(cgraph_ij->nodes[cgraph_ij->n_nodes-1]);
                }
                backend_allreduce_success = backend_ctx->comm_allreduce(backend_ctx->comm_ctx, nodes.ڈیٹا());
            }

            اگر (!backend_allreduce_success) {
                ثابت ggml_حالت حالت_2 = allreduce_fallback(i);
                اگر (حالت_2 != GGML_حالت_SUCCESS) {
                    واپس حالت_2;
                }
            }
        }
    }
    واپس GGML_حالت_SUCCESS;
}

ساکن ثابت ggml_backend_i ggml_backend_meta_i = {
    /* .get_name                = */ ggml_backend_meta_get_name,
    /* .free                    = */ ggml_backend_meta_free,
    /* .set_tensor_async        = */ ggml_backend_meta_سیٹ_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_meta_get_tensor_async,
    /* .set_tensor_2d_async     = */ خالی_اشاریہ,
    /* .get_tensor_2d_async     = */ خالی_اشاریہ,
    /* .cpy_tensor_async        = */ خالی_اشاریہ,
    /* .synchronize             = */ ggml_backend_meta_synchronize,
    /* .graph_plan_create       = */ خالی_اشاریہ,
    /* .graph_plan_free         = */ خالی_اشاریہ,
    /* .graph_plan_update       = */ خالی_اشاریہ,
    /* .graph_plan_compute      = */ خالی_اشاریہ,
    /* .graph_compute           = */ ggml_backend_meta_گراف_compute,
    /* .event_record            = */ خالی_اشاریہ,
    /* .event_wait              = */ خالی_اشاریہ,
    /* .graph_optimize          = */ خالی_اشاریہ,
};

منطقی ggml_backend_is_meta(ggml_backend_t backend) {
    واپس backend != خالی_اشاریہ && backend->iface.get_name == ggml_backend_meta_i.get_name;
}

ساکن ggml_backend_t ggml_backend_meta_device_init_backend(ggml_backend_dev_t dev, ثابت حرف * params) {
    ggml_backend_meta_context * backend_ctx = نیا ggml_backend_meta_context(dev, params);

    ggml_backend_t backend = نیا ساخت ggml_backend;
    backend->guid    = ggml_backend_meta_guid();
    backend->iface   = ggml_backend_meta_i;
    backend->device  = dev;
    backend->context = backend_ctx;
    واپس backend;
}

size_t ggml_backend_meta_n_backends(ggml_backend_t meta_backend) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    ثابت ggml_backend_meta_context * backend_ctx = (ثابت ggml_backend_meta_context *) meta_backend->context;
    واپس backend_ctx->backend_configs.size();
}

ggml_backend_t ggml_backend_meta_simple_backend(ggml_backend_t meta_backend, size_t اشاریہ) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    ثابت ggml_backend_meta_context * backend_ctx = (ثابت ggml_backend_meta_context *) meta_backend->context;
    واپس backend_ctx->backend_configs[اشاریہ].backend;
}
