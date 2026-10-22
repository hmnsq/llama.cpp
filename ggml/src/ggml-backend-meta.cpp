#ضمن "ggml.h"
#ضمن "ggml-impl.h"
#ضمن "ggml-backend.h"
#ضمن "ggml-backend-impl.h"
#ضمن "ggml-alloc.h"
#ضمن "ggml-cpp.h"

#ضمن <algorithm>
#ضمن <cassert>
#ضمن <cmath>
#ضمن <cstddef>
#ضمن <cstdint>
#ضمن <cstring>
#ضمن <map>
#ضمن <memory>
#ضمن <set>
#ضمن <string>
#ضمن <tuple>
#ضمن <utility>
#ضمن <vector>

بنية ggml_backend_meta_device;
بنية ggml_backend_meta_بفر_type;
بنية ggml_backend_meta_بفر;
بنية ggml_backend_meta;

ثابت حرف * ggml_backend_meta_ونڈو_axis_name(تعداد ggml_backend_meta_ونڈو_axis ونڈو_axis) {
    بدل (ونڈو_axis) {
        حالة GGML_BACKEND_ونڈو_AXIS_0:
            أرجع "0";
        حالة GGML_BACKEND_ونڈو_AXIS_1:
            أرجع "1";
        حالة GGML_BACKEND_ونڈو_AXIS_2:
            أرجع "2";
        حالة GGML_BACKEND_ونڈو_AXIS_3:
            أرجع "3";
        حالة GGML_BACKEND_ونڈو_AXIS_MIRRORED:
            أرجع "MIRRORED";
        حالة GGML_BACKEND_ونڈو_AXIS_PARTIAL:
            أرجع "PARTIAL";
        حالة GGML_BACKEND_ونڈو_AXIS_NONE:
            أرجع "NONE";
        حالة GGML_BACKEND_ونڈو_AXIS_UNKNOWN:
            أرجع "UNKNOWN";
        افتراضي:
            GGML_ABORT("fatal error");
    }
}

//
// meta backend device
//

بنية ggml_backend_meta_device_context {
    std::vector<ggml_backend_dev_t>     simple_devs;
    ggml_backend_meta_get_ونڈو_state_t get_ونڈو_state;
    فراغ *                              get_ونڈو_state_ud;

    std::string name;
    std::string description;

    ggml_backend_meta_device_context(
            std::vector<ggml_backend_dev_t> simple_devs, ggml_backend_meta_get_ونڈو_state_t get_ونڈو_state, فراغ * get_ونڈو_state_ud) :
            simple_devs(std::move(simple_devs)), get_ونڈو_state(get_ونڈو_state), get_ونڈو_state_ud(get_ونڈو_state_ud) {
        name        = std::string("Meta(");
        description = std::string("Meta(");
        لكل (size_t i = 0; i < simple_devs.size(); i++) {
            إذا (i > 0) {
                name        += ",";
                description += ",";
            }
            name        += ggml_backend_dev_name       (simple_devs[i]);
            description += ggml_backend_dev_description(simple_devs[i]);
        }
        name        += ")";
        description += ")";
    }

    منطقي عامل<(ثابت ggml_backend_meta_device_context & other) ثابت {
        أرجع std::tie(simple_devs, get_ونڈو_state, get_ونڈو_state_ud)
            < std::tie(other.simple_devs, other.get_ونڈو_state, other.get_ونڈو_state_ud);
    }
};

ساكن منطقي ggml_backend_dev_is_meta(ggml_backend_dev_t dev);

ساكن ثابت حرف * ggml_backend_meta_device_get_name(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ثابت ggml_backend_meta_device_context * meta_dev_ctx = (ثابت ggml_backend_meta_device_context *) dev->context;
    أرجع meta_dev_ctx->name.c_str();
}

ساكن ثابت حرف * ggml_backend_meta_device_get_description(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ثابت ggml_backend_meta_device_context * meta_dev_ctx = (ثابت ggml_backend_meta_device_context *) dev->context;
    أرجع meta_dev_ctx->description.c_str();
}

ساكن فراغ ggml_backend_meta_device_get_حافظہ(ggml_backend_dev_t dev, size_t * حرر, size_t * کل) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ثابت ggml_backend_meta_device_context * meta_dev_ctx = (ثابت ggml_backend_meta_device_context *) dev->context;
    *حرر  = 0;
    *کل = 0;
    لكل (ggml_backend_dev_t dev : meta_dev_ctx->simple_devs) {
        size_t tmp_free, tmp_کل;
        ggml_backend_dev_حافظہ(dev, &tmp_free, &tmp_کل);
        *حرر  += tmp_free;
        *کل += tmp_کل;
    }
}

ساكن تعداد ggml_backend_dev_type ggml_backend_meta_device_get_type(ggml_backend_dev_t dev) {
    أرجع GGML_BACKEND_DEVICE_TYPE_META;

    GGML_بن_ورتیا(dev);
}

ساكن فراغ ggml_backend_meta_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
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
        /* .host_buffer           = */ خطأ, // Not implemented.
        /* .buffer_from_host_ptr  = */ خطأ, // Not implemented.
        /* .events                = */ خطأ, // Not implemented.
        /* .mmap_support          = */ سچ,
    };
    لكل (ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
        ggml_backend_dev_props tmp_props;
        ggml_backend_dev_get_props(simple_dev, &tmp_props);
        props->caps.async                = props->caps.async                && tmp_props.caps.async;
        props->caps.host_بفر          = props->caps.host_بفر          && tmp_props.caps.host_بفر;
        props->caps.بفر_from_host_ptr = props->caps.بفر_from_host_ptr && tmp_props.caps.بفر_from_host_ptr;
        props->caps.events               = props->caps.events               && tmp_props.caps.events;
        props->caps.mmap_support         = props->caps.mmap_support         && tmp_props.caps.mmap_support;
    }
}

ساكن ggml_backend_t ggml_backend_meta_device_init_backend(ggml_backend_dev_t dev, ثابت حرف * params);

ساكن ggml_backend_بفر_type_t ggml_backend_meta_device_get_بفر_type(ggml_backend_dev_t dev);

ساكن ggml_backend_بفر_type_t ggml_backend_meta_device_get_host_بفر_type(ggml_backend_dev_t dev);

ساكن منطقي ggml_backend_meta_device_supports_op(ggml_backend_dev_t dev, ثابت ggml_tensor * op) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ثابت ggml_backend_meta_device_context * meta_dev_ctx = (ثابت ggml_backend_meta_device_context *) dev->context;
    أرجع std::all_of(meta_dev_ctx->simple_devs.begin(), meta_dev_ctx->simple_devs.end(),
        [op](ggml_backend_dev_t simple_dev) { أرجع ggml_backend_dev_supports_op(simple_dev, op); });
}

ساكن منطقي ggml_backend_meta_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_بفر_type_t buft) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ggml_backend_dev_t dev_buft = ggml_backend_buft_get_device(buft);
    إذا (!ggml_backend_dev_is_meta(dev_buft)) {
        أرجع خطأ;
    }
    ثابت ggml_backend_meta_device_context * meta_dev_ctx      = (ثابت ggml_backend_meta_device_context *) dev->context;
    ثابت ggml_backend_meta_device_context * meta_buft_dev_ctx = (ثابت ggml_backend_meta_device_context *) dev_buft->context;
    إذا (meta_dev_ctx->simple_devs.size() != meta_buft_dev_ctx->simple_devs.size()) {
        أرجع خطأ;
    }
    لكل (size_t i = 0; i < meta_dev_ctx->simple_devs.size(); i++) {
        إذا (meta_dev_ctx->simple_devs[i] != meta_buft_dev_ctx->simple_devs[i]) {
            أرجع خطأ;
        }
    }
    أرجع سچ;
}

ساكن ثابت ggml_backend_device_i ggml_backend_meta_device_iface = {
    /* .get_name             = */ ggml_backend_meta_device_get_name,
    /* .get_description      = */ ggml_backend_meta_device_get_description,
    /* .get_memory           = */ ggml_backend_meta_device_get_حافظہ,
    /* .get_type             = */ ggml_backend_meta_device_get_type,
    /* .get_props            = */ ggml_backend_meta_device_get_props,
    /* .init_backend         = */ ggml_backend_meta_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_meta_device_get_بفر_type,
    /* .get_host_buffer_type = */ ggml_backend_meta_device_get_host_بفر_type,
    /* .buffer_from_host_ptr = */ خالی_سوچک,
    /* .supports_op          = */ ggml_backend_meta_device_supports_op,
    /* .supports_buft        = */ ggml_backend_meta_device_supports_buft,
    /* .offload_op           = */ خالی_سوچک,
    /* .event_new            = */ خالی_سوچک,
    /* .event_free           = */ خالی_سوچک,
    /* .event_synchronize    = */ خالی_سوچک,
};

ساكن منطقي ggml_backend_dev_is_meta(ggml_backend_dev_t dev) {
    أرجع dev != خالی_سوچک && dev->iface.get_name == ggml_backend_meta_device_iface.get_name;
}

ساكن size_t ggml_backend_meta_dev_n_devs(ggml_backend_dev_t meta_dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(meta_dev));
    ثابت ggml_backend_meta_device_context * meta_dev_ctx = (ثابت ggml_backend_meta_device_context *) meta_dev->context;
    أرجع meta_dev_ctx->simple_devs.size();
}

ساكن ggml_backend_dev_t ggml_backend_meta_dev_simple_dev(ggml_backend_dev_t meta_dev, size_t اشاریہ) {
    GGML_ASSERT(ggml_backend_dev_is_meta(meta_dev));
    ثابت ggml_backend_meta_device_context * meta_dev_ctx = (ثابت ggml_backend_meta_device_context *) meta_dev->context;
    GGML_ASSERT(اشاریہ < meta_dev_ctx->simple_devs.size());
    أرجع meta_dev_ctx->simple_devs[اشاریہ];
}

ggml_backend_dev_t ggml_backend_meta_device(
        ggml_backend_dev_t * devs, size_t n_devs, ggml_backend_meta_get_ونڈو_state_t get_ونڈو_state, فراغ * get_ونڈو_state_ud) {
    GGML_ASSERT(n_devs <= GGML_BACKEND_META_MAX_DEVICES);
    // TODO: this is not thread-safe - needs to be fixed
    ساكن std::vector<std::unique_ptr<ggml_backend_meta_device_context>>         ctxs;
    ساكن std::map<ggml_backend_meta_device_context, بنية ggml_backend_device> meta_devs;

    std::vector<ggml_backend_dev_t> simple_devs;
    simple_devs.reserve(n_devs);
    لكل (size_t i = 0; i < n_devs; i++) {
        simple_devs.push_back(devs[i]);
    }
    ggml_backend_meta_device_context ctx(simple_devs, get_ونڈو_state, get_ونڈو_state_ud);

    {
        خودکار it = meta_devs.find(ctx);
        إذا (it != meta_devs.end()) {
            أرجع &it->second;
        }
    }
    ctxs.push_back(std::make_unique<ggml_backend_meta_device_context>(ctx));

    بنية ggml_backend_device meta_dev = {
        /*iface  =*/ ggml_backend_meta_device_iface,
        /*reg    =*/ خالی_سوچک,
        /*ctx    =*/ ctxs.back().get(),
    };

    خودکار نتیجہ = meta_devs.emplace(*ctxs.back(), meta_dev);
    أرجع &نتیجہ.پہلا->second;
}

//
// meta backend buffer type
//

بنية ggml_backend_meta_بفر_type_context {
    std::vector<ggml_backend_بفر_type_t> simple_bufts;

    std::string name;

    ggml_backend_meta_بفر_type_context(std::vector<ggml_backend_بفر_type_t> simple_bufts) : simple_bufts(std::move(simple_bufts)) {
        name = "Meta(";
        لكل (size_t i = 0; i < simple_bufts.size(); i++) {
            إذا (i > 0) {
                name += ",";
            }
            name += ggml_backend_buft_name(simple_bufts[i]);
        }
        name += ")";
    }

    منطقي عامل<(ثابت ggml_backend_meta_بفر_type_context & other) ثابت {
        أرجع simple_bufts < other.simple_bufts;
    }
};

ساكن size_t ggml_backend_meta_buft_n_bufts(ggml_backend_بفر_type_t meta_buft) {
    GGML_ASSERT(ggml_backend_buft_is_meta(meta_buft));
    ثابت ggml_backend_meta_بفر_type_context * meta_buft_ctx = (ثابت ggml_backend_meta_بفر_type_context *) meta_buft->context;
    أرجع meta_buft_ctx->simple_bufts.size();
}

ساكن ثابت حرف * ggml_backend_meta_بفر_type_get_name(ggml_backend_بفر_type_t buft) {
    GGML_ASSERT(ggml_backend_buft_is_meta(buft));
    ثابت ggml_backend_meta_بفر_type_context * meta_buft_ctx = (ثابت ggml_backend_meta_بفر_type_context *) buft->context;
    أرجع meta_buft_ctx->name.c_str();
}

ساكن ggml_backend_بفر_type_t ggml_backend_meta_buft_simple_buft(ggml_backend_بفر_type_t meta_buft, size_t اشاریہ) {
    GGML_ASSERT(ggml_backend_buft_is_meta(meta_buft));
    ثابت ggml_backend_meta_بفر_type_context * meta_buft_ctx = (ثابت ggml_backend_meta_بفر_type_context *) meta_buft->context;
    GGML_ASSERT(اشاریہ < meta_buft_ctx->simple_bufts.size());
    أرجع meta_buft_ctx->simple_bufts[اشاریہ];
}

ساكن ggml_backend_بفر_t ggml_backend_meta_بفر_type_alloc_بفر(ggml_backend_بفر_type_t buft, size_t size);

ساكن ggml_backend_بفر_t ggml_backend_meta_بفر_type_alloc_بفر_n(ggml_backend_بفر_type_t buft, ggml_tensor ** tensors, صحيح n_tensors);

ساكن size_t ggml_backend_meta_بفر_type_get_alignment(ggml_backend_بفر_type_t buft) {
    ثابت size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_alignment = 1;
    لكل (size_t i = 0; i < n_simple_bufts; i++) {
        ثابت size_t alignment = ggml_backend_buft_get_alignment(ggml_backend_meta_buft_simple_buft(buft, i));
        max_alignment = std::max(max_alignment, alignment);
        GGML_ASSERT(max_alignment % alignment == 0);
    }
    أرجع max_alignment;
}

ساكن size_t ggml_backend_meta_بفر_type_get_max_ناپ(ggml_backend_بفر_type_t buft) {
    ثابت size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_size = SIZE_MAX;
    لكل (size_t i = 0; i < n_simple_bufts; i++) {
        max_size = std::min(max_size, ggml_backend_buft_get_max_ناپ(ggml_backend_meta_buft_simple_buft(buft, i)));
    }
    أرجع max_size;
}

ساكن size_t ggml_backend_meta_بفر_type_get_alloc_ناپ(ggml_backend_بفر_type_t buft, ثابت ggml_tensor * tensor) {
    ثابت size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_alloc_ناپ = 0;
    لكل (size_t i = 0; i < n_simple_bufts; i++) {
        ثابت size_t alloc_ناپ = ggml_backend_buft_get_alloc_ناپ(ggml_backend_meta_buft_simple_buft(buft, i), tensor);
        max_alloc_ناپ = std::max(max_alloc_ناپ, alloc_ناپ);
    }
    أرجع max_alloc_ناپ;
}

ساكن منطقي ggml_backend_meta_بفر_type_is_host(ggml_backend_بفر_type_t buft) {
    ثابت size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    لكل (size_t i = 0; i < n_simple_bufts; i++) {
        إذا (!ggml_backend_buft_is_host(ggml_backend_meta_buft_simple_buft(buft, i))) {
            أرجع خطأ;
        }
    }
    أرجع سچ;
}

ساكن ثابت بنية ggml_backend_بفر_type_i ggml_backend_meta_بفر_type_iface = {
    /* .get_name            = */ ggml_backend_meta_بفر_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_meta_بفر_type_alloc_بفر,
    /* .alloc_buffer_n      = */ ggml_backend_meta_بفر_type_alloc_بفر_n,
    /* .get_alignment       = */ ggml_backend_meta_بفر_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_meta_بفر_type_get_max_ناپ,
    /* .get_alloc_size      = */ ggml_backend_meta_بفر_type_get_alloc_ناپ,
    /* .get_alloc_size_n    = */ عدم,
    /* .is_host             = */ ggml_backend_meta_بفر_type_is_host,
};

منطقي ggml_backend_buft_is_meta(ggml_backend_بفر_type_t buft) {
    أرجع buft != خالی_سوچک && buft->iface.get_name == ggml_backend_meta_بفر_type_iface.get_name;
}

ساكن ggml_backend_بفر_type_t ggml_backend_meta_device_get_بفر_type(ggml_backend_dev_t dev) {
    ساكن std::map<ggml_backend_dev_t, بنية ggml_backend_بفر_type> meta_bufts;
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    {
        خودکار it = meta_bufts.find(dev);
        إذا (it != meta_bufts.end()) {
            أرجع &it->second;
        }
    }

    ثابت size_t n_devs = ggml_backend_meta_dev_n_devs(dev);
    std::vector<ggml_backend_بفر_type_t> simple_bufts;
    simple_bufts.reserve(n_devs);
    لكل (size_t i = 0; i < n_devs; i++) {
        simple_bufts.push_back(ggml_backend_dev_بفر_type(ggml_backend_meta_dev_simple_dev(dev, i)));
    }
    ggml_backend_meta_بفر_type_context * buft_ctx = جديد ggml_backend_meta_بفر_type_context(simple_bufts);

    بنية ggml_backend_بفر_type meta_buft = {
        /*iface  =*/ ggml_backend_meta_بفر_type_iface,
        /*device =*/ dev,
        /*ctx    =*/ buft_ctx,
    };
    خودکار نتیجہ = meta_bufts.emplace(dev, meta_buft);
    أرجع &نتیجہ.پہلا->second;
}

ساكن ggml_backend_بفر_type_t ggml_backend_meta_device_get_host_بفر_type(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ثابت ggml_backend_meta_device_context * meta_dev_ctx = (ثابت ggml_backend_meta_device_context *) dev->context;

    ggml_backend_بفر_type_t host_buft = خالی_سوچک;
    لكل (ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
        ggml_backend_بفر_type_t simple_host_buft = ggml_backend_dev_host_بفر_type(simple_dev);
        إذا (simple_host_buft == خالی_سوچک) {
            أرجع خالی_سوچک;
        }
        إذا (host_buft == خالی_سوچک) {
            host_buft = simple_host_buft;
        } وإلا إذا (host_buft != simple_host_buft) {
            // if different simple devices have different host buffer types,
            // we cannot provide a single host buffer type for the meta device
            أرجع خالی_سوچک;
        }
    }
    أرجع host_buft;
}

//
// meta backend buffer
//

// Container to hold the tensor slices per simple ggml backend buffer.
بنية ggml_backend_meta_simple_tensor_container {
    std::vector<ggml_context_ptr> ctxs;
    std::map<ثابت ggml_tensor *, std::vector<ggml_tensor *>> simple_tensors;

    ggml_backend_meta_simple_tensor_container(ثابت ggml_init_params & params, ثابت صحيح n_simple) {
        ctxs.reserve(n_simple);
        لكل (صحيح i = 0; i < n_simple; i++) {
            ctxs.emplace_back(ggml_init(params));
        }
    }
    ggml_backend_meta_simple_tensor_container() {}
};

بنية ggml_backend_meta_بفر_context {
    // FIXME
    // Most tensors can simply be stored statically in their own buffer.
    // Externally created views however also need a mapping to simple tensors but they use the buffer of the view source.
    // If external views are simply using that buffer they will slowly deplete its memory.
    // Current solution: rotating set of 2 "compute" containers to hold external views, works correctly for llama.cpp.
    // Long-term: tie the lifetime of external views to the meta backend executing the graph instead,
    //     currently not possible due to graph-external operations in the backend scheduler.
    ggml_backend_meta_simple_tensor_container stc_static;
    ggml_backend_meta_simple_tensor_container stc_compute[2];
    صحيح stc_compute_اشاریہ      = 0;
    صحيح stc_compute_اشاریہ_اگلا = 0;
    std::vector<ggml_backend_بفر_ptr> bufs;

    // FIXME
    // The size of the split state cache is unbounded and can theoretically grow infinitely large.
    // However, it is also expensive to build and clearing it on every rebuild in ggml_backend_meta_graph_compute is too expensive.
    ساكن constexpr size_t nbtc = GGML_TENSOR_ناپ - حجم(ggml_tensor::padding);
    std::map<std::pair<ثابت ggml_tensor *, منطقي>, std::pair<ggml_backend_meta_ونڈو_state, حرف[nbtc]>> ونڈو_state_cache;

    صحيح debug;

    ggml_backend_meta_بفر_context(
            ggml_backend_meta_simple_tensor_container & stc_static,
            ggml_backend_meta_simple_tensor_container & stc_compute_0,
            ggml_backend_meta_simple_tensor_container & stc_compute_1,
            ثابت std::vector<ggml_backend_بفر_t> & bufs)
            : stc_static(std::move(stc_static)), stc_compute{std::move(stc_compute_0), std::move(stc_compute_1)} {
        هذا->bufs.reserve(bufs.size());
        لكل (ggml_backend_بفر_t buf : bufs) {
            هذا->bufs.emplace_back(buf);
        }
        ثابت حرف * GGML_META_DEBUG = getenv("GGML_META_DEBUG");
        debug = GGML_META_DEBUG ? atoi(GGML_META_DEBUG) : 0;
    }

    ggml_backend_meta_simple_tensor_container & get_simple_tensor_container(ثابت ggml_tensor * tensor) {
        إذا (stc_static.simple_tensors.find(tensor) != stc_static.simple_tensors.end()) {
            أرجع stc_static;
        }
        أرجع stc_compute[stc_compute_اشاریہ];
    }
};

ساكن فراغ ggml_backend_meta_بفر_free_بفر(ggml_backend_بفر_t بفر) {
    GGML_ASSERT(ggml_backend_بفر_is_meta(بفر));
    ggml_backend_meta_بفر_context * buf_ctx = (ggml_backend_meta_بفر_context *) بفر->context;
    مٹاؤ buf_ctx;
}

ساكن size_t ggml_backend_meta_بفر_n_bufs(ggml_backend_بفر_t meta_buf) {
    GGML_ASSERT(ggml_backend_بفر_is_meta(meta_buf));
    ggml_backend_meta_بفر_context * buf_ctx = (ggml_backend_meta_بفر_context *) meta_buf->context;
    أرجع buf_ctx->bufs.size();
}

ساكن ggml_backend_بفر_t ggml_backend_meta_بفر_simple_بفر(ggml_backend_بفر_t meta_buf, size_t اشاریہ) {
    GGML_ASSERT(ggml_backend_بفر_is_meta(meta_buf));
    ggml_backend_meta_بفر_context * buf_ctx = (ggml_backend_meta_بفر_context *) meta_buf->context;
    GGML_ASSERT(اشاریہ < buf_ctx->bufs.size());
    أرجع buf_ctx->bufs[اشاریہ].get();
}

ساكن بنية ggml_tensor * ggml_backend_meta_بفر_simple_tensor(ثابت بنية ggml_tensor * tensor, size_t اشاریہ) {
    GGML_ASSERT(ggml_backend_بفر_is_meta(tensor->بفر));
    ggml_backend_meta_بفر_context * buf_ctx = (ggml_backend_meta_بفر_context *) tensor->بفر->context;
    GGML_ASSERT(اشاریہ < buf_ctx->bufs.size());

    ggml_backend_meta_simple_tensor_container & stc = buf_ctx->get_simple_tensor_container(tensor);
    خودکار it = stc.simple_tensors.find(tensor);
    إذا (it == stc.simple_tensors.end()) {
        أرجع خالی_سوچک;
    }
    أرجع it->second[اشاریہ];
}

ساكن بنية ggml_backend_meta_ونڈو_state ggml_backend_meta_get_ونڈو_state(ثابت بنية ggml_tensor * tensor, منطقي assume_sync);

ساكن بنية ggml_backend_meta_ونڈو_state ggml_backend_meta_get_ونڈو_state(
        ggml_backend_meta_simple_tensor_container & stc, ثابت بنية ggml_tensor * tensor, منطقي assume_sync) {
    // FIXME Currently this function preserves/erases the information in n_segments and nr in an inconsistent way.
    // Since the operations in question are developed specifically for llama.cpp this currently does not manifest as a bug there.
    // However, in a broader ggml context with arbitrary ggml graphs this can lead to unexpected results.
    ثابت size_t n_bufs = ggml_backend_meta_بفر_n_bufs(tensor->بفر);
    ggml_backend_meta_بفر_context * buf_ctx = (ggml_backend_meta_بفر_context *) tensor->بفر->context;

    خودکار ونڈو_حالتاں_equal = [&](ثابت ggml_backend_meta_ونڈو_state & a, ثابت ggml_backend_meta_ونڈو_state & b) -> منطقي {
        إذا (a.axis != b.axis) {
            أرجع خطأ;
        }
        لكل (size_t j = 0; j < n_bufs; j++) {
            int64_t جوڑ_a = 0;
            لكل (size_t s = 0; s < a.n_segments; s++) {
                جوڑ_a += a.ne[s*n_bufs + j] * a.nr[s];
            }
            int64_t جوڑ_b = 0;
            لكل (size_t s = 0; s < b.n_segments; s++) {
                جوڑ_b += b.ne[s*n_bufs + j] * b.nr[s];
            }
            إذا (جوڑ_a != جوڑ_b) {
                أرجع خطأ;
            }
        }
        أرجع سچ;
    };

    خودکار handle_generic = [&](ثابت std::vector<ggml_backend_meta_ونڈو_state> & src_ss, منطقي scalar_only) -> ggml_backend_meta_ونڈو_state {
        ggml_backend_meta_ونڈو_state ret = {GGML_BACKEND_ونڈو_AXIS_NONE, {0}, {1}, 1};
        لكل (size_t i = 0; i < GGML_MAX_SRC; i++) {
            إذا (tensor->src[i] == خالی_سوچک || tensor->src[i] == tensor) {
                استمر;
            }
            إذا (ret.axis == GGML_BACKEND_ونڈو_AXIS_NONE) {
                ret = src_ss[i];
            } وإلا إذا (!ونڈو_حالتاں_equal(src_ss[i], ret)) {
                ret = {GGML_BACKEND_ونڈو_AXIS_UNKNOWN, {0}, {1}, 1};
                اكسر;
            }
        }
        إذا (ret.axis == GGML_BACKEND_ونڈو_AXIS_NONE) {
            ret = {GGML_BACKEND_ونڈو_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        إذا (scalar_only && ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
            ret = {GGML_BACKEND_ونڈو_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        GGML_ASSERT(ret.axis != GGML_BACKEND_ونڈو_AXIS_UNKNOWN);
        أرجع ret;
    };

    // Some ops process data on a per-row bases:
    خودکار handle_فی_صف = [&](ثابت std::vector<ggml_backend_meta_ونڈو_state> & src_ss) -> ggml_backend_meta_ونڈو_state {
        GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_ونڈو_AXIS_0);
        أرجع src_ss[0];
    };

    // Some ops broadcast the src1 data across src0:
    خودکار handle_bin_bcast = [&](ثابت std::vector<ggml_backend_meta_ونڈو_state> & src_ss) -> ggml_backend_meta_ونڈو_state {
        إذا (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS &&
                tensor->src[1]->ne[src_ss[0].axis] == 1 && src_ss[1].axis == GGML_BACKEND_ونڈو_AXIS_MIRRORED) {
            أرجع src_ss[0];
        }
        إذا (src_ss[2].axis == GGML_BACKEND_ونڈو_AXIS_MIRRORED && (src_ss[0].axis == src_ss[1].axis ||
           (src_ss[0].axis == GGML_BACKEND_ونڈو_AXIS_MIRRORED && (src_ss[1].axis == GGML_BACKEND_ونڈو_AXIS_PARTIAL)))) {
            أرجع src_ss[0]; // GGML_OP_ADD_ID
        }
        GGML_ASSERT(tensor->src[2] == خالی_سوچک || src_ss[2].axis == GGML_BACKEND_ونڈو_AXIS_MIRRORED);
        أرجع handle_generic(src_ss, /*scalar_only =*/ خطأ);
    };

    خودکار handle_concat = [&](ثابت std::vector<ggml_backend_meta_ونڈو_state> & src_ss) -> ggml_backend_meta_ونڈو_state {
        ثابت ggml_backend_meta_ونڈو_axis concat_axis = ggml_backend_meta_ونڈو_axis(ggml_get_op_params_i32(tensor, 0));
        إذا (src_ss[0].axis == GGML_BACKEND_ونڈو_AXIS_MIRRORED && src_ss[1].axis >= 0 && src_ss[1].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(concat_axis != src_ss[1].axis);
            أرجع src_ss[1];
        }
        إذا (src_ss[1].axis == GGML_BACKEND_ونڈو_AXIS_MIRRORED && src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(concat_axis != src_ss[0].axis);
            أرجع src_ss[0];
        }
        إذا (src_ss[0].axis == src_ss[1].axis && src_ss[0].axis != concat_axis) {
            أرجع src_ss[0];
        }
        أرجع handle_generic(src_ss, /*scalar_only =*/ سچ);
    };

    خودکار handle_mul_mat = [&](ثابت std::vector<ggml_backend_meta_ونڈو_state> & src_ss) -> ggml_backend_meta_ونڈو_state {
        إذا (src_ss[0].axis == GGML_BACKEND_ونڈو_AXIS_MIRRORED && src_ss[1].axis == GGML_BACKEND_ونڈو_AXIS_MIRRORED) {
            أرجع {GGML_BACKEND_ونڈو_AXIS_MIRRORED, {0}, {1}, 1};
        }
        إذا (src_ss[0].axis == GGML_BACKEND_ونڈو_AXIS_1 && src_ss[1].axis == GGML_BACKEND_ونڈو_AXIS_MIRRORED) {
            ggml_backend_meta_ونڈو_state ret = src_ss[0];
            ret.axis = GGML_BACKEND_ونڈو_AXIS_0;
            ret.nr[0] = 1;
            ret.n_segments = 1;
            أرجع ret;
        }
        إذا (src_ss[1].axis == GGML_BACKEND_ونڈو_AXIS_1 && src_ss[0].axis == GGML_BACKEND_ونڈو_AXIS_MIRRORED) {
            أرجع src_ss[1];
        }
        إذا (src_ss[0].axis == GGML_BACKEND_ونڈو_AXIS_0 && src_ss[1].axis == GGML_BACKEND_ونڈو_AXIS_0) {
            GGML_ASSERT(ونڈو_حالتاں_equal(src_ss[0], src_ss[1]));
            أرجع {assume_sync ? GGML_BACKEND_ونڈو_AXIS_MIRRORED : GGML_BACKEND_ونڈو_AXIS_PARTIAL, {0}, {1}, 1};
        }
        إذا (src_ss[0].axis == src_ss[1].axis && src_ss[0].axis >= GGML_BACKEND_ونڈو_AXIS_2 &&
                src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(ونڈو_حالتاں_equal(src_ss[0], src_ss[1]));
            أرجع src_ss[0];
        }
        // batched matmul with the batches split across devices and a replicated activation
        إذا (src_ss[0].axis >= GGML_BACKEND_ونڈو_AXIS_2 && src_ss[0].axis < GGML_MAX_DIMS &&
                src_ss[1].axis == GGML_BACKEND_ونڈو_AXIS_MIRRORED) {
            أرجع src_ss[0];
        }
        GGML_ABORT("unsupported mul_mat split states: node=%s src0=%s axis=%d src1=%s axis=%d",
            tensor->name, tensor->src[0]->name, (صحيح) src_ss[0].axis, tensor->src[1]->name, (صحيح) src_ss[1].axis);
        //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
    };

    خودکار handle_reshape = [&](ثابت std::vector<ggml_backend_meta_ونڈو_state> & src_ss) -> ggml_backend_meta_ونڈو_state {
        بدل (src_ss[0].axis) {
            حالة GGML_BACKEND_ونڈو_AXIS_0:
            حالة GGML_BACKEND_ونڈو_AXIS_1:
            حالة GGML_BACKEND_ونڈو_AXIS_2:
            حالة GGML_BACKEND_ونڈو_AXIS_3: {
                int64_t ادھار_ne_وچ = 1;
                لكل (صحيح dim = 0; dim <= src_ss[0].axis; dim++) {
                    ادھار_ne_وچ *= tensor->src[0]->ne[dim];
                }
                إذا (src_ss[0].n_segments == 1) {
                    ادھار_ne_وچ /= src_ss[0].nr[0];
                    إذا (src_ss[0].axis == ggml_n_dims(tensor->src[0]) - 1 && src_ss[0].nr[0] == 1) {
                        أرجع {ggml_backend_meta_ونڈو_axis(ggml_n_dims(tensor) - 1), {0}, {1}, 1};
                    }
                    إذا (src_ss[0].axis == GGML_BACKEND_ونڈو_AXIS_0 && tensor->ne[0] == tensor->src[0]->ne[0] &&
                            tensor->ne[1] == 1 && src_ss[0].nr[0] == 1) {
                        منطقي complete_صفاں = سچ;
                        لكل (size_t j = 0; j < n_bufs; j++) {
                            ثابت int64_t ne = src_ss[0].ne[j];
                            complete_صفاں = complete_صفاں && (ne == 0 || ne == tensor->src[0]->ne[0]);
                        }
                        إذا (complete_صفاں) {
                            // Move a complete dim-0 split to the following singleton dimension.
                            أرجع {GGML_BACKEND_ونڈو_AXIS_1, {0}, {1}, 1};
                        }
                    }
                }
                // Reshape outputs use one segment; split-state propagation merges source segments.
                int64_t ادھار_ne_باہرلا = 1;
                لكل (صحيح dim = 0; dim < GGML_MAX_DIMS; dim++) {
                    ادھار_ne_باہرلا *= tensor->ne[dim];
                    إذا (ادھار_ne_باہرلا % ادھار_ne_وچ == 0) {
                        أرجع {ggml_backend_meta_ونڈو_axis(dim), {0}, {uint32_t(ادھار_ne_باہرلا/ادھار_ne_وچ)}, 1};
                    }
                    إذا (ادھار_ne_باہرلا > ادھار_ne_وچ) {
                        GGML_ASSERT(src_ss[0].n_segments == 1);
                        GGML_ASSERT(src_ss[0].nr[0]      == 1);
                        أرجع {ggml_backend_meta_ونڈو_axis(dim), {0}, {1}, 1};
                    }
                }
                GGML_ABORT("shape mismatch for %s", ggml_op_name(tensor->op));
            }
            حالة GGML_BACKEND_ونڈو_AXIS_MIRRORED:
            حالة GGML_BACKEND_ونڈو_AXIS_PARTIAL: {
                أرجع src_ss[0];
            }
            افتراضي: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    خودکار handle_cpy = [&](ثابت std::vector<ggml_backend_meta_ونڈو_state> & src_ss) -> ggml_backend_meta_ونڈو_state {
        إذا (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            أرجع handle_reshape(src_ss);
        }
        أرجع handle_generic(src_ss, /*scalar_only =*/ خطأ);
    };

    خودکار handle_view = [&](ثابت std::vector<ggml_backend_meta_ونڈو_state> & src_ss) -> ggml_backend_meta_ونڈو_state {
        إذا (ggml_is_contiguous(tensor) && ggml_is_contiguous(tensor->src[0])) {
            أرجع handle_reshape(src_ss);
        }
        ثابت صحيح axis = src_ss[0].axis;
        {
            منطقي all_strides_the_same = سچ;
            لكل (صحيح dim = 0; dim < GGML_MAX_DIMS; dim++) {
                إذا (tensor->ne[dim] == 1 && tensor->src[0]->ne[dim] == 1) {
                    استمر;
                }
                إذا (tensor->nb[dim] != tensor->src[0]->nb[dim]) {
                    all_strides_the_same = خطأ;
                    اكسر;
                }
            }
            إذا (all_strides_the_same) {
                أرجع src_ss[0];
            }
        }
        إذا (!ggml_is_permuted(tensor) && !ggml_is_permuted(tensor->src[0]) && axis >= 0 && axis < GGML_MAX_DIMS-1) {
            لكل (صحيح dim = 0; dim < GGML_MAX_DIMS-1; dim++) {
                إذا (tensor->nb[dim+1] == tensor->src[0]->nb[axis+1]) {
                    أرجع {ggml_backend_meta_ونڈو_axis(dim), {0}, {1}, 1};
                }
            }
            GGML_ABORT("fatal error");
        }
        إذا (src_ss[0].axis == GGML_BACKEND_ونڈو_AXIS_MIRRORED || src_ss[0].axis == GGML_BACKEND_ونڈو_AXIS_PARTIAL) {
            أرجع src_ss[0];
        }
        GGML_ABORT("view of permuted tensor not implemented");
        //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
    };

    خودکار handle_ترتیب_بدلو = [&](ثابت std::vector<ggml_backend_meta_ونڈو_state> & src_ss) -> ggml_backend_meta_ونڈو_state {
        بدل (src_ss[0].axis) {
            حالة GGML_BACKEND_ونڈو_AXIS_0:
            حالة GGML_BACKEND_ونڈو_AXIS_1:
            حالة GGML_BACKEND_ونڈو_AXIS_2:
            حالة GGML_BACKEND_ونڈو_AXIS_3: {
                GGML_ASSERT(src_ss[0].n_segments == 1 || src_ss[0].nr[0] == 1);
                أرجع {ggml_backend_meta_ونڈو_axis(tensor->op_params[src_ss[0].axis]), {0}, {src_ss[0].nr[0]}, 1};
            }
            حالة GGML_BACKEND_ونڈو_AXIS_MIRRORED:
            حالة GGML_BACKEND_ونڈو_AXIS_PARTIAL: {
                أرجع src_ss[0];
            }
            افتراضي: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    خودکار handle_transpose = [&](ثابت std::vector<ggml_backend_meta_ونڈو_state> & src_ss) -> ggml_backend_meta_ونڈو_state {
        بدل (src_ss[0].axis) {
            حالة GGML_BACKEND_ونڈو_AXIS_0:
            حالة GGML_BACKEND_ونڈو_AXIS_1: {
                GGML_ASSERT(src_ss[0].n_segments == 1 || src_ss[0].nr[0] == 1);
                أرجع {ggml_backend_meta_ونڈو_axis(صحيح(src_ss[0].axis) ^ 1), {0}, {src_ss[0].nr[0]}, 1};
            }
            حالة GGML_BACKEND_ونڈو_AXIS_2:
            حالة GGML_BACKEND_ونڈو_AXIS_3:
            حالة GGML_BACKEND_ونڈو_AXIS_MIRRORED:
            حالة GGML_BACKEND_ونڈو_AXIS_PARTIAL: {
                أرجع src_ss[0];
            }
            افتراضي: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    خودکار handle_get_صفاں = [&](ثابت std::vector<ggml_backend_meta_ونڈو_state> & src_ss) -> ggml_backend_meta_ونڈو_state {
        إذا (src_ss[0].axis == GGML_BACKEND_ونڈو_AXIS_0 && src_ss[1].axis == GGML_BACKEND_ونڈو_AXIS_MIRRORED) {
            أرجع src_ss[0];
        }
        أرجع handle_generic(src_ss, /*scalar_only =*/ سچ);
    };

    خودکار handle_سیٹ_صفاں = [&](ثابت std::vector<ggml_backend_meta_ونڈو_state> & src_ss) -> ggml_backend_meta_ونڈو_state {
        GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_ونڈو_AXIS_1);
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_ونڈو_AXIS_MIRRORED);
        GGML_ASSERT(ونڈو_حالتاں_equal(src_ss[0], src_ss[2]));
        أرجع src_ss[0];
    };

    خودکار handle_rope = [&](ثابت std::vector<ggml_backend_meta_ونڈو_state> & src_ss) -> ggml_backend_meta_ونڈو_state {
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_ونڈو_AXIS_MIRRORED);
        أرجع src_ss[0];
    };

    خودکار handle_pad = [&](ثابت std::vector<ggml_backend_meta_ونڈو_state> & src_ss) -> ggml_backend_meta_ونڈو_state {
        إذا (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(tensor->op_params[2*src_ss[0].axis + 0] == 0);
            GGML_ASSERT(tensor->op_params[2*src_ss[0].axis + 1] == 0);
        }
        أرجع src_ss[0];
    };

    خودکار handle_pool_1d = [&](ثابت std::vector<ggml_backend_meta_ونڈو_state> & src_ss) -> ggml_backend_meta_ونڈو_state {
        إذا (src_ss[0].axis == GGML_BACKEND_ونڈو_AXIS_0) {
            أرجع {GGML_BACKEND_ونڈو_AXIS_MIRRORED, {0}, {1}, 1};
        }
        أرجع src_ss[0];
    };

    خودکار handle_flash_attn_ext = [&](ثابت std::vector<ggml_backend_meta_ونڈو_state> & src_ss) -> ggml_backend_meta_ونڈو_state {
        GGML_ASSERT(tensor->src[3] == خالی_سوچک || src_ss[3].axis == GGML_BACKEND_ونڈو_AXIS_MIRRORED);

        إذا (src_ss[0].axis == GGML_BACKEND_ونڈو_AXIS_MIRRORED) {
            GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_ونڈو_AXIS_MIRRORED);
            GGML_ASSERT(src_ss[2].axis == GGML_BACKEND_ونڈو_AXIS_MIRRORED);
            GGML_ASSERT(tensor->src[4] == خالی_سوچک || src_ss[4].axis == GGML_BACKEND_ونڈو_AXIS_MIRRORED);
            أرجع {GGML_BACKEND_ونڈو_AXIS_MIRRORED, {0}, {1}, 1};
        }

        GGML_ASSERT(src_ss[0].axis == GGML_BACKEND_ونڈو_AXIS_2);
        ثابت منطقي kv_ونڈو = src_ss[1].axis == GGML_BACKEND_ونڈو_AXIS_2 &&
                src_ss[2].axis == GGML_BACKEND_ونڈو_AXIS_2;
        ثابت منطقي kv_mirrored = src_ss[1].axis == GGML_BACKEND_ونڈو_AXIS_MIRRORED &&
                src_ss[2].axis == GGML_BACKEND_ونڈو_AXIS_MIRRORED;
        GGML_ASSERT(kv_ونڈو || kv_mirrored);
        GGML_ASSERT(tensor->src[4] == خالی_سوچک || src_ss[4].axis == GGML_BACKEND_ونڈو_AXIS_0);
        أرجع {GGML_BACKEND_ونڈو_AXIS_1, {0}, {1}, 1};
    };

    خودکار handle_lightning_indexer = [&](
            ثابت std::vector<ggml_backend_meta_ونڈو_state> & src_ss) -> ggml_backend_meta_ونڈو_state {
        لكل (size_t i = 0; i < 4; i++) {
            GGML_ASSERT(src_ss[i].axis == GGML_BACKEND_ونڈو_AXIS_MIRRORED);
        }
        أرجع {GGML_BACKEND_ونڈو_AXIS_MIRRORED, {0}, {1}, 1};
    };

    خودکار handle_ssm_conv = [&](ثابت std::vector<ggml_backend_meta_ونڈو_state> & src_ss) -> ggml_backend_meta_ونڈو_state {
        إذا (src_ss[0].axis == src_ss[1].axis) {
            إذا (src_ss[0].axis == GGML_BACKEND_ونڈو_AXIS_0) {
                أرجع {GGML_BACKEND_ونڈو_AXIS_1, {0}, {1}, 1};
            }
            إذا (src_ss[0].axis == GGML_BACKEND_ونڈو_AXIS_1) {
                أرجع {GGML_BACKEND_ونڈو_AXIS_0, {0}, {1}, 1};
            }
        }
        أرجع handle_generic(src_ss, /*scalar_only =*/ خطأ);
    };

    خودکار handle_gated_delta_net = [&](ثابت std::vector<ggml_backend_meta_ونڈو_state> & src_ss) -> ggml_backend_meta_ونڈو_state {
        إذا (src_ss[0].axis == GGML_BACKEND_ونڈو_AXIS_MIRRORED && src_ss[1].axis == GGML_BACKEND_ونڈو_AXIS_MIRRORED &&
                src_ss[2].axis == GGML_BACKEND_ونڈو_AXIS_MIRRORED && src_ss[3].axis == GGML_BACKEND_ونڈو_AXIS_MIRRORED &&
                src_ss[4].axis == GGML_BACKEND_ونڈو_AXIS_MIRRORED && src_ss[5].axis == GGML_BACKEND_ونڈو_AXIS_MIRRORED) {
            أرجع src_ss[0];
        }
        GGML_ASSERT(src_ss[0].axis == GGML_BACKEND_ونڈو_AXIS_1);
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_ونڈو_AXIS_1);
        GGML_ASSERT(src_ss[2].axis == GGML_BACKEND_ونڈو_AXIS_1);
        GGML_ASSERT(src_ss[3].axis == GGML_BACKEND_ونڈو_AXIS_1);
        GGML_ASSERT(src_ss[4].axis == GGML_BACKEND_ونڈو_AXIS_1);
        // state shape is [S_v, S_v, H_v, n_seqs] (s0 only); the heads dim is its own axis 2,
        // so a head-aligned split on the input cache lands on axis 2 here.
        GGML_ASSERT(src_ss[5].axis == GGML_BACKEND_ونڈو_AXIS_2 || src_ss[5].axis == GGML_BACKEND_ونڈو_AXIS_1 || src_ss[5].axis == GGML_BACKEND_ونڈو_AXIS_0);
        أرجع {GGML_BACKEND_ونڈو_AXIS_0, {0}, {1}, 1};
    };

    خودکار calculate_ونڈو_state = [&]() -> ggml_backend_meta_ونڈو_state {
        إذا (ggml_nelements(tensor) == 0) {
            أرجع {GGML_BACKEND_ونڈو_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        إذا (ggml_backend_بفر_get_usage(tensor->بفر) != GGML_BACKEND_بفر_USAGE_COMPUTE && tensor->view_src == خالی_سوچک) {
            ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_بفر_get_type(tensor->بفر));
            ثابت ggml_backend_meta_device_context * dev_ctx = (ثابت ggml_backend_meta_device_context *) dev->context;
            ggml_backend_meta_ونڈو_state ret = dev_ctx->get_ونڈو_state(tensor, dev_ctx->get_ونڈو_state_ud);
            إذا (ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
                ثابت int64_t granularity = ret.axis == GGML_BACKEND_ونڈو_AXIS_0 ? ggml_blck_ناپ(tensor->type) : 1;
                int64_t ne_جوڑ = 0;
                لكل (size_t s = 0; s < ret.n_segments; s++) {
                    لكل (size_t j = 0; j < n_bufs; j++) {
                        GGML_ASSERT(ret.ne[s*n_bufs + j] % granularity == 0);
                        ne_جوڑ += ret.ne[s*n_bufs + j] * ret.nr[s];
                    }
                }
                GGML_ASSERT(ne_جوڑ == tensor->ne[ret.axis]);
            } وإلا إذا (ret.axis == GGML_BACKEND_ونڈو_AXIS_PARTIAL) {
                GGML_ASSERT(ret.n_segments == 1);
                GGML_ASSERT(ret.nr[0] == 1);
            }
            أرجع ret;
        }

        std::vector<ggml_backend_meta_ونڈو_state> src_ss(GGML_MAX_SRC, {GGML_BACKEND_ونڈو_AXIS_NONE, {0}, {1}, 1});
        لكل (size_t i = 0; i < GGML_MAX_SRC; i++) {
            إذا (tensor->src[i] == خالی_سوچک || tensor->src[i] == tensor) {
                src_ss[i] = {GGML_BACKEND_ونڈو_AXIS_UNKNOWN, {0}, {1}, 1};
                استمر;
            }
            src_ss[i] = ggml_backend_meta_get_ونڈو_state(stc, tensor->src[i], /*assume_sync =*/ سچ);
            GGML_ASSERT(src_ss[i].axis != GGML_BACKEND_ونڈو_AXIS_UNKNOWN);
        }

        ggml_backend_meta_ونڈو_state ونڈو_state;
        بدل (tensor->op) {
            حالة GGML_OP_NONE: {
                إذا (tensor->view_src != خالی_سوچک) {
                    // full-tensor view created with ggml_view_tensor, transparent for the split state
                    ونڈو_state = ggml_backend_meta_get_ونڈو_state(stc, tensor->view_src, assume_sync);
                } وإلا {
                    ونڈو_state = {GGML_BACKEND_ونڈو_AXIS_MIRRORED, {0}, {1}, 1};
                }
            } اكسر;
            حالة GGML_OP_DUP: {
                ونڈو_state = handle_generic(src_ss, /*scalar_only =*/ سچ);
            } اكسر;
            حالة GGML_OP_جوڑو:
            حالة GGML_OP_جوڑو_ID: {
                ونڈو_state = handle_bin_bcast(src_ss);
            } اكسر;
            حالة GGML_OP_ADD1:
            حالة GGML_OP_ACC: {
                ونڈو_state = handle_generic(src_ss, /*scalar_only =*/ سچ);
            } اكسر;
            حالة GGML_OP_SUB:
            حالة GGML_OP_MUL:
            حالة GGML_OP_DIV: {
                ونڈو_state = handle_bin_bcast(src_ss);
            } اكسر;
            حالة GGML_OP_SQR:
            حالة GGML_OP_SQRT:
            حالة GGML_OP_LOG:
            حالة GGML_OP_SIN:
            حالة GGML_OP_COS: {
                ونڈو_state = handle_generic(src_ss, /*scalar_only =*/ خطأ);
            } اكسر;
            حالة GGML_OP_جوڑ: {
                ونڈو_state = handle_generic(src_ss, /*scalar_only =*/ سچ);
            } اكسر;
            حالة GGML_OP_جوڑ_صفاں:
            حالة GGML_OP_CUMSUM:
            حالة GGML_OP_اوسط:
            حالة GGML_OP_ARGMAX:
            حالة GGML_OP_گنتی_EQUAL: {
                ونڈو_state = handle_فی_صف(src_ss);
            } اكسر;
            حالة GGML_OP_REPEAT:
            حالة GGML_OP_REPEAT_پٹھ: {
                ونڈو_state = handle_generic(src_ss, /*scalar_only =*/ خطأ);
            } اكسر;
            حالة GGML_OP_CONCAT: {
                ونڈو_state = handle_concat(src_ss);
            } اكسر;
            حالة GGML_OP_SILU_پٹھ: {
                ونڈو_state = handle_generic(src_ss, /*scalar_only =*/ خطأ);
            } اكسر;
            حالة GGML_OP_NORM:
            حالة GGML_OP_RMS_NORM:
            حالة GGML_OP_RMS_NORM_پٹھ:
            حالة GGML_OP_گروہ_NORM:
            حالة GGML_OP_L2_NORM: {
                ونڈو_state = handle_فی_صف(src_ss);
            } اكسر;
            حالة GGML_OP_MUL_MAT:
            حالة GGML_OP_MUL_MAT_ID: {
                ونڈو_state = handle_mul_mat(src_ss);
            } اكسر;
            حالة GGML_OP_باہرلا_PROD: {
                ونڈو_state = handle_generic(src_ss, /*scalar_only =*/ سچ);
            } اكسر;
            حالة GGML_OP_SCALE: {
                ونڈو_state = handle_generic(src_ss, /*scalar_only =*/ خطأ);
            } اكسر;
            حالة GGML_OP_سیٹ: {
                ونڈو_state = handle_generic(src_ss, /*scalar_only =*/ سچ);
            } اكسر;
            حالة GGML_OP_CPY: {
                ونڈو_state = handle_cpy(src_ss);
            } اكسر;
            حالة GGML_OP_CONT:
            حالة GGML_OP_RESHAPE: {
                ونڈو_state = handle_reshape(src_ss);
            } اكسر;
            حالة GGML_OP_VIEW: {
                ونڈو_state = handle_view(src_ss);
            } اكسر;
            حالة GGML_OP_ترتیب_بدلو: {
                ونڈو_state = handle_ترتیب_بدلو(src_ss);
            } اكسر;
            حالة GGML_OP_TRANSPOSE: {
                ونڈو_state = handle_transpose(src_ss);
            } اكسر;
            حالة GGML_OP_GET_صفاں: {
                ونڈو_state = handle_get_صفاں(src_ss);
            } اكسر;
            حالة GGML_OP_GET_صفاں_پٹھ: {
                ونڈو_state = handle_generic(src_ss, /*scalar_only =*/ سچ);
            } اكسر;
            حالة GGML_OP_سیٹ_صفاں: {
                ونڈو_state = handle_سیٹ_صفاں(src_ss);
            } اكسر;
            حالة GGML_OP_DIAG:
            حالة GGML_OP_DIAG_نقاب_INF:
            حالة GGML_OP_DIAG_نقاب_ZERO: {
                ونڈو_state = handle_generic(src_ss, /*scalar_only =*/ سچ);
            } اكسر;
            حالة GGML_OP_SOFT_MAX:
            حالة GGML_OP_SOFT_MAX_پٹھ: {
                ونڈو_state = handle_generic(src_ss, /*scalar_only =*/ خطأ);
            } اكسر;
            حالة GGML_OP_ROPE: {
                ونڈو_state = handle_rope(src_ss);
            } اكسر;
            حالة GGML_OP_ROPE_پٹھ: {
                ونڈو_state = handle_rope(src_ss);
            } اكسر;
            حالة GGML_OP_CLAMP: {
                ونڈو_state = handle_generic(src_ss, /*scalar_only =*/ خطأ);
            } اكسر;
            حالة GGML_OP_CONV_TRANSPOSE_1D:
            حالة GGML_OP_IM2_خانہ:
            حالة GGML_OP_IM2_خانہ_پٹھ:
            حالة GGML_OP_IM2_خانہ_3_D:
            حالة GGML_OP_CONV_2D:
            حالة GGML_OP_CONV_3D:
            حالة GGML_OP_CONV_2D_DW:
            حالة GGML_OP_CONV_TRANSPOSE_2D:
            حالة GGML_OP_POOL_2D:
            حالة GGML_OP_POOL_2_D_پٹھ:
            حالة GGML_OP_UPSCALE: {
                ونڈو_state = handle_generic(src_ss, /*scalar_only =*/ سچ);
            } اكسر;
            حالة GGML_OP_POOL_1D: {
                ونڈو_state = handle_pool_1d(src_ss);
            } اكسر;
            حالة GGML_OP_PAD: {
                ونڈو_state = handle_pad(src_ss);
            } اكسر;
            حالة GGML_OP_PAD_REFLECT_1D:
            حالة GGML_OP_ROLL:
            حالة GGML_OP_ARANGE:
            حالة GGML_OP_TIMESTEP_EMBEDDING: {
                ونڈو_state = handle_generic(src_ss, /*scalar_only =*/ سچ);
            } اكسر;
            حالة GGML_OP_ARGSORT:
            حالة GGML_OP_چوٹی_K: {
                ونڈو_state = handle_فی_صف(src_ss);
            } اكسر;
            حالة GGML_OP_LEAKY_RELU: {
                ونڈو_state = handle_generic(src_ss, /*scalar_only =*/ خطأ);
            } اكسر;
            حالة GGML_OP_TRI: {
                ونڈو_state = handle_generic(src_ss, /*scalar_only =*/ سچ);
            } اكسر;
            حالة GGML_OP_FILL: {
                ونڈو_state = handle_generic(src_ss, /*scalar_only =*/ خطأ);
            } اكسر;
            حالة GGML_OP_FLASH_ATTN_EXT: {
                ونڈو_state = handle_flash_attn_ext(src_ss);
            } اكسر;
            حالة GGML_OP_FLASH_ATTN_پٹھ: {
                ونڈو_state = handle_generic(src_ss, /*scalar_only =*/ سچ);
            } اكسر;
            حالة GGML_OP_SSM_CONV: {
                ونڈو_state = handle_ssm_conv(src_ss);
            } اكسر;
            حالة GGML_OP_SSM_جانچ:
            حالة GGML_OP_WIN_PART:
            حالة GGML_OP_WIN_UNPART:
            حالة GGML_OP_GET_REL_POS:
            حالة GGML_OP_جوڑو_REL_POS:
            حالة GGML_OP_RWKV_WKV6:
            حالة GGML_OP_GATED_خطی_ATTN:
            حالة GGML_OP_RWKV_WKV7:
            حالة GGML_OP_SOLVE_TRI: {
                ونڈو_state = handle_generic(src_ss, /*scalar_only =*/ سچ);
            } اكسر;
            حالة GGML_OP_GATED_DELTA_NET: {
                ونڈو_state = handle_gated_delta_net(src_ss);
            } اكسر;
            حالة GGML_OP_LIGHTNING_INDEXER: {
                ونڈو_state = handle_lightning_indexer(src_ss);
            } اكسر;
            حالة GGML_OP_DSV4_HC_COMB:
            حالة GGML_OP_DSV4_HC_PRE:
            حالة GGML_OP_DSV4_HC_POST:
            حالة GGML_OP_چوٹی_K_نقاب: {
                ونڈو_state = handle_generic(src_ss, /*scalar_only =*/ سچ);
            } اكسر;
            حالة GGML_OP_UNARY: {
                ونڈو_state = handle_generic(src_ss, /*scalar_only =*/ خطأ);
            } اكسر;
            حالة GGML_OP_MAP_CUSTOM1:
            حالة GGML_OP_MAP_CUSTOM2:
            حالة GGML_OP_MAP_CUSTOM3:
            حالة GGML_OP_CUSTOM: {
                ونڈو_state = handle_generic(src_ss, /*scalar_only =*/ سچ);
            } اكسر;
            حالة GGML_OP_CROSS_ENTROPY_LOSS:
            حالة GGML_OP_CROSS_ENTROPY_LOSS_پٹھ: {
                ونڈو_state = handle_فی_صف(src_ss);
            } اكسر;
            حالة GGML_OP_OPT_قدم_ADAMW:
            حالة GGML_OP_OPT_قدم_SGD:
            حالة GGML_OP_GLU: {
                ونڈو_state = handle_generic(src_ss, /*scalar_only =*/ خطأ);
            } اكسر;
            افتراضي: {
                GGML_ABORT("ggml op not implemented: %s", ggml_op_name(tensor->op));
                ونڈو_state = {GGML_BACKEND_ونڈو_AXIS_UNKNOWN, {0}, {1}, 1};
            } اكسر;
        }
        إذا (ونڈو_state.axis >= 0 && ونڈو_state.axis < GGML_MAX_DIMS) {
            منطقي پہلا_src_ونڈو_by_axis = سچ;
            ثابت size_t n_bufs = ggml_backend_meta_بفر_n_bufs(tensor->بفر);

            لكل (size_t i = 0; i < GGML_MAX_SRC; i++) {
                إذا (tensor->src[i] == خالی_سوچک || src_ss[i].axis < 0 || src_ss[i].axis >= GGML_MAX_DIMS) {
                    استمر;
                }
                إذا (پہلا_src_ونڈو_by_axis) {
                    لكل (size_t j = 0; j < n_bufs; j++) {
                        // Take over ratio from src:
                        لكل (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            ونڈو_state.ne[s*n_bufs + j] = 0;
                        }
                        لكل (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            ونڈو_state.ne[j] += src_ss[i].ne[s*n_bufs + j] * src_ss[i].nr[s];
                        }
                        ونڈو_state.ne[j] *= tensor->ne[ونڈو_state.axis];
                        إذا (ونڈو_state.ne[j] != 0 || tensor->src[i]->ne[src_ss[i].axis] != 0) {
                            ثابت int64_t div = tensor->src[i]->ne[src_ss[i].axis] * ونڈو_state.nr[0];
                            GGML_ASSERT(ونڈو_state.ne[j] % div == 0);
                            ونڈو_state.ne[j] /= div;
                        }
                    }
                } وإلا {
                    GGML_ASSERT(ونڈو_state.n_segments == 1);
                    لكل (size_t j = 0; j < n_bufs; j++) {
                        // Assert that ratio is consistent:
                        int64_t جوڑ = 0;
                        لكل (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            جوڑ += src_ss[i].ne[s*n_bufs + j] * src_ss[i].nr[s];
                        }
                        GGML_ASSERT(ونڈو_state.ne[j]*ونڈو_state.nr[0] * tensor->src[i]->ne[src_ss[i].axis]
                                                                 == جوڑ * tensor->ne[ونڈو_state.axis]);
                    }
                }
                پہلا_src_ونڈو_by_axis = خطأ;
            }
            GGML_ASSERT(!پہلا_src_ونڈو_by_axis);
        }
        أرجع ونڈو_state;
    };

    ثابت std::pair کلید = std::make_pair(tensor, assume_sync);
    خودکار it = buf_ctx->ونڈو_state_cache.find(کلید);
    إذا (it != buf_ctx->ونڈو_state_cache.end() && memcmp(it->second.second, (ثابت حرف *) tensor, حجم(it->second.second)) != 0) {
        buf_ctx->ونڈو_state_cache.clear();
        it = buf_ctx->ونڈو_state_cache.end();
    }

    إذا (it == buf_ctx->ونڈو_state_cache.end()) {
        buf_ctx->ونڈو_state_cache[کلید].پہلا = calculate_ونڈو_state();
        حافظہ_نقل(buf_ctx->ونڈو_state_cache[کلید].second, tensor, حجم(buf_ctx->ونڈو_state_cache[کلید].second));
        إذا (buf_ctx->debug > 0) {
            std::string srcs_info;
            لكل (size_t i = 0; i < GGML_MAX_SRC; i++) {
                إذا (tensor->src[i] == خالی_سوچک || tensor->src[i] == tensor) {
                    استمر;
                }
                إذا (!srcs_info.خالی_تھاواں()) {
                    srcs_info += ", ";
                }
                ثابت ggml_backend_meta_ونڈو_state ونڈو_state =
                        ggml_backend_meta_get_ونڈو_state(tensor->src[i], سچ);
                GGML_ASSERT(ونڈو_state.n_segments == 1);
                ثابت حرف * axis_name = ggml_backend_meta_ونڈو_axis_name(ونڈو_state.axis);
                std::string ne_info;
                لكل (size_t j = 0; j < n_bufs; j++) {
                    إذا (!ne_info.خالی_تھاواں()) {
                        ne_info += ", ";
                    }
                    ne_info += std::to_string(ونڈو_state.ne[j]) + "x" + std::to_string(ونڈو_state.nr[0]);
                }
                srcs_info += std::string(tensor->src[i]->name) + "[" + ggml_op_name(tensor->src[i]->op) + ", " + axis_name + ", {" + ne_info + "}]";
            }
            std::string ne_info;
            لكل (size_t j = 0; j < n_bufs; j++) {
                إذا (!ne_info.خالی_تھاواں()) {
                    ne_info += ", ";
                }
                ثابت ggml_backend_meta_ونڈو_state & ss = buf_ctx->ونڈو_state_cache[کلید].پہلا;
                ne_info += std::to_string(ss.ne[j]) + "x" + std::to_string(ss.nr[0]);
            }
            GGML_LOG_DEBUG("SPLIT_STATE: {%s} -> %s[%s, %s, {%s}]\n", srcs_info.c_str(), tensor->name, ggml_op_name(tensor->op),
                ggml_backend_meta_ونڈو_axis_name(buf_ctx->ونڈو_state_cache[کلید].پہلا.axis), ne_info.c_str());
        }
    }

    ggml_backend_meta_ونڈو_state ret = buf_ctx->ونڈو_state_cache[کلید].پہلا;
    GGML_ASSERT(ret.axis != GGML_BACKEND_ونڈو_AXIS_NONE);
#جے_بنا_تعریف NDEBUG
    إذا (ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
        int64_t ne_ret = 0;
        لكل (size_t s = 0; s < ret.n_segments; s++) {
            لكل (size_t j = 0; j < n_bufs; j++) {
                ne_ret += ret.ne[s*n_bufs + j] * ret.nr[s];
            }
        }
        أكد(ne_ret == tensor->ne[صحيح(ret.axis)]);
    }
#جے_انت // NDEBUG
    أرجع ret;
}

ساكن منطقي ggml_backend_meta_is_host_view(ثابت بنية ggml_tensor * tensor) {
    أرجع ggml_is_view(tensor) && ggml_backend_بفر_is_host(tensor->view_src->بفر);
}

ساكن بنية ggml_backend_meta_ونڈو_state ggml_backend_meta_get_ونڈو_state(ثابت بنية ggml_tensor * tensor, منطقي assume_sync) {
    // [TAG_META_HOST_VIEWS]
    // TODO: technically, this check should not be needed if the backend scheduler correctly prevents assigning
    //       such host-buffer views to the meta backend. figure out how to update the scheduler logic to achieve that
    //       ref: https://github.com/ggml-org/llama.cpp/pull/30217
    إذا (!ggml_backend_بفر_is_meta(tensor->بفر)) {
        GGML_ASSERT(ggml_backend_meta_is_host_view(tensor));

        // the view is not allocated in the meta buffer, it is not split across the sub-devices
        أرجع { GGML_BACKEND_ونڈو_AXIS_NONE, {0}, {1}, 1 };
    }

    ggml_backend_meta_بفر_context * buf_ctx = (ggml_backend_meta_بفر_context *) tensor->بفر->context;
    أرجع ggml_backend_meta_get_ونڈو_state(buf_ctx->get_simple_tensor_container(tensor), tensor, assume_sync);
}

ساكن فراغ * ggml_backend_meta_بفر_get_ادھار(ggml_backend_بفر_t بفر) {
    GGML_بن_ورتیا(بفر);
    أرجع (فراغ *) 0x1000000000000000; // FIXME
}

ساكن تعداد ggml_حالت ggml_backend_meta_بفر_init_tensor_impl(ggml_backend_meta_simple_tensor_container & stc, ggml_tensor * tensor) {
    GGML_ASSERT(ggml_backend_بفر_is_meta(tensor->بفر));
    ggml_backend_meta_بفر_context * buf_ctx = (ggml_backend_meta_بفر_context *) tensor->بفر->context;
    ثابت size_t n_simple_bufs = ggml_backend_meta_بفر_n_bufs(tensor->بفر);

    ثابت ggml_backend_meta_ونڈو_state ونڈو_state = ggml_backend_meta_get_ونڈو_state(stc, tensor, /*assume_sync =*/ سچ);
    GGML_ASSERT(ggml_nelements(tensor) == 0 || ونڈو_state.axis != GGML_BACKEND_ونڈو_AXIS_UNKNOWN);
    GGML_ASSERT(ونڈو_state.n_segments <= 16);

    صحيح ونڈو_dim = ونڈو_state.axis;
    int64_t ne[GGML_MAX_DIMS];
    size_t  nb[GGML_MAX_DIMS];
    لكل (size_t k = 0; k < GGML_MAX_DIMS; k++) {
        ne[k] = tensor->ne[k];
        nb[k] = tensor->nb[k];
    }

    std::vector<ggml_tensor *> simple_tensors;
    simple_tensors.reserve(n_simple_bufs);
    لكل (size_t j = 0; j < n_simple_bufs; j++) {
        ggml_context          * simple_ctx = stc.ctxs[j].get();
        ggml_backend_بفر_t   simple_buf = buf_ctx->bufs[j].get();

        إذا (ونڈو_dim >= 0 && ونڈو_dim < GGML_MAX_DIMS) {
            // TODO: the following assert fails for llama-parallel even though the results are correct:
            // GGML_ASSERT(ggml_is_contiguously_allocated(tensor));
            ne[ونڈو_dim] = 0;
            لكل (size_t s = 0; s < ونڈو_state.n_segments; s++) {
                ne[ونڈو_dim] += ونڈو_state.ne[s*n_simple_bufs + j] * ونڈو_state.nr[s];
            }
            لكل (صحيح i = 0; i < GGML_MAX_DIMS; i++) {
                إذا (tensor->nb[i] > tensor->nb[ونڈو_dim]) {
                    nb[i] = tensor->nb[i] * ne[ونڈو_dim]/tensor->ne[ونڈو_dim];
                }
            }
        }

        ggml_tensor * t_ij = ggml_new_tensor(simple_ctx, tensor->type, GGML_MAX_DIMS, ne);
        t_ij->op = tensor->op;
        لكل (صحيح i = 0; i < GGML_MAX_DIMS; i++) {
            t_ij->nb[i] = nb[i];
        }
        t_ij->flags = tensor->flags;
        حافظہ_نقل(t_ij->op_params, tensor->op_params, حجم(tensor->op_params));
        ggml_سیٹ_name(t_ij, tensor->name);
        t_ij->بفر = simple_buf;
        t_ij->view_src = tensor->view_src;
        t_ij->view_offs = tensor->view_offs;
        إذا (t_ij->view_src != خالی_سوچک && ggml_backend_بفر_is_meta(t_ij->view_src->بفر)) {
            t_ij->view_src = ggml_backend_meta_بفر_simple_tensor(tensor->view_src, j);
            إذا (t_ij->view_offs > 0 && ونڈو_dim >= 0 && ونڈو_dim < GGML_MAX_DIMS) {
                GGML_ASSERT(tensor->ne[ونڈو_dim] != 0);
                ثابت صحيح ونڈو_dim_view_src = ggml_backend_meta_get_ونڈو_state(tensor->view_src, /*assume_sync =*/ سچ).axis;
                GGML_ASSERT(ونڈو_dim_view_src >= 0 && ونڈو_dim_view_src < GGML_MAX_DIMS);

                // The offset can be internal to the data split, in those cases the view offset should not be scaled.
                // If however, the offset is larger than the data split then it needs to be scaled proportionally.
                منطقي ونڈو_اندرونی_وتھ = t_ij->view_offs <= tensor->view_src->nb[ونڈو_dim_view_src];
                لكل (صحيح i = 0; i < GGML_MAX_DIMS; i++) {
                    ثابت size_t dim_ناپ = tensor->ne[i] * tensor->nb[i];
                    إذا (tensor->view_offs <= dim_ناپ && dim_ناپ < tensor->nb[ونڈو_dim]) {
                        ونڈو_اندرونی_وتھ = سچ;
                        اكسر;
                    }
                }
                إذا (!ونڈو_اندرونی_وتھ) {
                    t_ij->view_offs = t_ij->view_offs * ne[ونڈو_dim]/tensor->ne[ونڈو_dim];
                }
            }
        }
        // TODO: revisit once the graph allocator has been refactored, see https://github.com/ggml-org/llama.cpp/pull/25051#issuecomment-4842873396
        ggml_backend_بفر_t init_buf = simple_buf;
        إذا (t_ij->view_src != خالی_سوچک) {
            t_ij->انکڑے = (حرف *) t_ij->view_src->انکڑے + t_ij->view_offs;
            // views inherit the source slice's concrete sub-buffer (issue 22197)
            إذا (tensor->view_src != خالی_سوچک && ggml_backend_بفر_is_meta(tensor->view_src->بفر)
                    && t_ij->view_src->بفر != خالی_سوچک) {
                t_ij->بفر = t_ij->view_src->بفر;
                init_buf     = t_ij->view_src->بفر;
            }
        } وإلا إذا (simple_buf != خالی_سوچک) {
            إذا (ggml_backend_بفر_is_multi_بفر(simple_buf)) {
                GGML_ABORT("multi buffers are not supported by the meta backend");
            }
            t_ij->انکڑے = (حرف *) ggml_backend_بفر_get_ادھار(simple_buf)
                + size_t(tensor->انکڑے) - size_t(ggml_backend_بفر_get_ادھار(tensor->بفر));
        }

        إذا (init_buf) {
            // the backend that owns the buffer will set .extra
            ggml_backend_بفر_init_tensor(init_buf, t_ij);
        } وإلا {
            t_ij->extra = tensor->extra;
        }

        لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
            t_ij->src[i] = tensor->src[i];
            إذا (tensor->src[i] == tensor) {
                t_ij->src[i] = t_ij;
            } وإلا إذا (t_ij->src[i] != خالی_سوچک && ggml_backend_بفر_is_meta(t_ij->src[i]->بفر)) {
                t_ij->src[i] = ggml_backend_meta_بفر_simple_tensor(tensor->src[i], j);
            }
        }

        simple_tensors.push_back(t_ij);
    }

    // If one of the sources has a zero-sized slice, disable the computation:
    لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
        إذا (tensor->src[i] == خالی_سوچک || !ggml_backend_بفر_is_meta(tensor->src[i]->بفر)) {
            استمر;
        }

        ثابت ggml_backend_meta_ونڈو_state ونڈو_state_src = ggml_backend_meta_get_ونڈو_state(tensor->src[i], /*assume_sync =*/ سچ);
        إذا (ونڈو_state_src.axis < 0 || ونڈو_state_src.axis >= GGML_MAX_DIMS) {
            استمر;
        }
        لكل (size_t j = 0; j < n_simple_bufs; j++) {
            int64_t ne_جوڑ = 0;
            لكل (size_t s = 0; s < ونڈو_state_src.n_segments; s++) {
                ne_جوڑ += ونڈو_state_src.ne[s*n_simple_bufs + j] * ونڈو_state_src.nr[s];
            }
            إذا (ne_جوڑ == 0) {
                simple_tensors[j]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
            }
        }
    }

    stc.simple_tensors[tensor] = simple_tensors;

    أرجع GGML_حالت_SUCCESS;
}

ساكن تعداد ggml_حالت ggml_backend_meta_بفر_init_tensor(ggml_backend_بفر_t بفر, ggml_tensor * tensor) {
    GGML_ASSERT(ggml_backend_بفر_is_meta(بفر));
    ggml_backend_meta_بفر_context * buf_ctx = (ggml_backend_meta_بفر_context *) بفر->context;
    buf_ctx->stc_compute_اشاریہ = buf_ctx->stc_compute_اشاریہ_اگلا;
    أرجع ggml_backend_meta_بفر_init_tensor_impl(buf_ctx->get_simple_tensor_container(tensor), tensor);
}

ساكن فراغ ggml_backend_meta_بفر_memset_tensor(
        ggml_backend_بفر_t بفر, ggml_tensor * tensor, uint8_t value, size_t وتھ, size_t size) {
    ثابت size_t n_bufs = ggml_backend_meta_بفر_n_bufs(بفر);
    ثابت ggml_backend_meta_ونڈو_state ونڈو_state =
            ggml_backend_meta_get_ونڈو_state(tensor, /*assume_sync =*/ خطأ);
    GGML_ASSERT(ggml_is_contiguous(tensor) || ونڈو_state.axis == GGML_BACKEND_ونڈو_AXIS_MIRRORED);

    إذا (ونڈو_state.n_segments != 1 || ونڈو_state.nr[0] != 1) {
        GGML_ASSERT(ونڈو_state.axis >= 0 && ونڈو_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(ونڈو_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        std::vector<size_t> simple_offsets(n_bufs, 0);
        إذا (ونڈو_state.axis == GGML_BACKEND_ونڈو_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            ثابت size_t صف_stride = tensor->nb[1];
            GGML_ASSERT(وتھ % صف_stride == 0);
            GGML_ASSERT(size   % صف_stride == 0);
            ثابت int64_t صف_شروع = وتھ / صف_stride;
            ثابت int64_t صف_گنتی = size   / صف_stride;
            GGML_ASSERT(صف_شروع + صف_گنتی <= tensor->ne[1]);

            ثابت int64_t blck_ناپ = ggml_blck_ناپ(tensor->type);
            لكل (size_t s = 0; s < ونڈو_state.n_segments; s++) {
                لكل (size_t r = 0; r < ونڈو_state.nr[s]; r++) {
                    لكل (size_t j = 0; j < n_bufs; j++) {
                        ggml_tensor * simple_tensor = ggml_backend_meta_بفر_simple_tensor(tensor, j);
                        GGML_ASSERT(ونڈو_state.ne[s*n_bufs + j] % blck_ناپ == 0);
                        ثابت size_t nbytes = ونڈو_state.ne[s*n_bufs + j]/blck_ناپ * tensor->nb[0];
                        لكل (int64_t صف = 0; صف < صف_گنتی; صف++) {
                            ggml_backend_tensor_memset(simple_tensor, value,
                                    simple_offsets[j] + (صف_شروع + صف)*simple_tensor->nb[1], nbytes);
                        }
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            أرجع;
        }

        GGML_ASSERT(ونڈو_state.axis == GGML_BACKEND_ونڈو_AXIS_1);

        ثابت size_t صف_stride = tensor->nb[2];
        GGML_ASSERT(وتھ % صف_stride == 0);
        GGML_ASSERT(size   % صف_stride == 0);
        ثابت int64_t صف_شروع = وتھ / صف_stride;
        ثابت int64_t صف_گنتی = size   / صف_stride;
        GGML_ASSERT(صف_شروع + صف_گنتی <= tensor->ne[2]);

        لكل (size_t s = 0; s < ونڈو_state.n_segments; s++) {
            لكل (size_t r = 0; r < ونڈو_state.nr[s]; r++) {
                لكل (size_t j = 0; j < n_bufs; j++) {
                    ggml_tensor * simple_tensor = ggml_backend_meta_بفر_simple_tensor(tensor, j);
                    ثابت size_t nbytes = ونڈو_state.ne[s*n_bufs + j] * tensor->nb[1];
                    لكل (int64_t صف = 0; صف < صف_گنتی; صف++) {
                        ggml_backend_tensor_memset(simple_tensor, value,
                                simple_offsets[j] + (صف_شروع + صف)*simple_tensor->nb[2], nbytes);
                    }
                    simple_offsets[j] += nbytes;
                }
            }
        }
        أرجع;
    }

    بدل (ونڈو_state.axis) {
        حالة GGML_BACKEND_ونڈو_AXIS_0:
        حالة GGML_BACKEND_ونڈو_AXIS_1:
        حالة GGML_BACKEND_ونڈو_AXIS_2: {
            ثابت size_t chunk_ناپ_بھریا = tensor->nb[ونڈو_state.axis + 1];
            GGML_ASSERT(وتھ % chunk_ناپ_بھریا == 0);
            GGML_ASSERT(size   % chunk_ناپ_بھریا == 0);
            ثابت int64_t i_شروع =  وتھ        / chunk_ناپ_بھریا;
            ثابت int64_t i_stop  = (وتھ + size) / chunk_ناپ_بھریا;
            لكل (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_بفر_simple_tensor(tensor, j);
                ثابت size_t chunk_ناپ = simple_tensor->nb[ونڈو_state.axis + 1];
                إذا (chunk_ناپ == 0) {
                    استمر;
                }
                لكل (int64_t i = i_شروع; i < i_stop; i++) {
                    ggml_backend_tensor_memset(simple_tensor, value, i*chunk_ناپ, chunk_ناپ);
                }
            }
        } اكسر;
        حالة GGML_BACKEND_ونڈو_AXIS_PARTIAL: {
            GGML_ASSERT(value == 0);
            [[fallthrough]];
        }
        حالة GGML_BACKEND_ونڈو_AXIS_MIRRORED: {
            لكل (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_بفر_simple_tensor(tensor, j);
                ggml_backend_tensor_memset(simple_tensor, value, وتھ, size);
            }
        } اكسر;
        افتراضي: {
            GGML_ABORT("fatal error");
        }
    }
}

ساكن فراغ ggml_backend_meta_بفر_سیٹ_tensor(ggml_backend_بفر_t بفر, ggml_tensor * tensor, ثابت فراغ * انکڑے, size_t وتھ, size_t size) {
    ثابت size_t n_bufs = ggml_backend_meta_بفر_n_bufs(بفر);
    ثابت ggml_backend_meta_ونڈو_state ونڈو_state = ggml_backend_meta_get_ونڈو_state(tensor, /*assume_sync =*/ خطأ);
    GGML_ASSERT(ggml_is_contiguous(tensor) || ونڈو_state.axis == GGML_BACKEND_ونڈو_AXIS_MIRRORED);

    إذا (ونڈو_state.n_segments != 1 || ونڈو_state.nr[0] != 1) {
        GGML_ASSERT(ونڈو_state.axis >= 0 && ونڈو_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(ونڈو_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        size_t وتھ_انکڑے = 0;
        std::vector<size_t> simple_offsets(n_bufs, 0);
        إذا (ونڈو_state.axis == GGML_BACKEND_ونڈو_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            ثابت size_t صف_stride = tensor->nb[1];
            GGML_ASSERT(وتھ % صف_stride == 0);
            GGML_ASSERT(size   % صف_stride == 0);
            ثابت int64_t صف_شروع = وتھ / صف_stride;
            ثابت int64_t صف_گنتی = size   / صف_stride;
            GGML_ASSERT(صف_شروع + صف_گنتی <= tensor->ne[1]);

            ثابت int64_t blck_ناپ = ggml_blck_ناپ(tensor->type);
            لكل (size_t s = 0; s < ونڈو_state.n_segments; s++) {
                لكل (size_t r = 0; r < ونڈو_state.nr[s]; r++) {
                    لكل (size_t j = 0; j < n_bufs; j++) {
                        ggml_tensor * simple_tensor = ggml_backend_meta_بفر_simple_tensor(tensor, j);
                        GGML_ASSERT(ونڈو_state.ne[s*n_bufs + j] % blck_ناپ == 0);
                        ثابت size_t nbytes = ونڈو_state.ne[s*n_bufs + j]/blck_ناپ * tensor->nb[0];
                        ggml_backend_tensor_سیٹ_2d(simple_tensor, (ثابت حرف *) انکڑے + وتھ_انکڑے,
                            simple_offsets[j] + صف_شروع * simple_tensor->nb[1], nbytes,
                            صف_گنتی, simple_tensor->nb[1], tensor->nb[1]);
                        وتھ_انکڑے       += nbytes;
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            GGML_ASSERT(وتھ_انکڑے*صف_گنتی == size);
            أرجع;
        }
        GGML_ASSERT(ونڈو_state.axis == GGML_BACKEND_ونڈو_AXIS_1);

        ثابت size_t صف_stride = tensor->nb[2];
        GGML_ASSERT(وتھ % صف_stride == 0);
        GGML_ASSERT(size   % صف_stride == 0);
        ثابت int64_t صف_شروع = وتھ / صف_stride;
        ثابت int64_t صف_گنتی = size   / صف_stride;
        GGML_ASSERT(صف_شروع + صف_گنتی <= tensor->ne[2]);

        لكل (size_t s = 0; s < ونڈو_state.n_segments; s++) {
            لكل (size_t r = 0; r < ونڈو_state.nr[s]; r++) {
                لكل (size_t j = 0; j < n_bufs; j++) {
                    ggml_tensor * simple_tensor = ggml_backend_meta_بفر_simple_tensor(tensor, j);
                    ثابت size_t nbytes = ونڈو_state.ne[s*n_bufs + j] * tensor->nb[1];
                    ggml_backend_tensor_سیٹ_2d(simple_tensor, (ثابت حرف *) انکڑے + وتھ_انکڑے,
                        simple_offsets[j] + صف_شروع * simple_tensor->nb[2], nbytes,
                        صف_گنتی, simple_tensor->nb[2], tensor->nb[2]);
                    وتھ_انکڑے       += nbytes;
                    simple_offsets[j] += nbytes;
                }
            }
        }
        GGML_ASSERT(وتھ_انکڑے*صف_گنتی == size);
        أرجع;
    }

    بدل (ونڈو_state.axis) {
        حالة GGML_BACKEND_ونڈو_AXIS_0:
        حالة GGML_BACKEND_ونڈو_AXIS_1:
        حالة GGML_BACKEND_ونڈو_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            ثابت size_t chunk_ناپ_بھریا = tensor->nb[ونڈو_state.axis + 1];
            GGML_ASSERT(وتھ % chunk_ناپ_بھریا == 0);
            GGML_ASSERT(size   % chunk_ناپ_بھریا == 0);
            ثابت int64_t i_شروع =  وتھ        /chunk_ناپ_بھریا;
            ثابت int64_t i_stop  = (وتھ + size)/chunk_ناپ_بھریا;
            size_t وتھ_j = 0;
            لكل (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_بفر_simple_tensor(tensor, j);
                ثابت size_t chunk_ناپ_j = simple_tensor->nb[ونڈو_state.axis + 1];
                إذا (chunk_ناپ_j == 0) {
                    استمر;
                }
                ثابت size_t simple_وتھ = i_شروع * chunk_ناپ_j;
                ggml_backend_tensor_سیٹ_2d(simple_tensor, (ثابت حرف *) انکڑے + وتھ_j, simple_وتھ, chunk_ناپ_j, i_stop - i_شروع, chunk_ناپ_j, chunk_ناپ_بھریا);
                وتھ_j += chunk_ناپ_j;
            }
            GGML_ASSERT(وتھ_j == chunk_ناپ_بھریا);
        } اكسر;
        حالة GGML_BACKEND_ونڈو_AXIS_MIRRORED: {
            لكل (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_بفر_simple_tensor(tensor, j);
                ggml_backend_tensor_سیٹ(simple_tensor, انکڑے, وتھ, size);
            }
        } اكسر;
        حالة GGML_BACKEND_ونڈو_AXIS_PARTIAL: {
            GGML_ASSERT(tensor->type == GGML_TYPE_F32);
            GGML_ASSERT(وتھ % حجم(عائم) == 0);
            GGML_ASSERT(size   % حجم(عائم) == 0);
            ثابت size_t n_اقدار = size / حجم(عائم);
            size_t n_contributors = 0;
            لكل (size_t j = 0; j < n_bufs; j++) {
                n_contributors += ونڈو_state.ne[j] != 0;
            }
            ثابت منطقي has_contributor_نقاب = n_contributors != 0;
            إذا (!has_contributor_نقاب) {
                n_contributors = n_bufs;
            }
            std::vector<عائم> tmp(n_اقدار);
            لكل (size_t i = 0; i < n_اقدار; i++) {
                tmp[i] = ((ثابت عائم *) انکڑے)[i] / n_contributors;
            }
            std::vector<عائم> zero;
            إذا (has_contributor_نقاب) {
                zero.resize(n_اقدار, 0.0f);
            }
            لكل (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_بفر_simple_tensor(tensor, j);
                ثابت عائم * partial = has_contributor_نقاب && ونڈو_state.ne[j] == 0 ? zero.انکڑے() : tmp.انکڑے();
                ggml_backend_tensor_سیٹ(simple_tensor, partial, وتھ, size);
            }
        } اكسر;
        افتراضي: {
            GGML_ABORT("fatal error");
        }
    }
}

ساكن فراغ ggml_backend_meta_بفر_get_tensor(ggml_backend_بفر_t بفر, ثابت ggml_tensor * tensor, فراغ * انکڑے, size_t وتھ, size_t size) {
    ثابت size_t n_bufs = ggml_backend_meta_بفر_n_bufs(بفر);
    ثابت ggml_backend_meta_ونڈو_state ونڈو_state = ggml_backend_meta_get_ونڈو_state(tensor, /*assume_sync =*/ خطأ);
    GGML_ASSERT(ggml_is_contiguous(tensor) || ونڈو_state.axis == GGML_BACKEND_ونڈو_AXIS_MIRRORED);

    إذا (ونڈو_state.n_segments != 1 || ونڈو_state.nr[0] != 1) {
        GGML_ASSERT(ونڈو_state.axis >= 0 && ونڈو_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(ونڈو_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        size_t وتھ_انکڑے = 0;
        std::vector<size_t> simple_offsets(n_bufs, 0);
        إذا (ونڈو_state.axis == GGML_BACKEND_ونڈو_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            ثابت size_t صف_stride = tensor->nb[1];
            GGML_ASSERT(وتھ % صف_stride == 0);
            GGML_ASSERT(size   % صف_stride == 0);
            ثابت int64_t صف_شروع = وتھ / صف_stride;
            ثابت int64_t صف_گنتی = size   / صف_stride;
            GGML_ASSERT(صف_شروع + صف_گنتی <= tensor->ne[1]);

            ثابت int64_t blck_ناپ = ggml_blck_ناپ(tensor->type);
            لكل (size_t s = 0; s < ونڈو_state.n_segments; s++) {
                لكل (size_t r = 0; r < ونڈو_state.nr[s]; r++) {
                    لكل (size_t j = 0; j < n_bufs; j++) {
                        ثابت ggml_tensor * simple_tensor = ggml_backend_meta_بفر_simple_tensor(tensor, j);
                        GGML_ASSERT(ونڈو_state.ne[s*n_bufs + j] % blck_ناپ == 0);
                        ثابت size_t nbytes = ونڈو_state.ne[s*n_bufs + j]/blck_ناپ * tensor->nb[0];
                        ggml_backend_tensor_get_2d(simple_tensor, (حرف *) انکڑے + وتھ_انکڑے,
                            simple_offsets[j] + صف_شروع * simple_tensor->nb[1], nbytes,
                            صف_گنتی, simple_tensor->nb[1], tensor->nb[1]);
                        وتھ_انکڑے       += nbytes;
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            GGML_ASSERT(وتھ_انکڑے*صف_گنتی == size);
            أرجع;
        }
        GGML_ASSERT(ونڈو_state.axis == GGML_BACKEND_ونڈو_AXIS_1);

        ثابت size_t صف_stride = tensor->nb[2];
        GGML_ASSERT(وتھ % صف_stride == 0);
        GGML_ASSERT(size   % صف_stride == 0);
        ثابت int64_t صف_شروع = وتھ / صف_stride;
        ثابت int64_t صف_گنتی = size   / صف_stride;
        GGML_ASSERT(صف_شروع + صف_گنتی <= tensor->ne[2]);

        لكل (size_t s = 0; s < ونڈو_state.n_segments; s++) {
            لكل (size_t r = 0; r < ونڈو_state.nr[s]; r++) {
                لكل (size_t j = 0; j < n_bufs; j++) {
                    ثابت ggml_tensor * simple_tensor = ggml_backend_meta_بفر_simple_tensor(tensor, j);
                    ثابت size_t nbytes = ونڈو_state.ne[s*n_bufs + j] * tensor->nb[1];
                    ggml_backend_tensor_get_2d(simple_tensor, (حرف *) انکڑے + وتھ_انکڑے,
                        simple_offsets[j] + صف_شروع * simple_tensor->nb[2], nbytes,
                        صف_گنتی, simple_tensor->nb[2], tensor->nb[2]);
                    وتھ_انکڑے       += nbytes;
                    simple_offsets[j] += nbytes;
                }
            }
        }
        GGML_ASSERT(وتھ_انکڑے*صف_گنتی == size);
        أرجع;
    }

    بدل (ونڈو_state.axis) {
        حالة GGML_BACKEND_ونڈو_AXIS_0:
        حالة GGML_BACKEND_ونڈو_AXIS_1:
        حالة GGML_BACKEND_ونڈو_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            ثابت size_t chunk_ناپ_بھریا = tensor->nb[ونڈو_state.axis + 1];
            GGML_ASSERT(وتھ % chunk_ناپ_بھریا == 0);
            GGML_ASSERT(size   % chunk_ناپ_بھریا == 0);
            ثابت int64_t i_شروع =  وتھ        /chunk_ناپ_بھریا;
            ثابت int64_t i_stop  = (وتھ + size)/chunk_ناپ_بھریا;
            size_t وتھ_j = 0;
            لكل (size_t j = 0; j < n_bufs; j++){
                ثابت ggml_tensor * simple_tensor = ggml_backend_meta_بفر_simple_tensor(tensor, j);
                ثابت size_t chunk_ناپ_j = simple_tensor->nb[ونڈو_state.axis + 1];
                إذا (chunk_ناپ_j == 0) {
                    استمر;
                }
                ثابت size_t simple_وتھ = i_شروع * chunk_ناپ_j;
                ggml_backend_tensor_get_2d(simple_tensor, (حرف *) انکڑے + وتھ_j, simple_وتھ, chunk_ناپ_j, i_stop - i_شروع, chunk_ناپ_j, chunk_ناپ_بھریا);
                وتھ_j += chunk_ناپ_j;
            }
            GGML_ASSERT(وتھ_j == chunk_ناپ_بھریا);
        } اكسر;
        حالة GGML_BACKEND_ونڈو_AXIS_MIRRORED: {
            // TODO other simple backend may be better
            ثابت ggml_tensor * simple_tensor = ggml_backend_meta_بفر_simple_tensor(tensor, 0);
            ggml_backend_tensor_get(simple_tensor, انکڑے, وتھ, size);
        } اكسر;
        افتراضي: {
            GGML_ABORT("fatal error");
        }
    }
}

ساكن فراغ ggml_backend_meta_بفر_clear(ggml_backend_بفر_t بفر, uint8_t value) {
    ثابت size_t n_buffers = ggml_backend_meta_بفر_n_bufs(بفر);
    لكل (size_t i = 0; i < n_buffers; i++) {
        ggml_backend_بفر_clear(ggml_backend_meta_بفر_simple_بفر(بفر, i), value);
    }
}

ساكن فراغ ggml_backend_meta_بفر_reset(ggml_backend_بفر_t بفر) {
    GGML_ASSERT(ggml_backend_بفر_is_meta(بفر));
    ggml_backend_meta_بفر_context * buf_ctx = (ggml_backend_meta_بفر_context *) بفر->context;
    لكل (size_t i = 0; i < buf_ctx->bufs.size(); i++) {
        ggml_backend_بفر_reset(ggml_backend_meta_بفر_simple_بفر(بفر, i));
    }
}

ساكن ثابت ggml_backend_بفر_i ggml_backend_meta_بفر_iface = {
    /* .free_buffer     = */ ggml_backend_meta_بفر_free_بفر,
    /* .get_base        = */ ggml_backend_meta_بفر_get_ادھار,
    /* .init_tensor     = */ ggml_backend_meta_بفر_init_tensor,
    /* .memset_tensor   = */ ggml_backend_meta_بفر_memset_tensor,
    /* .set_tensor      = */ ggml_backend_meta_بفر_سیٹ_tensor,
    /* .get_tensor      = */ ggml_backend_meta_بفر_get_tensor,
    /* .set_tensor_2d   = */ خالی_سوچک,
    /* .get_tensor_2d   = */ خالی_سوچک,
    /* .cpy_tensor      = */ خالی_سوچک,
    /* .clear           = */ ggml_backend_meta_بفر_clear,
    /* .reset           = */ ggml_backend_meta_بفر_reset,
};

منطقي ggml_backend_بفر_is_meta(ggml_backend_بفر_t buf) {
    أرجع buf != خالی_سوچک && buf->iface.free_بفر == ggml_backend_meta_بفر_iface.free_بفر;
}

فراغ ggml_backend_meta_بفر_سیٹ_usage(ggml_backend_بفر_t بفر, تعداد ggml_backend_بفر_usage usage) {
    GGML_ASSERT(ggml_backend_بفر_is_meta(بفر));
    ggml_backend_meta_بفر_context * buf_ctx = (ggml_backend_meta_بفر_context *) بفر->context;
    لكل (size_t i = 0; i < buf_ctx->bufs.size(); i++) {
        إذا (buf_ctx->bufs[i]) {
            ggml_backend_بفر_سیٹ_usage(buf_ctx->bufs[i].get(), usage);
        }
    }
}

ساكن ggml_backend_بفر_t ggml_backend_meta_بفر_type_alloc_بفر(ggml_backend_بفر_type_t buft, size_t size) {
    ثابت size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);

    ثابت ggml_init_params params = {
        /*.mem_size   =*/ 1024*1024*ggml_tensor_overhead(), // FIXME
        /*.mem_buffer =*/ خالی_سوچک,
        /*.no_alloc   =*/ سچ,
    };
    ggml_backend_meta_simple_tensor_container stc_static;
    ggml_backend_meta_simple_tensor_container stc_compute_0(params, n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_1(params, n_simple_bufts);

    size_t max_size = 0;
    std::vector<ggml_backend_بفر_t> bufs;
    bufs.reserve(n_simple_bufts);
    لكل (size_t i = 0; i < n_simple_bufts; i++) {
        bufs.push_back(ggml_backend_buft_alloc_بفر(ggml_backend_meta_buft_simple_buft(buft, i), size));
        GGML_ASSERT(bufs.back() != خالی_سوچک);
        max_size = std::max(max_size, ggml_backend_بفر_get_ناپ(bufs.back()));
    }
    ggml_backend_meta_بفر_context * buf_ctx = جديد ggml_backend_meta_بفر_context(stc_static, stc_compute_0, stc_compute_1, bufs);

    أرجع ggml_backend_بفر_init(buft, ggml_backend_meta_بفر_iface, buf_ctx, max_size);
}

ساكن ggml_backend_بفر_t ggml_backend_meta_بفر_type_alloc_بفر_n(ggml_backend_بفر_type_t buft, ggml_tensor ** tensors, صحيح n_tensors) {
    ثابت size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);

    constexpr size_t compute_headroom = 16; // Maximum number of views per statically allocated tensor that can be created between evals.
    ثابت ggml_init_params params_static = {
        /*.mem_size   =*/ n_tensors * ggml_tensor_overhead(),
        /*.mem_buffer =*/ خالی_سوچک,
        /*.no_alloc   =*/ سچ,
    };
    ثابت ggml_init_params params_compute = {
        /*.mem_size   =*/ compute_headroom * n_tensors * ggml_tensor_overhead(),
        /*.mem_buffer =*/ خالی_سوچک,
        /*.no_alloc   =*/ سچ,
    };
    ggml_backend_meta_simple_tensor_container stc_static   (params_static,  n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_0(params_compute, n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_1(params_compute, n_simple_bufts);

    std::vector<ggml_backend_بفر_t> bufs(n_simple_bufts, خالی_سوچک);
    ggml_backend_meta_بفر_context * meta_buf_ctx = جديد ggml_backend_meta_بفر_context(stc_static, stc_compute_0, stc_compute_1, bufs);

    ggml_backend_بفر_t meta_buf = ggml_backend_بفر_init(buft, ggml_backend_meta_بفر_iface, meta_buf_ctx, 0);
    لكل (صحيح i = 0; i < n_tensors; i++) {
        ggml_tensor * t = tensors[i];
        t->بفر = meta_buf;
        ggml_backend_meta_بفر_init_tensor_impl(meta_buf_ctx->stc_static, t);
        t->انکڑے = (فراغ *) 0x2000000000000000; // FIXME
    }
    لكل (size_t i = 0; i < n_simple_bufts; i++) {
        ggml_context * ctx = meta_buf_ctx->stc_static.ctxs[i].get();
        ggml_backend_بفر_type_t simple_buft = ggml_backend_meta_buft_simple_buft(buft, i);

        // If a ggml_context only has zero-sized tensors, ggml_backend_alloc_ctx_tensors_from_buft returns NULL.
        // For those edge cases, allocate a dummy buffer instead.
        منطقي any_nonzero_ٹکڑا = خطأ;
        لكل (ggml_tensor * t = ggml_get_پہلا_tensor(ctx); t != خالی_سوچک; t = ggml_get_اگلا_tensor(ctx, t)) {
            إذا (ggml_nelements(t) != 0) {
                any_nonzero_ٹکڑا = سچ;
                اكسر;
            }
        }
        إذا (any_nonzero_ٹکڑا) {
            meta_buf_ctx->bufs[i].reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx, simple_buft));
        } وإلا {
            meta_buf_ctx->bufs[i].reset(ggml_backend_buft_alloc_بفر(simple_buft, 0));
            لكل (ggml_tensor * t = ggml_get_پہلا_tensor(ctx); t != خالی_سوچک; t = ggml_get_اگلا_tensor(ctx, t)) {
                t->بفر = meta_buf_ctx->bufs[i].get();
            }
        }
        GGML_ASSERT(meta_buf_ctx->bufs[i]);
        meta_buf->size = std::max(meta_buf->size, ggml_backend_بفر_get_ناپ(meta_buf_ctx->bufs[i].get()));
    }
    أرجع meta_buf;
}

//
// meta backend
//

ساكن ggml_guid_t ggml_backend_meta_guid() {
    ساكن ggml_guid guid = {0xf1, 0x0e, 0x34, 0xcf, 0x9c, 0x6f, 0x43, 0xcb, 0x96, 0x92, 0xbe, 0x8e, 0xbb, 0x71, 0x3f, 0xda};
    أرجع &guid;
}

بنية ggml_backend_meta_context {
    بنية cgraph_config {
        ggml_cgraph * cgraph_main = خالی_سوچک;
        صحيح           وتھ      = 0; // Node offset vs. original graph

        std::vector<ggml_cgraph *> cgraphs_aux;
    };
    بنية backend_config {
        ggml_backend_t backend;

        std::vector<cgraph_config>           cgraphs;
        std::vector<ggml_tensor *>           nodes;
        std::vector<ggml_backend_بفر_ptr> bufs;

        backend_config(ggml_backend_t backend, ثابت size_t n_reduce_قدماں) : backend(backend) {
            bufs.resize(n_reduce_قدماں);
        }
    };
    std::string                 name;
    std::vector<backend_config> backend_configs;
    ggml_context_ptr            ctx;
    std::vector<ggml_cgraph *>  cgraphs_aux;
    std::vector<ggml_tensor *>  nodes_aux;
    size_t                      n_reduce_قدماں;
    صحيح                         max_nnodes    = 0;
    size_t                      max_tmp_ناپ  = 0;
    size_t                      max_subgraphs = 0;
    size_t                      n_subgraphs   = 0;
    uint64_t                    uid           = 0;

    فراغ *                               comm_ctx       = خالی_سوچک;
    ggml_backend_comm_allreduce_tensor_t comm_allreduce = خالی_سوچک;

    ggml_backend_meta_context(ggml_backend_dev_t meta_dev, ثابت حرف * params) {
        ثابت size_t n_devs = ggml_backend_meta_dev_n_devs(meta_dev);
        n_reduce_قدماں = std::ceil(std::log2(n_devs));
        name = "Meta(";
        std::vector<ggml_backend_t> simple_backends;
        backend_configs.reserve(n_devs);
        simple_backends.reserve(n_devs);
        لكل (size_t i = 0; i < n_devs; i++) {
            ggml_backend_dev_t simple_dev = ggml_backend_meta_dev_simple_dev(meta_dev, i);
            إذا (i > 0) {
                name += ",";
            }
            name += ggml_backend_dev_name(simple_dev);
            simple_backends.push_back(ggml_backend_dev_init(simple_dev, params));
            backend_configs.emplace_back(simple_backends.back(), n_reduce_قدماں);
        }
        name += ")";

        إذا (n_devs > 1) {
            ggml_backend_comm_init_t comm_init = (ggml_backend_comm_init_t) ggml_backend_reg_get_proc_پتہ(
                ggml_backend_dev_backend_reg(ggml_backend_get_device(simple_backends[0])), "ggml_backend_comm_init");
            إذا (comm_init != خالی_سوچک) {
                comm_ctx = comm_init(simple_backends.انکڑے(), simple_backends.size());
            }
        }
        إذا (comm_ctx != خالی_سوچک) {
            comm_allreduce = (ggml_backend_comm_allreduce_tensor_t)
                ggml_backend_reg_get_proc_پتہ(ggml_backend_dev_backend_reg(
                    ggml_backend_get_device(simple_backends[0])), "ggml_backend_comm_allreduce_tensor");
            GGML_ASSERT(comm_allreduce != خالی_سوچک);
        }
    }

    ~ggml_backend_meta_context() {
        إذا (comm_ctx != خالی_سوچک) {
            ggml_backend_comm_free_t comm_free = (ggml_backend_comm_free_t) ggml_backend_reg_get_proc_پتہ(
                ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_configs[0].backend)), "ggml_backend_comm_free");
            GGML_ASSERT(comm_free != خالی_سوچک);
            comm_free(comm_ctx);
        }
        لكل (خودکار & bc : backend_configs) {
            ggml_backend_free(bc.backend);
        }
    }
};

ساكن ثابت حرف * ggml_backend_meta_get_name(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    ثابت ggml_backend_meta_context * backend_ctx = (ثابت ggml_backend_meta_context *) backend->context;
    أرجع backend_ctx->name.c_str();
}

ساكن فراغ ggml_backend_meta_free(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;
    مٹاؤ backend_ctx;
    مٹاؤ backend;
}

ساكن فراغ ggml_backend_meta_سیٹ_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, ثابت فراغ * انکڑے, size_t وتھ, size_t size) {
    ثابت size_t n_backends = ggml_backend_meta_n_backends(backend);
    GGML_ASSERT(وتھ == 0);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    ثابت ggml_backend_meta_ونڈو_state ونڈو_state = ggml_backend_meta_get_ونڈو_state(tensor, /*assume_sync =*/ خطأ);
    GGML_ASSERT(ونڈو_state.n_segments == 1);
    GGML_ASSERT(ونڈو_state.nr[0]      == 1);

    بدل (ونڈو_state.axis) {
        حالة GGML_BACKEND_ونڈو_AXIS_0:
        حالة GGML_BACKEND_ونڈو_AXIS_1:
        حالة GGML_BACKEND_ونڈو_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            ثابت size_t chunk_ناپ_بھریا = tensor->nb[ونڈو_state.axis + 1];
            GGML_ASSERT(وتھ % chunk_ناپ_بھریا == 0);
            GGML_ASSERT(size   % chunk_ناپ_بھریا == 0);
            ثابت int64_t i_شروع =  وتھ        /chunk_ناپ_بھریا;
            ثابت int64_t i_stop  = (وتھ + size)/chunk_ناپ_بھریا;
            size_t وتھ_j = 0;
            لكل (size_t j = 0; j < n_backends; j++){
                ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
                ggml_tensor * simple_tensor = ggml_backend_meta_بفر_simple_tensor(tensor, j);
                ثابت size_t chunk_ناپ_j = simple_tensor->nb[ونڈو_state.axis + 1];
                إذا (chunk_ناپ_j == 0) {
                    استمر;
                }
                ggml_backend_tensor_سیٹ_2d_async(simple_backend, simple_tensor, (ثابت حرف *) انکڑے + وتھ_j, وتھ, chunk_ناپ_j,
                    i_stop - i_شروع, chunk_ناپ_j, chunk_ناپ_بھریا);
                وتھ_j += chunk_ناپ_j;
            }
            GGML_ASSERT(وتھ_j == chunk_ناپ_بھریا);
        } اكسر;
        حالة GGML_BACKEND_ونڈو_AXIS_MIRRORED: {
            لكل (size_t j = 0; j < n_backends; j++) {
                ggml_backend_tensor_سیٹ_async(
                    ggml_backend_meta_simple_backend(backend, j), ggml_backend_meta_بفر_simple_tensor(tensor, j), انکڑے, وتھ, size);
            }
        } اكسر;
        افتراضي: {
            GGML_ABORT("fatal error");
        }
    }
}

ساكن فراغ ggml_backend_meta_get_tensor_async(ggml_backend_t backend, ثابت ggml_tensor * tensor, فراغ * انکڑے, size_t وتھ, size_t size) {
    ثابت size_t n_backends = ggml_backend_meta_n_backends(backend);
    GGML_ASSERT(وتھ == 0);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    ثابت ggml_backend_meta_ونڈو_state ونڈو_state = ggml_backend_meta_get_ونڈو_state(tensor, /*assume_sync =*/ خطأ);
    GGML_ASSERT(ونڈو_state.n_segments == 1);
    GGML_ASSERT(ونڈو_state.nr[0]      == 1);

    بدل (ونڈو_state.axis) {
        حالة GGML_BACKEND_ونڈو_AXIS_0:
        حالة GGML_BACKEND_ونڈو_AXIS_1:
        حالة GGML_BACKEND_ونڈو_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            ثابت size_t chunk_ناپ_بھریا = tensor->nb[ونڈو_state.axis + 1];
            GGML_ASSERT(وتھ % chunk_ناپ_بھریا == 0);
            GGML_ASSERT(size   % chunk_ناپ_بھریا == 0);
            ثابت int64_t i_شروع =  وتھ        /chunk_ناپ_بھریا;
            ثابت int64_t i_stop  = (وتھ + size)/chunk_ناپ_بھریا;
            size_t وتھ_j = 0;
            لكل (size_t j = 0; j < n_backends; j++){
                ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
                ثابت ggml_tensor * simple_tensor = ggml_backend_meta_بفر_simple_tensor(tensor, j);
                ثابت size_t chunk_ناپ_j = simple_tensor->nb[ونڈو_state.axis + 1];
                إذا (chunk_ناپ_j == 0) {
                    استمر;
                }
                ggml_backend_tensor_get_2d_async(simple_backend, simple_tensor, (حرف *) انکڑے + وتھ_j, وتھ, chunk_ناپ_j,
                    i_stop - i_شروع, chunk_ناپ_j, chunk_ناپ_بھریا);
                وتھ_j += chunk_ناپ_j;
            }
            GGML_ASSERT(وتھ_j == chunk_ناپ_بھریا);
        } اكسر;
        حالة GGML_BACKEND_ونڈو_AXIS_MIRRORED: {
            // TODO other simple backend may be better
            ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, 0);
            ثابت ggml_tensor * simple_tensor = ggml_backend_meta_بفر_simple_tensor(tensor, 0);
            ggml_backend_tensor_get_async(simple_backend, simple_tensor, انکڑے, وتھ, size);
        } اكسر;
        افتراضي: {
            GGML_ABORT("fatal error");
        }
    }
}

ساكن فراغ ggml_backend_meta_synchronize(ggml_backend_t backend) {
    ثابت size_t n_backends = ggml_backend_meta_n_backends(backend);
    لكل (size_t i = 0; i < n_backends; i++) {
        ggml_backend_synchronize(ggml_backend_meta_simple_backend(backend, i));
    }
}

ساكن تعداد ggml_حالت ggml_backend_meta_گراف_compute(ggml_backend_t backend, بنية ggml_cgraph * cgraph) {
    GGML_ASSERT(cgraph->grads == خالی_سوچک);
    ثابت size_t n_backends = ggml_backend_meta_n_backends(backend);
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;

    // If the previous cgraph had a defined UID it can be used to skip rebuilding the subgraphs per simple backend.
    ثابت منطقي needs_rebuild = (cgraph->uid == 0) || (cgraph->uid != backend_ctx->uid);

    منطقي max_nnodes_raised = خطأ;
    إذا (cgraph->n_nodes > backend_ctx->max_nnodes) {
        لكل (size_t j = 0; j < n_backends; j++) {
            خودکار & bcj = backend_ctx->backend_configs[j];
            bcj.nodes.resize(cgraph->n_nodes);
            bcj.cgraphs.resize(cgraph->n_nodes);
        }
        backend_ctx->max_nnodes = cgraph->n_nodes;
        max_nnodes_raised = سچ;
        أكد(needs_rebuild);
    }

    إذا (needs_rebuild) {
        std::set<ggml_backend_بفر_t> ورتیا_buffers;
        لكل (صحيح i = 0; i < cgraph->n_leafs; i++) {
            إذا (ggml_backend_بفر_is_meta(cgraph->leafs[i]->بفر)) {
                ورتیا_buffers.emplace(cgraph->leafs[i]->بفر);
            }
        }
        لكل (صحيح i = 0; i < cgraph->n_nodes; i++) {
            إذا (ggml_backend_بفر_is_meta(cgraph->nodes[i]->بفر)) {
                ورتیا_buffers.emplace(cgraph->nodes[i]->بفر);
            }
        }
        لكل (ggml_backend_بفر_t buf : ورتیا_buffers) {
            ggml_backend_meta_بفر_context * buf_ctx = (ggml_backend_meta_بفر_context *) buf->context;
            buf_ctx->stc_compute_اشاریہ_اگلا = buf_ctx->stc_compute_اشاریہ ^ 1;
            ggml_backend_meta_simple_tensor_container & stc = buf_ctx->stc_compute[buf_ctx->stc_compute_اشاریہ_اگلا];
            لكل (ggml_context_ptr & ctx : stc.ctxs) {
                ggml_reset(ctx.get());
            }
            stc.simple_tensors.clear();
        }
        size_t n_subgraphs  = 0;
        size_t max_tmp_ناپ = 0;

        لكل (size_t j = 0; j < n_backends; j++) {
            خودکار & bcj = backend_ctx->backend_configs[j];

            لكل (صحيح i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * گرہ = cgraph->nodes[i];
                إذا (!ggml_backend_بفر_is_meta(گرہ->بفر)) {
                    // [TAG_META_HOST_VIEWS]
                    GGML_ASSERT(ggml_backend_meta_is_host_view(گرہ));

                    // keep the node as is, mapping it to a simple tensor is not possible
                    bcj.nodes[i] = گرہ;
                    استمر;
                }
                bcj.nodes[i] = ggml_backend_meta_بفر_simple_tensor(گرہ, j);
                GGML_ASSERT(bcj.nodes[i]);
            }
        }

        {
            // For MoE models it may make sense to delay the AllReduce in order to reduce I/O:
            خودکار get_i_delayed_branch = [&](ثابت صحيح i) -> صحيح {
                صحيح id = i; // i_delayed
                صحيح idr = i; // i_delayed return, last safe return value

                ggml_tensor * گرہ = cgraph->nodes[id];
                int32_t n_ورتیا = ggml_گرہ_get_use_گنتی(cgraph, id);

                // Skip MIRRORED nodes that don't consume node
                خودکار skip_unrelated = [&]() {
                    طالما (id + 1 < cgraph->n_nodes) {
                        ggml_tensor * اگلا = cgraph->nodes[id+1];
                        إذا (ggml_backend_meta_get_ونڈو_state(اگلا, خطأ).axis != GGML_BACKEND_ونڈو_AXIS_MIRRORED) {
                            اكسر;
                        }
                        منطقي محفوظ = سچ;
                        لكل (صحيح s = 0; s < GGML_MAX_SRC; s++) {
                            إذا (اگلا->src[s] == خالی_سوچک) {
                                استمر;
                            }
                            إذا (اگلا->src[s] == گرہ) {
                                محفوظ = خطأ;
                                اكسر;
                            }
                            إذا (ggml_backend_meta_get_ونڈو_state(اگلا->src[s], خطأ).axis != GGML_BACKEND_ونڈو_AXIS_MIRRORED) {
                                محفوظ = خطأ;
                                اكسر;
                            }
                        }
                        إذا (!محفوظ) {
                            اكسر;
                        }
                        id++;
                    }
                };

                skip_unrelated();
                إذا (id + 1 >= cgraph->n_nodes) {
                    أرجع idr;
                }
                {
                    ggml_tensor * اگلا = cgraph->nodes[id+1];
                    إذا (اگلا->op == GGML_OP_جوڑو_ID && اگلا->src[0] == گرہ &&
                            ggml_backend_meta_get_ونڈو_state(اگلا->src[1], خطأ).axis == GGML_BACKEND_ونڈو_AXIS_PARTIAL &&
                            ggml_backend_meta_get_ونڈو_state(اگلا->src[2], خطأ).axis == GGML_BACKEND_ونڈو_AXIS_MIRRORED) {
                        گرہ = اگلا;
                        id++;
                        idr = id;
                        n_ورتیا = ggml_گرہ_get_use_گنتی(cgraph, id);
                    }
                }
                // Chain of MULs with MIRRORED src[1]
                طالما (سچ) {
                    skip_unrelated();
                    إذا (id + 1 >= cgraph->n_nodes) {
                        أرجع idr;
                    }
                    ggml_tensor * اگلا = cgraph->nodes[id+1];
                    إذا (اگلا->op == GGML_OP_MUL && اگلا->src[0] == گرہ &&
                            ggml_backend_meta_get_ونڈو_state(اگلا->src[1], خطأ).axis == GGML_BACKEND_ونڈو_AXIS_MIRRORED) {
                        گرہ = اگلا;
                        id++;
                        idr = id;
                        n_ورتیا = ggml_گرہ_get_use_گنتی(cgraph, id);
                    } وإلا {
                        اكسر;
                    }
                }

                إذا (n_ورتیا != گرہ->ne[1] || id + 2*n_ورتیا-1 >= cgraph->n_nodes) {
                    أرجع idr;
                }
                لكل (int32_t k = 0; k < n_ورتیا; k++) {
                    ggml_tensor * اگلا = cgraph->nodes[id+1];
                    إذا (اگلا->op != GGML_OP_VIEW || اگلا->view_src != گرہ || اگلا->view_offs != k*گرہ->nb[1] ||
                            اگلا->ne[0] != گرہ->ne[0] || اگلا->ne[1] != گرہ->ne[2] || اگلا->nb[1] != گرہ->nb[2] ||
                            ggml_گرہ_get_use_گنتی(cgraph, id+1) != 1) {
                        أرجع idr;
                    }
                    id++;
                }
                {
                    ggml_tensor * اگلا = cgraph->nodes[id+1];
                    إذا (اگلا->op != GGML_OP_جوڑو || اگلا->src[0] != cgraph->nodes[id - (n_ورتیا-1)] ||
                            اگلا->src[1] != cgraph->nodes[id - (n_ورتیا-2)] || ggml_گرہ_get_use_گنتی(cgraph, id+1) != 1) {
                        أرجع idr;
                    }
                    id++;
                }
                لكل (int32_t k = 0; k < n_ورتیا - 2; k++) {
                    ggml_tensor * اگلا = cgraph->nodes[id+1];
                    إذا (اگلا->op != GGML_OP_جوڑو || اگلا->src[0] != cgraph->nodes[id] ||
                            اگلا->src[1] != cgraph->nodes[id - (n_ورتیا-2)] || ggml_گرہ_get_use_گنتی(cgraph, id+1) != 1) {
                        أرجع idr;
                    }
                    id++;
                }
                idr = id;
                أرجع idr;
            };

            // AllReduce(a) + AllReduce(b) == AllReduce(a + b) for independent partial branches.
            خودکار get_i_delayed = [&](ثابت صحيح i) -> صحيح {
                ثابت صحيح i_delayed = get_i_delayed_branch(i);
                ggml_tensor * گرہ = cgraph->nodes[i_delayed];

                إذا (ggml_گرہ_get_use_گنتی(cgraph, i_delayed) != 1) {
                    أرجع i_delayed;
                }

                لكل (صحيح id = i_delayed + 1; id < cgraph->n_nodes; id++) {
                    ggml_tensor * اگلا = cgraph->nodes[id];
                    إذا (اگلا->view_src == گرہ) {
                        أرجع i_delayed;
                    }
                    لكل (صحيح s = 0; s < GGML_MAX_SRC; s++) {
                        إذا (اگلا->src[s] == گرہ) {
                            أرجع i_delayed;
                        }
                    }

                    إذا (اگلا->view_src != خالی_سوچک && اگلا->view_src->op == GGML_OP_NONE && ggml_backend_بفر_is_host(اگلا->view_src->بفر)) {
                        استمر;
                    }
                    إذا (ggml_backend_meta_get_ونڈو_state(اگلا, خطأ).axis != GGML_BACKEND_ونڈو_AXIS_PARTIAL) {
                        استمر;
                    }

                    ثابت صحيح i_other = id;
                    ثابت صحيح i_other_delayed = get_i_delayed_branch(i_other);
                    ggml_tensor * other = cgraph->nodes[i_other_delayed];
                    إذا (ggml_گرہ_get_use_گنتی(cgraph, i_other_delayed) != 1 || i_other_delayed + 1 >= cgraph->n_nodes) {
                        أرجع i_delayed;
                    }

                    ggml_tensor * جوڑ = cgraph->nodes[i_other_delayed + 1];
                    إذا (جوڑ->op != GGML_OP_جوڑو ||
                            !ggml_are_same_shape(گرہ, other) || گرہ->type != other->type || جوڑ->type != گرہ->type ||
                            !((جوڑ->src[0] == گرہ && جوڑ->src[1] == other) ||
                              (جوڑ->src[0] == other && جوڑ->src[1] == گرہ)) ||
                            ggml_backend_meta_get_ونڈو_state(جوڑ, خطأ).axis != GGML_BACKEND_ونڈو_AXIS_MIRRORED) {
                        أرجع i_delayed;
                    }

                    لكل (size_t j = 0; j < n_backends; j++) {
                        خودکار & bcj = backend_ctx->backend_configs[j];
                        ثابت منطقي compute       = bcj.nodes[i]->flags       & GGML_TENSOR_FLAG_COMPUTE;
                        ثابت منطقي compute_other = bcj.nodes[i_other]->flags & GGML_TENSOR_FLAG_COMPUTE;
                        إذا (compute != compute_other) {
                            أرجع i_delayed;
                        }
                    }
                    أرجع i_other_delayed + 1;
                }
                أرجع i_delayed;
            };

            صحيح i_شروع = 0;
            لكل (صحيح i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * گرہ = cgraph->nodes[i];
                إذا (گرہ->view_src != خالی_سوچک && گرہ->view_src->op == GGML_OP_NONE && ggml_backend_بفر_is_host(گرہ->view_src->بفر)) {
                    استمر;
                }
                ثابت ggml_backend_meta_ونڈو_state ونڈو_state = ggml_backend_meta_get_ونڈو_state(گرہ, /*assume_sync =*/ خطأ);
                إذا (ونڈو_state.axis == GGML_BACKEND_ونڈو_AXIS_PARTIAL) {
                    max_tmp_ناپ = std::max(max_tmp_ناپ, ggml_nbytes(گرہ));
                }
                ثابت منطقي new_subgraph = i + 1 == cgraph->n_nodes || ونڈو_state.axis == GGML_BACKEND_ونڈو_AXIS_PARTIAL;
                إذا (!new_subgraph) {
                    استمر;
                }

                ثابت صحيح i_delayed = get_i_delayed(i);

                // If we can delay the AllReduce we need to consider the interaction with zero-sized tensor slices.
                // A backend with such a slice would normally have valid data after participating in the AllReduce with a node that has
                //     its compute flag disabled and thus gets its data zeroed out.
                // If the AllReduce is delayed then the nodes until that point also need to have their compute flag disabled.
                إذا (i_delayed > i) {
                    لكل (size_t j = 0; j < n_backends; j++) {
                        خودکار & bcj = backend_ctx->backend_configs[j];
                        إذا ((bcj.nodes[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                            لكل (صحيح ii = i + 1; ii <= i_delayed; ii++) {
                                bcj.nodes[ii]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
                            }
                        }
                    }
                }

                i = i_delayed;

                لكل (size_t j = 0; j < n_backends; j++) {
                    خودکار & bcj = backend_ctx->backend_configs[j];
                    bcj.cgraphs[n_subgraphs].وتھ = i_شروع;
                }
                n_subgraphs++;
                i_شروع = i + 1;
            }
            GGML_ASSERT(i_شروع == cgraph->n_nodes);
        }

        backend_ctx->uid         = cgraph->uid;
        backend_ctx->n_subgraphs = n_subgraphs;

        إذا (max_tmp_ناپ > backend_ctx->max_tmp_ناپ) {
            لكل (size_t j = 0; j < n_backends; j++) {
                خودکار & bcj = backend_ctx->backend_configs[j];
                لكل (size_t i = 0; i < backend_ctx->n_reduce_قدماں; i++) {
                    bcj.bufs[i].reset(ggml_backend_alloc_بفر(bcj.backend, max_tmp_ناپ));
                }
            }
            backend_ctx->max_tmp_ناپ = max_tmp_ناپ;
        }

        إذا (max_nnodes_raised || n_subgraphs > backend_ctx->max_subgraphs) {
            backend_ctx->max_subgraphs = std::max(backend_ctx->max_subgraphs, n_subgraphs);
            ثابت size_t n_nodes_فی_device = 3 * backend_ctx->n_reduce_قدماں; // tmp + ADD (+zeroing) graph per step and device
            ثابت size_t n_cgraphs_فی_device = 2 * backend_ctx->n_reduce_قدماں; // ADD ( + zeroing) graph per step and device
            ثابت size_t mem_فی_device_graphs_main = backend_ctx->max_subgraphs*ggml_گراف_overhead_custom(backend_ctx->max_nnodes, cgraph->grads);
            ثابت size_t mem_فی_device_graphs_aux = n_cgraphs_فی_device*backend_ctx->max_subgraphs*ggml_گراف_overhead_custom(1, cgraph->grads);
            ثابت size_t mem_فی_device_nodes_aux = n_nodes_فی_device*backend_ctx->max_subgraphs*ggml_tensor_overhead();
            ثابت ggml_init_params params = {
                /*.mem_size   =*/ n_backends * (mem_فی_device_graphs_main + mem_فی_device_graphs_aux + mem_فی_device_nodes_aux),
                /*.mem_buffer =*/ خالی_سوچک,
                /*.no_alloc   =*/ سچ,
            };
            backend_ctx->ctx.reset(ggml_init(params));
            لكل (size_t j = 0; j < n_backends; j++) {
                خودکار & bcj = backend_ctx->backend_configs[j];
                لكل (size_t i = 0; i < backend_ctx->max_subgraphs; i++) {
                    bcj.cgraphs[i].cgraph_main = ggml_new_گراف_custom(backend_ctx->ctx.get(), backend_ctx->max_nnodes, /*grads =*/ خطأ);
                }
            }
            backend_ctx->cgraphs_aux.resize(n_backends*n_cgraphs_فی_device*backend_ctx->max_subgraphs);
            لكل (size_t k = 0; k < backend_ctx->cgraphs_aux.size(); k++) {
                backend_ctx->cgraphs_aux[k] = ggml_new_گراف_custom(backend_ctx->ctx.get(), 1, cgraph->grads);
            }
            backend_ctx->nodes_aux.resize(n_backends*n_nodes_فی_device*backend_ctx->max_subgraphs);
            لكل (size_t k = 0; k < backend_ctx->nodes_aux.size(); k++) {
                backend_ctx->nodes_aux[k] = ggml_new_tensor_1d(backend_ctx->ctx.get(), GGML_TYPE_F32, 1);
            }
        }

        لكل (size_t j = 0; j < n_backends; j++) {
            خودکار & bcj = backend_ctx->backend_configs[j];
            لكل (size_t i_گراف = 0; i_گراف < n_subgraphs; i_گراف++) {
                ggml_cgraph * cgraph_ij = bcj.cgraphs[i_گراف].cgraph_main;
                ثابت size_t i_گرہ_شروع = bcj.cgraphs[i_گراف].وتھ;
                ثابت size_t i_گرہ_stop = i_گراف + 1 < n_subgraphs ? bcj.cgraphs[i_گراف + 1].وتھ : cgraph->n_nodes;
                cgraph_ij->n_nodes = i_گرہ_stop - i_گرہ_شروع;
                ggml_hash_سیٹ_reset(&cgraph_ij->ویکھیا_hash_سیٹ);
                لكل (size_t i_گرہ = i_گرہ_شروع; i_گرہ < i_گرہ_stop; i_گرہ++) {
                    ggml_tensor * گرہ_ij = bcj.nodes[i_گرہ];
                    cgraph_ij->nodes[i_گرہ - i_گرہ_شروع] = گرہ_ij;
                    ثابت size_t hash_pos_orig = ggml_hash_لبھو(&cgraph->ویکھیا_hash_سیٹ, cgraph->nodes[i_گرہ]);
                    ثابت size_t hash_pos_ij = ggml_hash_داخل_کرو(&cgraph_ij->ویکھیا_hash_سیٹ, گرہ_ij);
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
        لكل (size_t k = 0; k < GGML_MAX_DIMS; k++) {
            ret->ne[k] = t->ne[k];
            ret->nb[k] = t->nb[k];
        }
        أرجع ret;
    };
    خودکار سیٹ_tmp_انکڑے = [&](ggml_tensor * tensor, ثابت size_t j, ثابت size_t i_buf) {
        خودکار & bcj = backend_ctx->backend_configs[j];
        ggml_backend_بفر_ptr & buf_ptr = bcj.bufs[i_buf];
        إذا (!buf_ptr || ggml_backend_بفر_get_ناپ(buf_ptr.get()) < backend_ctx->max_tmp_ناپ) {
            buf_ptr.reset(ggml_backend_alloc_بفر(bcj.backend, backend_ctx->max_tmp_ناپ));
        }
        tensor->بفر = buf_ptr.get();
        tensor->انکڑے   = ggml_backend_بفر_get_ادھار(buf_ptr.get());
    };
    // FIXME usage_counts
    خودکار get_cgraph_aux = [&]() -> ggml_cgraph * {
        ggml_cgraph * ret = backend_ctx->cgraphs_aux[iga++];
        أرجع ret;
    };

    // Preferentially use backend-specific allreduce_tensor_async (e.g. NCCL for CUDA), use a generic fallback if unavailable:
    خودکار allreduce_fallback = [&](size_t i) -> ggml_حالت {
        std::vector<ggml_cgraph *> قدم_cgraphs(n_backends, خالی_سوچک);

        // Zero out nodes that were disabled due to having a zero-sized slice:
        لكل (size_t j = 0; j < n_backends; j++) {
            خودکار & bcj = backend_ctx->backend_configs[j];
            ggml_tensor * گرہ = bcj.cgraphs[i].cgraph_main->nodes[bcj.cgraphs[i].cgraph_main->n_nodes - 1];
            إذا (گرہ->flags & GGML_TENSOR_FLAG_COMPUTE) {
                استمر;
            }
            إذا (ggml_nelements(گرہ) > 0) {
                ggml_tensor * گرہ_zero = get_گرہ_aux(گرہ);
                گرہ_zero->op = GGML_OP_FILL;
                گرہ_zero->src[0] = گرہ; // only used for the shape, the data is not read
                ggml_سیٹ_op_params_f32(گرہ_zero, 0, 0.0f);
                گرہ_zero->انکڑے = گرہ->انکڑے;
                گرہ_zero->بفر = گرہ->بفر;
                گرہ_zero->flags |= GGML_TENSOR_FLAG_COMPUTE;

                قدم_cgraphs[j] = get_cgraph_aux();
                قدم_cgraphs[j]->nodes[0] = گرہ_zero;
                قدم_cgraphs[j]->n_nodes = 1;
                ثابت ggml_حالت حالت = ggml_backend_گراف_compute_async(bcj.backend, قدم_cgraphs[j]);
                إذا (حالت != GGML_حالت_SUCCESS) {
                    أرجع حالت;
                }
            }
        }
        std::fill(قدم_cgraphs.begin(), قدم_cgraphs.end(), خالی_سوچک);

        خودکار ڈالو_انکڑے = [&](ثابت size_t j_src, ثابت size_t j_dst, ثابت size_t i_buf) {
            أكد(قدم_cgraphs[j_dst] == خالی_سوچک);
            خودکار & bcj_src = backend_ctx->backend_configs[j_src];
            خودکار & bcj_dst = backend_ctx->backend_configs[j_dst];

            ggml_tensor * گرہ_src = bcj_src.cgraphs[i].cgraph_main->nodes[bcj_src.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_tensor * گرہ_dst = bcj_dst.cgraphs[i].cgraph_main->nodes[bcj_dst.cgraphs[i].cgraph_main->n_nodes - 1];
            GGML_ASSERT(ggml_is_contiguous(گرہ_src));
            GGML_ASSERT(ggml_is_contiguous(گرہ_dst));

            ggml_tensor * گرہ_tmp = get_گرہ_aux(گرہ_dst);
            سیٹ_tmp_انکڑے(گرہ_tmp, j_dst, i_buf);

            ggml_backend_tensor_copy_async(bcj_src.backend, bcj_dst.backend, گرہ_src, گرہ_tmp);

            ggml_tensor * گرہ_red = get_گرہ_aux(گرہ_dst);
            گرہ_red->view_src = گرہ_dst->view_src == خالی_سوچک ? گرہ_dst : گرہ_dst->view_src;
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

        size_t وتھ_j = n_backends/2;
        طالما ((وتھ_j & (وتھ_j - 1)) != 0) {
            وتھ_j--;
        }
        ثابت size_t وتھ_j_max = وتھ_j;
        size_t i_buf = 0;

        // If n_backends is not a power of 2, fold in the excess prior to butterfly reduction:
        لكل (size_t j_src = 2*وتھ_j_max; j_src < n_backends; j_src++) {
            ثابت size_t j_dst = j_src - 2*وتھ_j_max;
            ڈالو_انکڑے(j_src, j_dst, i_buf);
            ثابت ggml_حالت حالت = ggml_backend_گراف_compute_async(backend_ctx->backend_configs[j_dst].backend, قدم_cgraphs[j_dst]);
            إذا (حالت != GGML_حالت_SUCCESS) {
                أرجع حالت;
            }
            i_buf = 1;
        }

        // Butterfly reduction:
        لكل (; وتھ_j >= 1; وتھ_j /= 2) {
            std::fill(قدم_cgraphs.begin(), قدم_cgraphs.end(), خالی_سوچک);

            لكل (size_t j = 0; j < 2*وتھ_j_max; j++) {
                ثابت size_t j_other = j ^ وتھ_j;
                إذا (j_other >= n_backends) {
                    استمر;
                }
                ڈالو_انکڑے(j, j_other, i_buf);
            }

            لكل (size_t j = 0; j < 2*وتھ_j_max; j++) {
                إذا (قدم_cgraphs[j] == خالی_سوچک) {
                    استمر;
                }
                خودکار & bcj = backend_ctx->backend_configs[j];
                ثابت ggml_حالت حالت = ggml_backend_گراف_compute_async(bcj.backend, قدم_cgraphs[j]);
                إذا (حالت != GGML_حالت_SUCCESS) {
                    أرجع حالت;
                }
            }
            i_buf++;
        }
        أكد(i_buf == backend_ctx->n_reduce_قدماں);

        // If n_backends is not a power of 2, copy back the reduced tensors to the excess:
        لكل (size_t j = 2*وتھ_j_max; j < n_backends; j++) {
            خودکار & bcj_src = backend_ctx->backend_configs[j - 2*وتھ_j_max];
            خودکار & bcj_dst = backend_ctx->backend_configs[j];

            ggml_tensor * گرہ_src = bcj_src.cgraphs[i].cgraph_main->nodes[bcj_src.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_tensor * گرہ_dst = bcj_dst.cgraphs[i].cgraph_main->nodes[bcj_dst.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_backend_tensor_copy_async(bcj_src.backend, bcj_dst.backend, گرہ_src, گرہ_dst);
        }

        أرجع GGML_حالت_SUCCESS;
    };


    لكل (size_t i = 0; i < backend_ctx->n_subgraphs; i++) {
        لكل (size_t j = 0; j < n_backends; j++) {
            خودکار & bcj = backend_ctx->backend_configs[j];
            ثابت ggml_حالت حالت = ggml_backend_گراف_compute_async(bcj.backend, bcj.cgraphs[i].cgraph_main);
            إذا (حالت != GGML_حالت_SUCCESS) {
                أرجع حالت;
            }
        }

        إذا (n_backends > 1 && i < backend_ctx->n_subgraphs - 1) {
            منطقي backend_allreduce_success = خطأ;
            إذا (backend_ctx->comm_ctx) {
                std::vector<ggml_tensor *> nodes;
                nodes.reserve(n_backends);
                لكل (size_t j = 0; j < n_backends; j++) {
                    خودکار & bcj = backend_ctx->backend_configs[j];
                    ggml_cgraph * cgraph_ij = bcj.cgraphs[i].cgraph_main;
                    nodes.push_back(cgraph_ij->nodes[cgraph_ij->n_nodes-1]);
                }
                backend_allreduce_success = backend_ctx->comm_allreduce(backend_ctx->comm_ctx, nodes.انکڑے());
            }

            إذا (!backend_allreduce_success) {
                ثابت ggml_حالت حالت = allreduce_fallback(i);
                إذا (حالت != GGML_حالت_SUCCESS) {
                    أرجع حالت;
                }
            }
        }
    }
    أرجع GGML_حالت_SUCCESS;
}

ساكن ثابت ggml_backend_i ggml_backend_meta_i = {
    /* .get_name                = */ ggml_backend_meta_get_name,
    /* .free                    = */ ggml_backend_meta_free,
    /* .set_tensor_async        = */ ggml_backend_meta_سیٹ_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_meta_get_tensor_async,
    /* .set_tensor_2d_async     = */ خالی_سوچک,
    /* .get_tensor_2d_async     = */ خالی_سوچک,
    /* .cpy_tensor_async        = */ خالی_سوچک,
    /* .synchronize             = */ ggml_backend_meta_synchronize,
    /* .graph_plan_create       = */ خالی_سوچک,
    /* .graph_plan_free         = */ خالی_سوچک,
    /* .graph_plan_update       = */ خالی_سوچک,
    /* .graph_plan_compute      = */ خالی_سوچک,
    /* .graph_compute           = */ ggml_backend_meta_گراف_compute,
    /* .event_record            = */ خالی_سوچک,
    /* .event_wait              = */ خالی_سوچک,
    /* .graph_optimize          = */ خالی_سوچک,
};

منطقي ggml_backend_is_meta(ggml_backend_t backend) {
    أرجع backend != خالی_سوچک && backend->iface.get_name == ggml_backend_meta_i.get_name;
}

ساكن ggml_backend_t ggml_backend_meta_device_init_backend(ggml_backend_dev_t dev, ثابت حرف * params) {
    ggml_backend_meta_context * backend_ctx = جديد ggml_backend_meta_context(dev, params);

    ggml_backend_t backend = جديد بنية ggml_backend;
    backend->guid    = ggml_backend_meta_guid();
    backend->iface   = ggml_backend_meta_i;
    backend->device  = dev;
    backend->context = backend_ctx;
    أرجع backend;
}

size_t ggml_backend_meta_n_backends(ggml_backend_t meta_backend) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    ثابت ggml_backend_meta_context * backend_ctx = (ثابت ggml_backend_meta_context *) meta_backend->context;
    أرجع backend_ctx->backend_configs.size();
}

ggml_backend_t ggml_backend_meta_simple_backend(ggml_backend_t meta_backend, size_t اشاریہ) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    ثابت ggml_backend_meta_context * backend_ctx = (ثابت ggml_backend_meta_context *) meta_backend->context;
    أرجع backend_ctx->backend_configs[اشاریہ].backend;
}
