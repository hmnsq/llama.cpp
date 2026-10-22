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

ثابت حرف * ggml_backend_meta_وېشل_axis_name(تعداد ggml_backend_meta_وېشل_axis وېشل_axis) {
    بدل (وېشل_axis) {
        حالة GGML_BACKEND_وېشل_AXIS_0:
            أرجع "0";
        حالة GGML_BACKEND_وېشل_AXIS_1:
            أرجع "1";
        حالة GGML_BACKEND_وېشل_AXIS_2:
            أرجع "2";
        حالة GGML_BACKEND_وېشل_AXIS_3:
            أرجع "3";
        حالة GGML_BACKEND_وېشل_AXIS_MIRRORED:
            أرجع "MIRRORED";
        حالة GGML_BACKEND_وېشل_AXIS_PARTIAL:
            أرجع "PARTIAL";
        حالة GGML_BACKEND_وېشل_AXIS_NONE:
            أرجع "NONE";
        حالة GGML_BACKEND_وېشل_AXIS_UNKNOWN:
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
    ggml_backend_meta_get_وېشل_state_t get_وېشل_state;
    فراغ *                              get_وېشل_state_ud;

    std::string name;
    std::string description;

    ggml_backend_meta_device_context(
            std::vector<ggml_backend_dev_t> simple_devs, ggml_backend_meta_get_وېشل_state_t get_وېشل_state, فراغ * get_وېشل_state_ud) :
            simple_devs(std::move(simple_devs)), get_وېشل_state(get_وېشل_state), get_وېشل_state_ud(get_وېشل_state_ud) {
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

    منطقي چلوونکی<(ثابت ggml_backend_meta_device_context & other) ثابت {
        أرجع std::tie(simple_devs, get_وېشل_state, get_وېشل_state_ud)
            < std::tie(other.simple_devs, other.get_وېشل_state, other.get_وېشل_state_ud);
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

ساكن فراغ ggml_backend_meta_device_get_حافظه(ggml_backend_dev_t dev, size_t * حرر, size_t * ټول) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ثابت ggml_backend_meta_device_context * meta_dev_ctx = (ثابت ggml_backend_meta_device_context *) dev->context;
    *حرر  = 0;
    *ټول = 0;
    لكل (ggml_backend_dev_t dev : meta_dev_ctx->simple_devs) {
        size_t tmp_free, tmp_ټول;
        ggml_backend_dev_حافظه(dev, &tmp_free, &tmp_ټول);
        *حرر  += tmp_free;
        *ټول += tmp_ټول;
    }
}

ساكن تعداد ggml_backend_dev_type ggml_backend_meta_device_get_type(ggml_backend_dev_t dev) {
    أرجع GGML_BACKEND_DEVICE_TYPE_META;

    GGML_ناکارول_شوی(dev);
}

ساكن فراغ ggml_backend_meta_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ثابت ggml_backend_meta_device_context * meta_dev_ctx = (ثابت ggml_backend_meta_device_context *) dev->context;

    // TODO replace placeholders
    props->name        = ggml_backend_meta_device_get_name(dev);
    props->description = ggml_backend_meta_device_get_description(dev);
    props->type        = ggml_backend_meta_device_get_type(dev);
    props->device_id   = 0;

    ggml_backend_meta_device_get_حافظه(dev, &props->حافظه_free, &props->حافظه_ټول);

    props->caps = {
        /* .async                 = */ ریښتیا,
        /* .host_buffer           = */ خطأ, // Not implemented.
        /* .buffer_from_host_ptr  = */ خطأ, // Not implemented.
        /* .events                = */ خطأ, // Not implemented.
        /* .mmap_support          = */ ریښتیا,
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
    أرجع ریښتیا;
}

ساكن ثابت ggml_backend_device_i ggml_backend_meta_device_iface = {
    /* .get_name             = */ ggml_backend_meta_device_get_name,
    /* .get_description      = */ ggml_backend_meta_device_get_description,
    /* .get_memory           = */ ggml_backend_meta_device_get_حافظه,
    /* .get_type             = */ ggml_backend_meta_device_get_type,
    /* .get_props            = */ ggml_backend_meta_device_get_props,
    /* .init_backend         = */ ggml_backend_meta_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_meta_device_get_بفر_type,
    /* .get_host_buffer_type = */ ggml_backend_meta_device_get_host_بفر_type,
    /* .buffer_from_host_ptr = */ تش_اشاره,
    /* .supports_op          = */ ggml_backend_meta_device_supports_op,
    /* .supports_buft        = */ ggml_backend_meta_device_supports_buft,
    /* .offload_op           = */ تش_اشاره,
    /* .event_new            = */ تش_اشاره,
    /* .event_free           = */ تش_اشاره,
    /* .event_synchronize    = */ تش_اشاره,
};

ساكن منطقي ggml_backend_dev_is_meta(ggml_backend_dev_t dev) {
    أرجع dev != تش_اشاره && dev->iface.get_name == ggml_backend_meta_device_iface.get_name;
}

ساكن size_t ggml_backend_meta_dev_n_devs(ggml_backend_dev_t meta_dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(meta_dev));
    ثابت ggml_backend_meta_device_context * meta_dev_ctx = (ثابت ggml_backend_meta_device_context *) meta_dev->context;
    أرجع meta_dev_ctx->simple_devs.size();
}

ساكن ggml_backend_dev_t ggml_backend_meta_dev_simple_dev(ggml_backend_dev_t meta_dev, size_t شاخص) {
    GGML_ASSERT(ggml_backend_dev_is_meta(meta_dev));
    ثابت ggml_backend_meta_device_context * meta_dev_ctx = (ثابت ggml_backend_meta_device_context *) meta_dev->context;
    GGML_ASSERT(شاخص < meta_dev_ctx->simple_devs.size());
    أرجع meta_dev_ctx->simple_devs[شاخص];
}

ggml_backend_dev_t ggml_backend_meta_device(
        ggml_backend_dev_t * devs, size_t n_devs, ggml_backend_meta_get_وېشل_state_t get_وېشل_state, فراغ * get_وېشل_state_ud) {
    GGML_ASSERT(n_devs <= GGML_BACKEND_META_MAX_DEVICES);
    // TODO: this is not thread-safe - needs to be fixed
    ساكن std::vector<std::unique_ptr<ggml_backend_meta_device_context>>         ctxs;
    ساكن std::map<ggml_backend_meta_device_context, بنية ggml_backend_device> meta_devs;

    std::vector<ggml_backend_dev_t> simple_devs;
    simple_devs.reserve(n_devs);
    لكل (size_t i = 0; i < n_devs; i++) {
        simple_devs.push_back(devs[i]);
    }
    ggml_backend_meta_device_context ctx(simple_devs, get_وېشل_state, get_وېشل_state_ud);

    {
        خپلکاری it = meta_devs.find(ctx);
        إذا (it != meta_devs.end()) {
            أرجع &it->second;
        }
    }
    ctxs.push_back(std::make_unique<ggml_backend_meta_device_context>(ctx));

    بنية ggml_backend_device meta_dev = {
        /*iface  =*/ ggml_backend_meta_device_iface,
        /*reg    =*/ تش_اشاره,
        /*ctx    =*/ ctxs.back().get(),
    };

    خپلکاری پایله = meta_devs.emplace(*ctxs.back(), meta_dev);
    أرجع &پایله.لومړی->second;
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

    منطقي چلوونکی<(ثابت ggml_backend_meta_بفر_type_context & other) ثابت {
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

ساكن ggml_backend_بفر_type_t ggml_backend_meta_buft_simple_buft(ggml_backend_بفر_type_t meta_buft, size_t شاخص) {
    GGML_ASSERT(ggml_backend_buft_is_meta(meta_buft));
    ثابت ggml_backend_meta_بفر_type_context * meta_buft_ctx = (ثابت ggml_backend_meta_بفر_type_context *) meta_buft->context;
    GGML_ASSERT(شاخص < meta_buft_ctx->simple_bufts.size());
    أرجع meta_buft_ctx->simple_bufts[شاخص];
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

ساكن size_t ggml_backend_meta_بفر_type_get_max_اندازه(ggml_backend_بفر_type_t buft) {
    ثابت size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_size = SIZE_MAX;
    لكل (size_t i = 0; i < n_simple_bufts; i++) {
        max_size = std::min(max_size, ggml_backend_buft_get_max_اندازه(ggml_backend_meta_buft_simple_buft(buft, i)));
    }
    أرجع max_size;
}

ساكن size_t ggml_backend_meta_بفر_type_get_alloc_اندازه(ggml_backend_بفر_type_t buft, ثابت ggml_tensor * tensor) {
    ثابت size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_alloc_اندازه = 0;
    لكل (size_t i = 0; i < n_simple_bufts; i++) {
        ثابت size_t alloc_اندازه = ggml_backend_buft_get_alloc_اندازه(ggml_backend_meta_buft_simple_buft(buft, i), tensor);
        max_alloc_اندازه = std::max(max_alloc_اندازه, alloc_اندازه);
    }
    أرجع max_alloc_اندازه;
}

ساكن منطقي ggml_backend_meta_بفر_type_is_host(ggml_backend_بفر_type_t buft) {
    ثابت size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    لكل (size_t i = 0; i < n_simple_bufts; i++) {
        إذا (!ggml_backend_buft_is_host(ggml_backend_meta_buft_simple_buft(buft, i))) {
            أرجع خطأ;
        }
    }
    أرجع ریښتیا;
}

ساكن ثابت بنية ggml_backend_بفر_type_i ggml_backend_meta_بفر_type_iface = {
    /* .get_name            = */ ggml_backend_meta_بفر_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_meta_بفر_type_alloc_بفر,
    /* .alloc_buffer_n      = */ ggml_backend_meta_بفر_type_alloc_بفر_n,
    /* .get_alignment       = */ ggml_backend_meta_بفر_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_meta_بفر_type_get_max_اندازه,
    /* .get_alloc_size      = */ ggml_backend_meta_بفر_type_get_alloc_اندازه,
    /* .get_alloc_size_n    = */ عدم,
    /* .is_host             = */ ggml_backend_meta_بفر_type_is_host,
};

منطقي ggml_backend_buft_is_meta(ggml_backend_بفر_type_t buft) {
    أرجع buft != تش_اشاره && buft->iface.get_name == ggml_backend_meta_بفر_type_iface.get_name;
}

ساكن ggml_backend_بفر_type_t ggml_backend_meta_device_get_بفر_type(ggml_backend_dev_t dev) {
    ساكن std::map<ggml_backend_dev_t, بنية ggml_backend_بفر_type> meta_bufts;
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    {
        خپلکاری it = meta_bufts.find(dev);
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
    خپلکاری پایله = meta_bufts.emplace(dev, meta_buft);
    أرجع &پایله.لومړی->second;
}

ساكن ggml_backend_بفر_type_t ggml_backend_meta_device_get_host_بفر_type(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ثابت ggml_backend_meta_device_context * meta_dev_ctx = (ثابت ggml_backend_meta_device_context *) dev->context;

    ggml_backend_بفر_type_t host_buft = تش_اشاره;
    لكل (ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
        ggml_backend_بفر_type_t simple_host_buft = ggml_backend_dev_host_بفر_type(simple_dev);
        إذا (simple_host_buft == تش_اشاره) {
            أرجع تش_اشاره;
        }
        إذا (host_buft == تش_اشاره) {
            host_buft = simple_host_buft;
        } وإلا إذا (host_buft != simple_host_buft) {
            // if different simple devices have different host buffer types,
            // we cannot provide a single host buffer type for the meta device
            أرجع تش_اشاره;
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
    صحيح stc_compute_شاخص      = 0;
    صحيح stc_compute_شاخص_بل = 0;
    std::vector<ggml_backend_بفر_ptr> bufs;

    // FIXME
    // The size of the split state cache is unbounded and can theoretically grow infinitely large.
    // However, it is also expensive to build and clearing it on every rebuild in ggml_backend_meta_graph_compute is too expensive.
    ساكن constexpr size_t nbtc = GGML_TENSOR_اندازه - حجم(ggml_tensor::padding);
    std::map<std::pair<ثابت ggml_tensor *, منطقي>, std::pair<ggml_backend_meta_وېشل_state, حرف[nbtc]>> وېشل_state_cache;

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
        أرجع stc_compute[stc_compute_شاخص];
    }
};

ساكن فراغ ggml_backend_meta_بفر_free_بفر(ggml_backend_بفر_t بفر) {
    GGML_ASSERT(ggml_backend_بفر_is_meta(بفر));
    ggml_backend_meta_بفر_context * buf_ctx = (ggml_backend_meta_بفر_context *) بفر->context;
    ړنګول buf_ctx;
}

ساكن size_t ggml_backend_meta_بفر_n_bufs(ggml_backend_بفر_t meta_buf) {
    GGML_ASSERT(ggml_backend_بفر_is_meta(meta_buf));
    ggml_backend_meta_بفر_context * buf_ctx = (ggml_backend_meta_بفر_context *) meta_buf->context;
    أرجع buf_ctx->bufs.size();
}

ساكن ggml_backend_بفر_t ggml_backend_meta_بفر_simple_بفر(ggml_backend_بفر_t meta_buf, size_t شاخص) {
    GGML_ASSERT(ggml_backend_بفر_is_meta(meta_buf));
    ggml_backend_meta_بفر_context * buf_ctx = (ggml_backend_meta_بفر_context *) meta_buf->context;
    GGML_ASSERT(شاخص < buf_ctx->bufs.size());
    أرجع buf_ctx->bufs[شاخص].get();
}

ساكن بنية ggml_tensor * ggml_backend_meta_بفر_simple_tensor(ثابت بنية ggml_tensor * tensor, size_t شاخص) {
    GGML_ASSERT(ggml_backend_بفر_is_meta(tensor->بفر));
    ggml_backend_meta_بفر_context * buf_ctx = (ggml_backend_meta_بفر_context *) tensor->بفر->context;
    GGML_ASSERT(شاخص < buf_ctx->bufs.size());

    ggml_backend_meta_simple_tensor_container & stc = buf_ctx->get_simple_tensor_container(tensor);
    خپلکاری it = stc.simple_tensors.find(tensor);
    إذا (it == stc.simple_tensors.end()) {
        أرجع تش_اشاره;
    }
    أرجع it->second[شاخص];
}

ساكن بنية ggml_backend_meta_وېشل_state ggml_backend_meta_get_وېشل_state(ثابت بنية ggml_tensor * tensor, منطقي assume_sync);

ساكن بنية ggml_backend_meta_وېشل_state ggml_backend_meta_get_وېشل_state(
        ggml_backend_meta_simple_tensor_container & stc, ثابت بنية ggml_tensor * tensor, منطقي assume_sync) {
    // FIXME Currently this function preserves/erases the information in n_segments and nr in an inconsistent way.
    // Since the operations in question are developed specifically for llama.cpp this currently does not manifest as a bug there.
    // However, in a broader ggml context with arbitrary ggml graphs this can lead to unexpected results.
    ثابت size_t n_bufs = ggml_backend_meta_بفر_n_bufs(tensor->بفر);
    ggml_backend_meta_بفر_context * buf_ctx = (ggml_backend_meta_بفر_context *) tensor->بفر->context;

    خپلکاری وېشل_حالتونه_equal = [&](ثابت ggml_backend_meta_وېشل_state & a, ثابت ggml_backend_meta_وېشل_state & b) -> منطقي {
        إذا (a.axis != b.axis) {
            أرجع خطأ;
        }
        لكل (size_t j = 0; j < n_bufs; j++) {
            int64_t مجموعه_a = 0;
            لكل (size_t s = 0; s < a.n_segments; s++) {
                مجموعه_a += a.ne[s*n_bufs + j] * a.nr[s];
            }
            int64_t مجموعه_b = 0;
            لكل (size_t s = 0; s < b.n_segments; s++) {
                مجموعه_b += b.ne[s*n_bufs + j] * b.nr[s];
            }
            إذا (مجموعه_a != مجموعه_b) {
                أرجع خطأ;
            }
        }
        أرجع ریښتیا;
    };

    خپلکاری handle_generic = [&](ثابت std::vector<ggml_backend_meta_وېشل_state> & src_ss, منطقي scalar_only) -> ggml_backend_meta_وېشل_state {
        ggml_backend_meta_وېشل_state ret = {GGML_BACKEND_وېشل_AXIS_NONE, {0}, {1}, 1};
        لكل (size_t i = 0; i < GGML_MAX_SRC; i++) {
            إذا (tensor->src[i] == تش_اشاره || tensor->src[i] == tensor) {
                استمر;
            }
            إذا (ret.axis == GGML_BACKEND_وېشل_AXIS_NONE) {
                ret = src_ss[i];
            } وإلا إذا (!وېشل_حالتونه_equal(src_ss[i], ret)) {
                ret = {GGML_BACKEND_وېشل_AXIS_UNKNOWN, {0}, {1}, 1};
                اكسر;
            }
        }
        إذا (ret.axis == GGML_BACKEND_وېشل_AXIS_NONE) {
            ret = {GGML_BACKEND_وېشل_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        إذا (scalar_only && ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
            ret = {GGML_BACKEND_وېشل_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        GGML_ASSERT(ret.axis != GGML_BACKEND_وېشل_AXIS_UNKNOWN);
        أرجع ret;
    };

    // Some ops process data on a per-row bases:
    خپلکاری handle_فی_لیکه = [&](ثابت std::vector<ggml_backend_meta_وېشل_state> & src_ss) -> ggml_backend_meta_وېشل_state {
        GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_وېشل_AXIS_0);
        أرجع src_ss[0];
    };

    // Some ops broadcast the src1 data across src0:
    خپلکاری handle_bin_bcast = [&](ثابت std::vector<ggml_backend_meta_وېشل_state> & src_ss) -> ggml_backend_meta_وېشل_state {
        إذا (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS &&
                tensor->src[1]->ne[src_ss[0].axis] == 1 && src_ss[1].axis == GGML_BACKEND_وېشل_AXIS_MIRRORED) {
            أرجع src_ss[0];
        }
        إذا (src_ss[2].axis == GGML_BACKEND_وېشل_AXIS_MIRRORED && (src_ss[0].axis == src_ss[1].axis ||
           (src_ss[0].axis == GGML_BACKEND_وېشل_AXIS_MIRRORED && (src_ss[1].axis == GGML_BACKEND_وېشل_AXIS_PARTIAL)))) {
            أرجع src_ss[0]; // GGML_OP_ADD_ID
        }
        GGML_ASSERT(tensor->src[2] == تش_اشاره || src_ss[2].axis == GGML_BACKEND_وېشل_AXIS_MIRRORED);
        أرجع handle_generic(src_ss, /*scalar_only =*/ خطأ);
    };

    خپلکاری handle_concat = [&](ثابت std::vector<ggml_backend_meta_وېشل_state> & src_ss) -> ggml_backend_meta_وېشل_state {
        ثابت ggml_backend_meta_وېشل_axis concat_axis = ggml_backend_meta_وېشل_axis(ggml_get_op_params_i32(tensor, 0));
        إذا (src_ss[0].axis == GGML_BACKEND_وېشل_AXIS_MIRRORED && src_ss[1].axis >= 0 && src_ss[1].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(concat_axis != src_ss[1].axis);
            أرجع src_ss[1];
        }
        إذا (src_ss[1].axis == GGML_BACKEND_وېشل_AXIS_MIRRORED && src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(concat_axis != src_ss[0].axis);
            أرجع src_ss[0];
        }
        إذا (src_ss[0].axis == src_ss[1].axis && src_ss[0].axis != concat_axis) {
            أرجع src_ss[0];
        }
        أرجع handle_generic(src_ss, /*scalar_only =*/ ریښتیا);
    };

    خپلکاری handle_mul_mat = [&](ثابت std::vector<ggml_backend_meta_وېشل_state> & src_ss) -> ggml_backend_meta_وېشل_state {
        إذا (src_ss[0].axis == GGML_BACKEND_وېشل_AXIS_MIRRORED && src_ss[1].axis == GGML_BACKEND_وېشل_AXIS_MIRRORED) {
            أرجع {GGML_BACKEND_وېشل_AXIS_MIRRORED, {0}, {1}, 1};
        }
        إذا (src_ss[0].axis == GGML_BACKEND_وېشل_AXIS_1 && src_ss[1].axis == GGML_BACKEND_وېشل_AXIS_MIRRORED) {
            ggml_backend_meta_وېشل_state ret = src_ss[0];
            ret.axis = GGML_BACKEND_وېشل_AXIS_0;
            ret.nr[0] = 1;
            ret.n_segments = 1;
            أرجع ret;
        }
        إذا (src_ss[1].axis == GGML_BACKEND_وېشل_AXIS_1 && src_ss[0].axis == GGML_BACKEND_وېشل_AXIS_MIRRORED) {
            أرجع src_ss[1];
        }
        إذا (src_ss[0].axis == GGML_BACKEND_وېشل_AXIS_0 && src_ss[1].axis == GGML_BACKEND_وېشل_AXIS_0) {
            GGML_ASSERT(وېشل_حالتونه_equal(src_ss[0], src_ss[1]));
            أرجع {assume_sync ? GGML_BACKEND_وېشل_AXIS_MIRRORED : GGML_BACKEND_وېشل_AXIS_PARTIAL, {0}, {1}, 1};
        }
        إذا (src_ss[0].axis == src_ss[1].axis && src_ss[0].axis >= GGML_BACKEND_وېشل_AXIS_2 &&
                src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(وېشل_حالتونه_equal(src_ss[0], src_ss[1]));
            أرجع src_ss[0];
        }
        // batched matmul with the batches split across devices and a replicated activation
        إذا (src_ss[0].axis >= GGML_BACKEND_وېشل_AXIS_2 && src_ss[0].axis < GGML_MAX_DIMS &&
                src_ss[1].axis == GGML_BACKEND_وېشل_AXIS_MIRRORED) {
            أرجع src_ss[0];
        }
        GGML_ABORT("unsupported mul_mat split states: node=%s src0=%s axis=%d src1=%s axis=%d",
            tensor->name, tensor->src[0]->name, (صحيح) src_ss[0].axis, tensor->src[1]->name, (صحيح) src_ss[1].axis);
        //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
    };

    خپلکاری handle_reshape = [&](ثابت std::vector<ggml_backend_meta_وېشل_state> & src_ss) -> ggml_backend_meta_وېشل_state {
        بدل (src_ss[0].axis) {
            حالة GGML_BACKEND_وېشل_AXIS_0:
            حالة GGML_BACKEND_وېشل_AXIS_1:
            حالة GGML_BACKEND_وېشل_AXIS_2:
            حالة GGML_BACKEND_وېشل_AXIS_3: {
                int64_t بنسټ_ne_دننه = 1;
                لكل (صحيح dim = 0; dim <= src_ss[0].axis; dim++) {
                    بنسټ_ne_دننه *= tensor->src[0]->ne[dim];
                }
                إذا (src_ss[0].n_segments == 1) {
                    بنسټ_ne_دننه /= src_ss[0].nr[0];
                    إذا (src_ss[0].axis == ggml_n_dims(tensor->src[0]) - 1 && src_ss[0].nr[0] == 1) {
                        أرجع {ggml_backend_meta_وېشل_axis(ggml_n_dims(tensor) - 1), {0}, {1}, 1};
                    }
                    إذا (src_ss[0].axis == GGML_BACKEND_وېشل_AXIS_0 && tensor->ne[0] == tensor->src[0]->ne[0] &&
                            tensor->ne[1] == 1 && src_ss[0].nr[0] == 1) {
                        منطقي complete_لیکې = ریښتیا;
                        لكل (size_t j = 0; j < n_bufs; j++) {
                            ثابت int64_t ne = src_ss[0].ne[j];
                            complete_لیکې = complete_لیکې && (ne == 0 || ne == tensor->src[0]->ne[0]);
                        }
                        إذا (complete_لیکې) {
                            // Move a complete dim-0 split to the following singleton dimension.
                            أرجع {GGML_BACKEND_وېشل_AXIS_1, {0}, {1}, 1};
                        }
                    }
                }
                // Reshape outputs use one segment; split-state propagation merges source segments.
                int64_t بنسټ_ne_بهر = 1;
                لكل (صحيح dim = 0; dim < GGML_MAX_DIMS; dim++) {
                    بنسټ_ne_بهر *= tensor->ne[dim];
                    إذا (بنسټ_ne_بهر % بنسټ_ne_دننه == 0) {
                        أرجع {ggml_backend_meta_وېشل_axis(dim), {0}, {uint32_t(بنسټ_ne_بهر/بنسټ_ne_دننه)}, 1};
                    }
                    إذا (بنسټ_ne_بهر > بنسټ_ne_دننه) {
                        GGML_ASSERT(src_ss[0].n_segments == 1);
                        GGML_ASSERT(src_ss[0].nr[0]      == 1);
                        أرجع {ggml_backend_meta_وېشل_axis(dim), {0}, {1}, 1};
                    }
                }
                GGML_ABORT("shape mismatch for %s", ggml_op_name(tensor->op));
            }
            حالة GGML_BACKEND_وېشل_AXIS_MIRRORED:
            حالة GGML_BACKEND_وېشل_AXIS_PARTIAL: {
                أرجع src_ss[0];
            }
            افتراضي: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    خپلکاری handle_cpy = [&](ثابت std::vector<ggml_backend_meta_وېشل_state> & src_ss) -> ggml_backend_meta_وېشل_state {
        إذا (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            أرجع handle_reshape(src_ss);
        }
        أرجع handle_generic(src_ss, /*scalar_only =*/ خطأ);
    };

    خپلکاری handle_view = [&](ثابت std::vector<ggml_backend_meta_وېشل_state> & src_ss) -> ggml_backend_meta_وېشل_state {
        إذا (ggml_is_contiguous(tensor) && ggml_is_contiguous(tensor->src[0])) {
            أرجع handle_reshape(src_ss);
        }
        ثابت صحيح axis = src_ss[0].axis;
        {
            منطقي all_strides_the_same = ریښتیا;
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
                    أرجع {ggml_backend_meta_وېشل_axis(dim), {0}, {1}, 1};
                }
            }
            GGML_ABORT("fatal error");
        }
        إذا (src_ss[0].axis == GGML_BACKEND_وېشل_AXIS_MIRRORED || src_ss[0].axis == GGML_BACKEND_وېشل_AXIS_PARTIAL) {
            أرجع src_ss[0];
        }
        GGML_ABORT("view of permuted tensor not implemented");
        //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
    };

    خپلکاری handle_ترتیب_بدلول = [&](ثابت std::vector<ggml_backend_meta_وېشل_state> & src_ss) -> ggml_backend_meta_وېشل_state {
        بدل (src_ss[0].axis) {
            حالة GGML_BACKEND_وېشل_AXIS_0:
            حالة GGML_BACKEND_وېشل_AXIS_1:
            حالة GGML_BACKEND_وېشل_AXIS_2:
            حالة GGML_BACKEND_وېشل_AXIS_3: {
                GGML_ASSERT(src_ss[0].n_segments == 1 || src_ss[0].nr[0] == 1);
                أرجع {ggml_backend_meta_وېشل_axis(tensor->op_params[src_ss[0].axis]), {0}, {src_ss[0].nr[0]}, 1};
            }
            حالة GGML_BACKEND_وېشل_AXIS_MIRRORED:
            حالة GGML_BACKEND_وېشل_AXIS_PARTIAL: {
                أرجع src_ss[0];
            }
            افتراضي: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    خپلکاری handle_transpose = [&](ثابت std::vector<ggml_backend_meta_وېشل_state> & src_ss) -> ggml_backend_meta_وېشل_state {
        بدل (src_ss[0].axis) {
            حالة GGML_BACKEND_وېشل_AXIS_0:
            حالة GGML_BACKEND_وېشل_AXIS_1: {
                GGML_ASSERT(src_ss[0].n_segments == 1 || src_ss[0].nr[0] == 1);
                أرجع {ggml_backend_meta_وېشل_axis(صحيح(src_ss[0].axis) ^ 1), {0}, {src_ss[0].nr[0]}, 1};
            }
            حالة GGML_BACKEND_وېشل_AXIS_2:
            حالة GGML_BACKEND_وېشل_AXIS_3:
            حالة GGML_BACKEND_وېشل_AXIS_MIRRORED:
            حالة GGML_BACKEND_وېشل_AXIS_PARTIAL: {
                أرجع src_ss[0];
            }
            افتراضي: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    خپلکاری handle_get_لیکې = [&](ثابت std::vector<ggml_backend_meta_وېشل_state> & src_ss) -> ggml_backend_meta_وېشل_state {
        إذا (src_ss[0].axis == GGML_BACKEND_وېشل_AXIS_0 && src_ss[1].axis == GGML_BACKEND_وېشل_AXIS_MIRRORED) {
            أرجع src_ss[0];
        }
        أرجع handle_generic(src_ss, /*scalar_only =*/ ریښتیا);
    };

    خپلکاری handle_سېټ_لیکې = [&](ثابت std::vector<ggml_backend_meta_وېشل_state> & src_ss) -> ggml_backend_meta_وېشل_state {
        GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_وېشل_AXIS_1);
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_وېشل_AXIS_MIRRORED);
        GGML_ASSERT(وېشل_حالتونه_equal(src_ss[0], src_ss[2]));
        أرجع src_ss[0];
    };

    خپلکاری handle_rope = [&](ثابت std::vector<ggml_backend_meta_وېشل_state> & src_ss) -> ggml_backend_meta_وېشل_state {
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_وېشل_AXIS_MIRRORED);
        أرجع src_ss[0];
    };

    خپلکاری handle_pad = [&](ثابت std::vector<ggml_backend_meta_وېشل_state> & src_ss) -> ggml_backend_meta_وېشل_state {
        إذا (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(tensor->op_params[2*src_ss[0].axis + 0] == 0);
            GGML_ASSERT(tensor->op_params[2*src_ss[0].axis + 1] == 0);
        }
        أرجع src_ss[0];
    };

    خپلکاری handle_pool_1d = [&](ثابت std::vector<ggml_backend_meta_وېشل_state> & src_ss) -> ggml_backend_meta_وېشل_state {
        إذا (src_ss[0].axis == GGML_BACKEND_وېشل_AXIS_0) {
            أرجع {GGML_BACKEND_وېشل_AXIS_MIRRORED, {0}, {1}, 1};
        }
        أرجع src_ss[0];
    };

    خپلکاری handle_flash_attn_ext = [&](ثابت std::vector<ggml_backend_meta_وېشل_state> & src_ss) -> ggml_backend_meta_وېشل_state {
        GGML_ASSERT(tensor->src[3] == تش_اشاره || src_ss[3].axis == GGML_BACKEND_وېشل_AXIS_MIRRORED);

        إذا (src_ss[0].axis == GGML_BACKEND_وېشل_AXIS_MIRRORED) {
            GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_وېشل_AXIS_MIRRORED);
            GGML_ASSERT(src_ss[2].axis == GGML_BACKEND_وېشل_AXIS_MIRRORED);
            GGML_ASSERT(tensor->src[4] == تش_اشاره || src_ss[4].axis == GGML_BACKEND_وېشل_AXIS_MIRRORED);
            أرجع {GGML_BACKEND_وېشل_AXIS_MIRRORED, {0}, {1}, 1};
        }

        GGML_ASSERT(src_ss[0].axis == GGML_BACKEND_وېشل_AXIS_2);
        ثابت منطقي kv_وېشل = src_ss[1].axis == GGML_BACKEND_وېشل_AXIS_2 &&
                src_ss[2].axis == GGML_BACKEND_وېشل_AXIS_2;
        ثابت منطقي kv_mirrored = src_ss[1].axis == GGML_BACKEND_وېشل_AXIS_MIRRORED &&
                src_ss[2].axis == GGML_BACKEND_وېشل_AXIS_MIRRORED;
        GGML_ASSERT(kv_وېشل || kv_mirrored);
        GGML_ASSERT(tensor->src[4] == تش_اشاره || src_ss[4].axis == GGML_BACKEND_وېشل_AXIS_0);
        أرجع {GGML_BACKEND_وېشل_AXIS_1, {0}, {1}, 1};
    };

    خپلکاری handle_lightning_indexer = [&](
            ثابت std::vector<ggml_backend_meta_وېشل_state> & src_ss) -> ggml_backend_meta_وېشل_state {
        لكل (size_t i = 0; i < 4; i++) {
            GGML_ASSERT(src_ss[i].axis == GGML_BACKEND_وېشل_AXIS_MIRRORED);
        }
        أرجع {GGML_BACKEND_وېشل_AXIS_MIRRORED, {0}, {1}, 1};
    };

    خپلکاری handle_ssm_conv = [&](ثابت std::vector<ggml_backend_meta_وېشل_state> & src_ss) -> ggml_backend_meta_وېشل_state {
        إذا (src_ss[0].axis == src_ss[1].axis) {
            إذا (src_ss[0].axis == GGML_BACKEND_وېشل_AXIS_0) {
                أرجع {GGML_BACKEND_وېشل_AXIS_1, {0}, {1}, 1};
            }
            إذا (src_ss[0].axis == GGML_BACKEND_وېشل_AXIS_1) {
                أرجع {GGML_BACKEND_وېشل_AXIS_0, {0}, {1}, 1};
            }
        }
        أرجع handle_generic(src_ss, /*scalar_only =*/ خطأ);
    };

    خپلکاری handle_gated_delta_net = [&](ثابت std::vector<ggml_backend_meta_وېشل_state> & src_ss) -> ggml_backend_meta_وېشل_state {
        إذا (src_ss[0].axis == GGML_BACKEND_وېشل_AXIS_MIRRORED && src_ss[1].axis == GGML_BACKEND_وېشل_AXIS_MIRRORED &&
                src_ss[2].axis == GGML_BACKEND_وېشل_AXIS_MIRRORED && src_ss[3].axis == GGML_BACKEND_وېشل_AXIS_MIRRORED &&
                src_ss[4].axis == GGML_BACKEND_وېشل_AXIS_MIRRORED && src_ss[5].axis == GGML_BACKEND_وېشل_AXIS_MIRRORED) {
            أرجع src_ss[0];
        }
        GGML_ASSERT(src_ss[0].axis == GGML_BACKEND_وېشل_AXIS_1);
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_وېشل_AXIS_1);
        GGML_ASSERT(src_ss[2].axis == GGML_BACKEND_وېشل_AXIS_1);
        GGML_ASSERT(src_ss[3].axis == GGML_BACKEND_وېشل_AXIS_1);
        GGML_ASSERT(src_ss[4].axis == GGML_BACKEND_وېشل_AXIS_1);
        // state shape is [S_v, S_v, H_v, n_seqs] (s0 only); the heads dim is its own axis 2,
        // so a head-aligned split on the input cache lands on axis 2 here.
        GGML_ASSERT(src_ss[5].axis == GGML_BACKEND_وېشل_AXIS_2 || src_ss[5].axis == GGML_BACKEND_وېشل_AXIS_1 || src_ss[5].axis == GGML_BACKEND_وېشل_AXIS_0);
        أرجع {GGML_BACKEND_وېشل_AXIS_0, {0}, {1}, 1};
    };

    خپلکاری calculate_وېشل_state = [&]() -> ggml_backend_meta_وېشل_state {
        إذا (ggml_nelements(tensor) == 0) {
            أرجع {GGML_BACKEND_وېشل_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        إذا (ggml_backend_بفر_get_usage(tensor->بفر) != GGML_BACKEND_بفر_USAGE_COMPUTE && tensor->view_src == تش_اشاره) {
            ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_بفر_get_type(tensor->بفر));
            ثابت ggml_backend_meta_device_context * dev_ctx = (ثابت ggml_backend_meta_device_context *) dev->context;
            ggml_backend_meta_وېشل_state ret = dev_ctx->get_وېشل_state(tensor, dev_ctx->get_وېشل_state_ud);
            إذا (ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
                ثابت int64_t granularity = ret.axis == GGML_BACKEND_وېشل_AXIS_0 ? ggml_blck_اندازه(tensor->type) : 1;
                int64_t ne_مجموعه = 0;
                لكل (size_t s = 0; s < ret.n_segments; s++) {
                    لكل (size_t j = 0; j < n_bufs; j++) {
                        GGML_ASSERT(ret.ne[s*n_bufs + j] % granularity == 0);
                        ne_مجموعه += ret.ne[s*n_bufs + j] * ret.nr[s];
                    }
                }
                GGML_ASSERT(ne_مجموعه == tensor->ne[ret.axis]);
            } وإلا إذا (ret.axis == GGML_BACKEND_وېشل_AXIS_PARTIAL) {
                GGML_ASSERT(ret.n_segments == 1);
                GGML_ASSERT(ret.nr[0] == 1);
            }
            أرجع ret;
        }

        std::vector<ggml_backend_meta_وېشل_state> src_ss(GGML_MAX_SRC, {GGML_BACKEND_وېشل_AXIS_NONE, {0}, {1}, 1});
        لكل (size_t i = 0; i < GGML_MAX_SRC; i++) {
            إذا (tensor->src[i] == تش_اشاره || tensor->src[i] == tensor) {
                src_ss[i] = {GGML_BACKEND_وېشل_AXIS_UNKNOWN, {0}, {1}, 1};
                استمر;
            }
            src_ss[i] = ggml_backend_meta_get_وېشل_state(stc, tensor->src[i], /*assume_sync =*/ ریښتیا);
            GGML_ASSERT(src_ss[i].axis != GGML_BACKEND_وېشل_AXIS_UNKNOWN);
        }

        ggml_backend_meta_وېشل_state وېشل_state;
        بدل (tensor->op) {
            حالة GGML_OP_NONE: {
                إذا (tensor->view_src != تش_اشاره) {
                    // full-tensor view created with ggml_view_tensor, transparent for the split state
                    وېشل_state = ggml_backend_meta_get_وېشل_state(stc, tensor->view_src, assume_sync);
                } وإلا {
                    وېشل_state = {GGML_BACKEND_وېشل_AXIS_MIRRORED, {0}, {1}, 1};
                }
            } اكسر;
            حالة GGML_OP_DUP: {
                وېشل_state = handle_generic(src_ss, /*scalar_only =*/ ریښتیا);
            } اكسر;
            حالة GGML_OP_جمع_کړه:
            حالة GGML_OP_جمع_کړه_ID: {
                وېشل_state = handle_bin_bcast(src_ss);
            } اكسر;
            حالة GGML_OP_ADD1:
            حالة GGML_OP_ACC: {
                وېشل_state = handle_generic(src_ss, /*scalar_only =*/ ریښتیا);
            } اكسر;
            حالة GGML_OP_SUB:
            حالة GGML_OP_MUL:
            حالة GGML_OP_DIV: {
                وېشل_state = handle_bin_bcast(src_ss);
            } اكسر;
            حالة GGML_OP_SQR:
            حالة GGML_OP_SQRT:
            حالة GGML_OP_LOG:
            حالة GGML_OP_SIN:
            حالة GGML_OP_COS: {
                وېشل_state = handle_generic(src_ss, /*scalar_only =*/ خطأ);
            } اكسر;
            حالة GGML_OP_مجموعه: {
                وېشل_state = handle_generic(src_ss, /*scalar_only =*/ ریښتیا);
            } اكسر;
            حالة GGML_OP_مجموعه_لیکې:
            حالة GGML_OP_CUMSUM:
            حالة GGML_OP_منځنۍ:
            حالة GGML_OP_ARGMAX:
            حالة GGML_OP_شمېر_EQUAL: {
                وېشل_state = handle_فی_لیکه(src_ss);
            } اكسر;
            حالة GGML_OP_REPEAT:
            حالة GGML_OP_REPEAT_شا: {
                وېشل_state = handle_generic(src_ss, /*scalar_only =*/ خطأ);
            } اكسر;
            حالة GGML_OP_CONCAT: {
                وېشل_state = handle_concat(src_ss);
            } اكسر;
            حالة GGML_OP_SILU_شا: {
                وېشل_state = handle_generic(src_ss, /*scalar_only =*/ خطأ);
            } اكسر;
            حالة GGML_OP_NORM:
            حالة GGML_OP_RMS_NORM:
            حالة GGML_OP_RMS_NORM_شا:
            حالة GGML_OP_ډله_NORM:
            حالة GGML_OP_L2_NORM: {
                وېشل_state = handle_فی_لیکه(src_ss);
            } اكسر;
            حالة GGML_OP_MUL_MAT:
            حالة GGML_OP_MUL_MAT_ID: {
                وېشل_state = handle_mul_mat(src_ss);
            } اكسر;
            حالة GGML_OP_بهر_PROD: {
                وېشل_state = handle_generic(src_ss, /*scalar_only =*/ ریښتیا);
            } اكسر;
            حالة GGML_OP_SCALE: {
                وېشل_state = handle_generic(src_ss, /*scalar_only =*/ خطأ);
            } اكسر;
            حالة GGML_OP_سېټ: {
                وېشل_state = handle_generic(src_ss, /*scalar_only =*/ ریښتیا);
            } اكسر;
            حالة GGML_OP_CPY: {
                وېشل_state = handle_cpy(src_ss);
            } اكسر;
            حالة GGML_OP_CONT:
            حالة GGML_OP_RESHAPE: {
                وېشل_state = handle_reshape(src_ss);
            } اكسر;
            حالة GGML_OP_VIEW: {
                وېشل_state = handle_view(src_ss);
            } اكسر;
            حالة GGML_OP_ترتیب_بدلول: {
                وېشل_state = handle_ترتیب_بدلول(src_ss);
            } اكسر;
            حالة GGML_OP_TRANSPOSE: {
                وېشل_state = handle_transpose(src_ss);
            } اكسر;
            حالة GGML_OP_GET_لیکې: {
                وېشل_state = handle_get_لیکې(src_ss);
            } اكسر;
            حالة GGML_OP_GET_لیکې_شا: {
                وېشل_state = handle_generic(src_ss, /*scalar_only =*/ ریښتیا);
            } اكسر;
            حالة GGML_OP_سېټ_لیکې: {
                وېشل_state = handle_سېټ_لیکې(src_ss);
            } اكسر;
            حالة GGML_OP_DIAG:
            حالة GGML_OP_DIAG_ماسک_INF:
            حالة GGML_OP_DIAG_ماسک_ZERO: {
                وېشل_state = handle_generic(src_ss, /*scalar_only =*/ ریښتیا);
            } اكسر;
            حالة GGML_OP_SOFT_MAX:
            حالة GGML_OP_SOFT_MAX_شا: {
                وېشل_state = handle_generic(src_ss, /*scalar_only =*/ خطأ);
            } اكسر;
            حالة GGML_OP_ROPE: {
                وېشل_state = handle_rope(src_ss);
            } اكسر;
            حالة GGML_OP_ROPE_شا: {
                وېشل_state = handle_rope(src_ss);
            } اكسر;
            حالة GGML_OP_CLAMP: {
                وېشل_state = handle_generic(src_ss, /*scalar_only =*/ خطأ);
            } اكسر;
            حالة GGML_OP_CONV_TRANSPOSE_1D:
            حالة GGML_OP_IM2_ستون:
            حالة GGML_OP_IM2_ستون_شا:
            حالة GGML_OP_IM2_ستون_3_D:
            حالة GGML_OP_CONV_2D:
            حالة GGML_OP_CONV_3D:
            حالة GGML_OP_CONV_2D_DW:
            حالة GGML_OP_CONV_TRANSPOSE_2D:
            حالة GGML_OP_POOL_2D:
            حالة GGML_OP_POOL_2_D_شا:
            حالة GGML_OP_UPSCALE: {
                وېشل_state = handle_generic(src_ss, /*scalar_only =*/ ریښتیا);
            } اكسر;
            حالة GGML_OP_POOL_1D: {
                وېشل_state = handle_pool_1d(src_ss);
            } اكسر;
            حالة GGML_OP_PAD: {
                وېشل_state = handle_pad(src_ss);
            } اكسر;
            حالة GGML_OP_PAD_REFLECT_1D:
            حالة GGML_OP_ROLL:
            حالة GGML_OP_ARANGE:
            حالة GGML_OP_TIMESTEP_EMBEDDING: {
                وېشل_state = handle_generic(src_ss, /*scalar_only =*/ ریښتیا);
            } اكسر;
            حالة GGML_OP_ARGSORT:
            حالة GGML_OP_سر_K: {
                وېشل_state = handle_فی_لیکه(src_ss);
            } اكسر;
            حالة GGML_OP_LEAKY_RELU: {
                وېشل_state = handle_generic(src_ss, /*scalar_only =*/ خطأ);
            } اكسر;
            حالة GGML_OP_TRI: {
                وېشل_state = handle_generic(src_ss, /*scalar_only =*/ ریښتیا);
            } اكسر;
            حالة GGML_OP_FILL: {
                وېشل_state = handle_generic(src_ss, /*scalar_only =*/ خطأ);
            } اكسر;
            حالة GGML_OP_FLASH_ATTN_EXT: {
                وېشل_state = handle_flash_attn_ext(src_ss);
            } اكسر;
            حالة GGML_OP_FLASH_ATTN_شا: {
                وېشل_state = handle_generic(src_ss, /*scalar_only =*/ ریښتیا);
            } اكسر;
            حالة GGML_OP_SSM_CONV: {
                وېشل_state = handle_ssm_conv(src_ss);
            } اكسر;
            حالة GGML_OP_SSM_لټون:
            حالة GGML_OP_WIN_PART:
            حالة GGML_OP_WIN_UNPART:
            حالة GGML_OP_GET_REL_POS:
            حالة GGML_OP_جمع_کړه_REL_POS:
            حالة GGML_OP_RWKV_WKV6:
            حالة GGML_OP_GATED_خطي_ATTN:
            حالة GGML_OP_RWKV_WKV7:
            حالة GGML_OP_SOLVE_TRI: {
                وېشل_state = handle_generic(src_ss, /*scalar_only =*/ ریښتیا);
            } اكسر;
            حالة GGML_OP_GATED_DELTA_NET: {
                وېشل_state = handle_gated_delta_net(src_ss);
            } اكسر;
            حالة GGML_OP_LIGHTNING_INDEXER: {
                وېشل_state = handle_lightning_indexer(src_ss);
            } اكسر;
            حالة GGML_OP_DSV4_HC_COMB:
            حالة GGML_OP_DSV4_HC_PRE:
            حالة GGML_OP_DSV4_HC_POST:
            حالة GGML_OP_سر_K_ماسک: {
                وېشل_state = handle_generic(src_ss, /*scalar_only =*/ ریښتیا);
            } اكسر;
            حالة GGML_OP_UNARY: {
                وېشل_state = handle_generic(src_ss, /*scalar_only =*/ خطأ);
            } اكسر;
            حالة GGML_OP_MAP_CUSTOM1:
            حالة GGML_OP_MAP_CUSTOM2:
            حالة GGML_OP_MAP_CUSTOM3:
            حالة GGML_OP_CUSTOM: {
                وېشل_state = handle_generic(src_ss, /*scalar_only =*/ ریښتیا);
            } اكسر;
            حالة GGML_OP_CROSS_ENTROPY_LOSS:
            حالة GGML_OP_CROSS_ENTROPY_LOSS_شا: {
                وېشل_state = handle_فی_لیکه(src_ss);
            } اكسر;
            حالة GGML_OP_OPT_ګام_ADAMW:
            حالة GGML_OP_OPT_ګام_SGD:
            حالة GGML_OP_GLU: {
                وېشل_state = handle_generic(src_ss, /*scalar_only =*/ خطأ);
            } اكسر;
            افتراضي: {
                GGML_ABORT("ggml op not implemented: %s", ggml_op_name(tensor->op));
                وېشل_state = {GGML_BACKEND_وېشل_AXIS_UNKNOWN, {0}, {1}, 1};
            } اكسر;
        }
        إذا (وېشل_state.axis >= 0 && وېشل_state.axis < GGML_MAX_DIMS) {
            منطقي لومړی_src_وېشل_by_axis = ریښتیا;
            ثابت size_t n_bufs = ggml_backend_meta_بفر_n_bufs(tensor->بفر);

            لكل (size_t i = 0; i < GGML_MAX_SRC; i++) {
                إذا (tensor->src[i] == تش_اشاره || src_ss[i].axis < 0 || src_ss[i].axis >= GGML_MAX_DIMS) {
                    استمر;
                }
                إذا (لومړی_src_وېشل_by_axis) {
                    لكل (size_t j = 0; j < n_bufs; j++) {
                        // Take over ratio from src:
                        لكل (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            وېشل_state.ne[s*n_bufs + j] = 0;
                        }
                        لكل (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            وېشل_state.ne[j] += src_ss[i].ne[s*n_bufs + j] * src_ss[i].nr[s];
                        }
                        وېشل_state.ne[j] *= tensor->ne[وېشل_state.axis];
                        إذا (وېشل_state.ne[j] != 0 || tensor->src[i]->ne[src_ss[i].axis] != 0) {
                            ثابت int64_t div = tensor->src[i]->ne[src_ss[i].axis] * وېشل_state.nr[0];
                            GGML_ASSERT(وېشل_state.ne[j] % div == 0);
                            وېشل_state.ne[j] /= div;
                        }
                    }
                } وإلا {
                    GGML_ASSERT(وېشل_state.n_segments == 1);
                    لكل (size_t j = 0; j < n_bufs; j++) {
                        // Assert that ratio is consistent:
                        int64_t مجموعه = 0;
                        لكل (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            مجموعه += src_ss[i].ne[s*n_bufs + j] * src_ss[i].nr[s];
                        }
                        GGML_ASSERT(وېشل_state.ne[j]*وېشل_state.nr[0] * tensor->src[i]->ne[src_ss[i].axis]
                                                                 == مجموعه * tensor->ne[وېشل_state.axis]);
                    }
                }
                لومړی_src_وېشل_by_axis = خطأ;
            }
            GGML_ASSERT(!لومړی_src_وېشل_by_axis);
        }
        أرجع وېشل_state;
    };

    ثابت std::pair کلی = std::make_pair(tensor, assume_sync);
    خپلکاری it = buf_ctx->وېشل_state_cache.find(کلی);
    إذا (it != buf_ctx->وېشل_state_cache.end() && memcmp(it->second.second, (ثابت حرف *) tensor, حجم(it->second.second)) != 0) {
        buf_ctx->وېشل_state_cache.clear();
        it = buf_ctx->وېشل_state_cache.end();
    }

    إذا (it == buf_ctx->وېشل_state_cache.end()) {
        buf_ctx->وېشل_state_cache[کلی].لومړی = calculate_وېشل_state();
        حافظه_نقل(buf_ctx->وېشل_state_cache[کلی].second, tensor, حجم(buf_ctx->وېشل_state_cache[کلی].second));
        إذا (buf_ctx->debug > 0) {
            std::string srcs_info;
            لكل (size_t i = 0; i < GGML_MAX_SRC; i++) {
                إذا (tensor->src[i] == تش_اشاره || tensor->src[i] == tensor) {
                    استمر;
                }
                إذا (!srcs_info.تشه()) {
                    srcs_info += ", ";
                }
                ثابت ggml_backend_meta_وېشل_state وېشل_state =
                        ggml_backend_meta_get_وېشل_state(tensor->src[i], ریښتیا);
                GGML_ASSERT(وېشل_state.n_segments == 1);
                ثابت حرف * axis_name = ggml_backend_meta_وېشل_axis_name(وېشل_state.axis);
                std::string ne_info;
                لكل (size_t j = 0; j < n_bufs; j++) {
                    إذا (!ne_info.تشه()) {
                        ne_info += ", ";
                    }
                    ne_info += std::to_string(وېشل_state.ne[j]) + "x" + std::to_string(وېشل_state.nr[0]);
                }
                srcs_info += std::string(tensor->src[i]->name) + "[" + ggml_op_name(tensor->src[i]->op) + ", " + axis_name + ", {" + ne_info + "}]";
            }
            std::string ne_info;
            لكل (size_t j = 0; j < n_bufs; j++) {
                إذا (!ne_info.تشه()) {
                    ne_info += ", ";
                }
                ثابت ggml_backend_meta_وېشل_state & ss = buf_ctx->وېشل_state_cache[کلی].لومړی;
                ne_info += std::to_string(ss.ne[j]) + "x" + std::to_string(ss.nr[0]);
            }
            GGML_LOG_DEBUG("SPLIT_STATE: {%s} -> %s[%s, %s, {%s}]\n", srcs_info.c_str(), tensor->name, ggml_op_name(tensor->op),
                ggml_backend_meta_وېشل_axis_name(buf_ctx->وېشل_state_cache[کلی].لومړی.axis), ne_info.c_str());
        }
    }

    ggml_backend_meta_وېشل_state ret = buf_ctx->وېشل_state_cache[کلی].لومړی;
    GGML_ASSERT(ret.axis != GGML_BACKEND_وېشل_AXIS_NONE);
#که_نه_تعریف_شوی NDEBUG
    إذا (ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
        int64_t ne_ret = 0;
        لكل (size_t s = 0; s < ret.n_segments; s++) {
            لكل (size_t j = 0; j < n_bufs; j++) {
                ne_ret += ret.ne[s*n_bufs + j] * ret.nr[s];
            }
        }
        أكد(ne_ret == tensor->ne[صحيح(ret.axis)]);
    }
#د_که_پای // NDEBUG
    أرجع ret;
}

ساكن منطقي ggml_backend_meta_is_host_view(ثابت بنية ggml_tensor * tensor) {
    أرجع ggml_is_view(tensor) && ggml_backend_بفر_is_host(tensor->view_src->بفر);
}

ساكن بنية ggml_backend_meta_وېشل_state ggml_backend_meta_get_وېشل_state(ثابت بنية ggml_tensor * tensor, منطقي assume_sync) {
    // [TAG_META_HOST_VIEWS]
    // TODO: technically, this check should not be needed if the backend scheduler correctly prevents assigning
    //       such host-buffer views to the meta backend. figure out how to update the scheduler logic to achieve that
    //       ref: https://github.com/ggml-org/llama.cpp/pull/30217
    إذا (!ggml_backend_بفر_is_meta(tensor->بفر)) {
        GGML_ASSERT(ggml_backend_meta_is_host_view(tensor));

        // the view is not allocated in the meta buffer, it is not split across the sub-devices
        أرجع { GGML_BACKEND_وېشل_AXIS_NONE, {0}, {1}, 1 };
    }

    ggml_backend_meta_بفر_context * buf_ctx = (ggml_backend_meta_بفر_context *) tensor->بفر->context;
    أرجع ggml_backend_meta_get_وېشل_state(buf_ctx->get_simple_tensor_container(tensor), tensor, assume_sync);
}

ساكن فراغ * ggml_backend_meta_بفر_get_بنسټ(ggml_backend_بفر_t بفر) {
    GGML_ناکارول_شوی(بفر);
    أرجع (فراغ *) 0x1000000000000000; // FIXME
}

ساكن تعداد ggml_حالت ggml_backend_meta_بفر_init_tensor_impl(ggml_backend_meta_simple_tensor_container & stc, ggml_tensor * tensor) {
    GGML_ASSERT(ggml_backend_بفر_is_meta(tensor->بفر));
    ggml_backend_meta_بفر_context * buf_ctx = (ggml_backend_meta_بفر_context *) tensor->بفر->context;
    ثابت size_t n_simple_bufs = ggml_backend_meta_بفر_n_bufs(tensor->بفر);

    ثابت ggml_backend_meta_وېشل_state وېشل_state = ggml_backend_meta_get_وېشل_state(stc, tensor, /*assume_sync =*/ ریښتیا);
    GGML_ASSERT(ggml_nelements(tensor) == 0 || وېشل_state.axis != GGML_BACKEND_وېشل_AXIS_UNKNOWN);
    GGML_ASSERT(وېشل_state.n_segments <= 16);

    صحيح وېشل_dim = وېشل_state.axis;
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

        إذا (وېشل_dim >= 0 && وېشل_dim < GGML_MAX_DIMS) {
            // TODO: the following assert fails for llama-parallel even though the results are correct:
            // GGML_ASSERT(ggml_is_contiguously_allocated(tensor));
            ne[وېشل_dim] = 0;
            لكل (size_t s = 0; s < وېشل_state.n_segments; s++) {
                ne[وېشل_dim] += وېشل_state.ne[s*n_simple_bufs + j] * وېشل_state.nr[s];
            }
            لكل (صحيح i = 0; i < GGML_MAX_DIMS; i++) {
                إذا (tensor->nb[i] > tensor->nb[وېشل_dim]) {
                    nb[i] = tensor->nb[i] * ne[وېشل_dim]/tensor->ne[وېشل_dim];
                }
            }
        }

        ggml_tensor * t_ij = ggml_new_tensor(simple_ctx, tensor->type, GGML_MAX_DIMS, ne);
        t_ij->op = tensor->op;
        لكل (صحيح i = 0; i < GGML_MAX_DIMS; i++) {
            t_ij->nb[i] = nb[i];
        }
        t_ij->flags = tensor->flags;
        حافظه_نقل(t_ij->op_params, tensor->op_params, حجم(tensor->op_params));
        ggml_سېټ_name(t_ij, tensor->name);
        t_ij->بفر = simple_buf;
        t_ij->view_src = tensor->view_src;
        t_ij->view_offs = tensor->view_offs;
        إذا (t_ij->view_src != تش_اشاره && ggml_backend_بفر_is_meta(t_ij->view_src->بفر)) {
            t_ij->view_src = ggml_backend_meta_بفر_simple_tensor(tensor->view_src, j);
            إذا (t_ij->view_offs > 0 && وېشل_dim >= 0 && وېشل_dim < GGML_MAX_DIMS) {
                GGML_ASSERT(tensor->ne[وېشل_dim] != 0);
                ثابت صحيح وېشل_dim_view_src = ggml_backend_meta_get_وېشل_state(tensor->view_src, /*assume_sync =*/ ریښتیا).axis;
                GGML_ASSERT(وېشل_dim_view_src >= 0 && وېشل_dim_view_src < GGML_MAX_DIMS);

                // The offset can be internal to the data split, in those cases the view offset should not be scaled.
                // If however, the offset is larger than the data split then it needs to be scaled proportionally.
                منطقي وېشل_دننی_دننه_واټن = t_ij->view_offs <= tensor->view_src->nb[وېشل_dim_view_src];
                لكل (صحيح i = 0; i < GGML_MAX_DIMS; i++) {
                    ثابت size_t dim_اندازه = tensor->ne[i] * tensor->nb[i];
                    إذا (tensor->view_offs <= dim_اندازه && dim_اندازه < tensor->nb[وېشل_dim]) {
                        وېشل_دننی_دننه_واټن = ریښتیا;
                        اكسر;
                    }
                }
                إذا (!وېشل_دننی_دننه_واټن) {
                    t_ij->view_offs = t_ij->view_offs * ne[وېشل_dim]/tensor->ne[وېشل_dim];
                }
            }
        }
        // TODO: revisit once the graph allocator has been refactored, see https://github.com/ggml-org/llama.cpp/pull/25051#issuecomment-4842873396
        ggml_backend_بفر_t init_buf = simple_buf;
        إذا (t_ij->view_src != تش_اشاره) {
            t_ij->معلومات = (حرف *) t_ij->view_src->معلومات + t_ij->view_offs;
            // views inherit the source slice's concrete sub-buffer (issue 22197)
            إذا (tensor->view_src != تش_اشاره && ggml_backend_بفر_is_meta(tensor->view_src->بفر)
                    && t_ij->view_src->بفر != تش_اشاره) {
                t_ij->بفر = t_ij->view_src->بفر;
                init_buf     = t_ij->view_src->بفر;
            }
        } وإلا إذا (simple_buf != تش_اشاره) {
            إذا (ggml_backend_بفر_is_multi_بفر(simple_buf)) {
                GGML_ABORT("multi buffers are not supported by the meta backend");
            }
            t_ij->معلومات = (حرف *) ggml_backend_بفر_get_بنسټ(simple_buf)
                + size_t(tensor->معلومات) - size_t(ggml_backend_بفر_get_بنسټ(tensor->بفر));
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
            } وإلا إذا (t_ij->src[i] != تش_اشاره && ggml_backend_بفر_is_meta(t_ij->src[i]->بفر)) {
                t_ij->src[i] = ggml_backend_meta_بفر_simple_tensor(tensor->src[i], j);
            }
        }

        simple_tensors.push_back(t_ij);
    }

    // If one of the sources has a zero-sized slice, disable the computation:
    لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
        إذا (tensor->src[i] == تش_اشاره || !ggml_backend_بفر_is_meta(tensor->src[i]->بفر)) {
            استمر;
        }

        ثابت ggml_backend_meta_وېشل_state وېشل_state_src = ggml_backend_meta_get_وېشل_state(tensor->src[i], /*assume_sync =*/ ریښتیا);
        إذا (وېشل_state_src.axis < 0 || وېشل_state_src.axis >= GGML_MAX_DIMS) {
            استمر;
        }
        لكل (size_t j = 0; j < n_simple_bufs; j++) {
            int64_t ne_مجموعه = 0;
            لكل (size_t s = 0; s < وېشل_state_src.n_segments; s++) {
                ne_مجموعه += وېشل_state_src.ne[s*n_simple_bufs + j] * وېشل_state_src.nr[s];
            }
            إذا (ne_مجموعه == 0) {
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
    buf_ctx->stc_compute_شاخص = buf_ctx->stc_compute_شاخص_بل;
    أرجع ggml_backend_meta_بفر_init_tensor_impl(buf_ctx->get_simple_tensor_container(tensor), tensor);
}

ساكن فراغ ggml_backend_meta_بفر_memset_tensor(
        ggml_backend_بفر_t بفر, ggml_tensor * tensor, uint8_t value, size_t دننه_واټن, size_t size) {
    ثابت size_t n_bufs = ggml_backend_meta_بفر_n_bufs(بفر);
    ثابت ggml_backend_meta_وېشل_state وېشل_state =
            ggml_backend_meta_get_وېشل_state(tensor, /*assume_sync =*/ خطأ);
    GGML_ASSERT(ggml_is_contiguous(tensor) || وېشل_state.axis == GGML_BACKEND_وېشل_AXIS_MIRRORED);

    إذا (وېشل_state.n_segments != 1 || وېشل_state.nr[0] != 1) {
        GGML_ASSERT(وېشل_state.axis >= 0 && وېشل_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(وېشل_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        std::vector<size_t> simple_offsets(n_bufs, 0);
        إذا (وېشل_state.axis == GGML_BACKEND_وېشل_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            ثابت size_t لیکه_stride = tensor->nb[1];
            GGML_ASSERT(دننه_واټن % لیکه_stride == 0);
            GGML_ASSERT(size   % لیکه_stride == 0);
            ثابت int64_t لیکه_پیل = دننه_واټن / لیکه_stride;
            ثابت int64_t لیکه_شمېر = size   / لیکه_stride;
            GGML_ASSERT(لیکه_پیل + لیکه_شمېر <= tensor->ne[1]);

            ثابت int64_t blck_اندازه = ggml_blck_اندازه(tensor->type);
            لكل (size_t s = 0; s < وېشل_state.n_segments; s++) {
                لكل (size_t r = 0; r < وېشل_state.nr[s]; r++) {
                    لكل (size_t j = 0; j < n_bufs; j++) {
                        ggml_tensor * simple_tensor = ggml_backend_meta_بفر_simple_tensor(tensor, j);
                        GGML_ASSERT(وېشل_state.ne[s*n_bufs + j] % blck_اندازه == 0);
                        ثابت size_t nbytes = وېشل_state.ne[s*n_bufs + j]/blck_اندازه * tensor->nb[0];
                        لكل (int64_t لیکه = 0; لیکه < لیکه_شمېر; لیکه++) {
                            ggml_backend_tensor_memset(simple_tensor, value,
                                    simple_offsets[j] + (لیکه_پیل + لیکه)*simple_tensor->nb[1], nbytes);
                        }
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            أرجع;
        }

        GGML_ASSERT(وېشل_state.axis == GGML_BACKEND_وېشل_AXIS_1);

        ثابت size_t لیکه_stride = tensor->nb[2];
        GGML_ASSERT(دننه_واټن % لیکه_stride == 0);
        GGML_ASSERT(size   % لیکه_stride == 0);
        ثابت int64_t لیکه_پیل = دننه_واټن / لیکه_stride;
        ثابت int64_t لیکه_شمېر = size   / لیکه_stride;
        GGML_ASSERT(لیکه_پیل + لیکه_شمېر <= tensor->ne[2]);

        لكل (size_t s = 0; s < وېشل_state.n_segments; s++) {
            لكل (size_t r = 0; r < وېشل_state.nr[s]; r++) {
                لكل (size_t j = 0; j < n_bufs; j++) {
                    ggml_tensor * simple_tensor = ggml_backend_meta_بفر_simple_tensor(tensor, j);
                    ثابت size_t nbytes = وېشل_state.ne[s*n_bufs + j] * tensor->nb[1];
                    لكل (int64_t لیکه = 0; لیکه < لیکه_شمېر; لیکه++) {
                        ggml_backend_tensor_memset(simple_tensor, value,
                                simple_offsets[j] + (لیکه_پیل + لیکه)*simple_tensor->nb[2], nbytes);
                    }
                    simple_offsets[j] += nbytes;
                }
            }
        }
        أرجع;
    }

    بدل (وېشل_state.axis) {
        حالة GGML_BACKEND_وېشل_AXIS_0:
        حالة GGML_BACKEND_وېشل_AXIS_1:
        حالة GGML_BACKEND_وېشل_AXIS_2: {
            ثابت size_t chunk_اندازه_ډک = tensor->nb[وېشل_state.axis + 1];
            GGML_ASSERT(دننه_واټن % chunk_اندازه_ډک == 0);
            GGML_ASSERT(size   % chunk_اندازه_ډک == 0);
            ثابت int64_t i_پیل =  دننه_واټن        / chunk_اندازه_ډک;
            ثابت int64_t i_stop  = (دننه_واټن + size) / chunk_اندازه_ډک;
            لكل (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_بفر_simple_tensor(tensor, j);
                ثابت size_t chunk_اندازه = simple_tensor->nb[وېشل_state.axis + 1];
                إذا (chunk_اندازه == 0) {
                    استمر;
                }
                لكل (int64_t i = i_پیل; i < i_stop; i++) {
                    ggml_backend_tensor_memset(simple_tensor, value, i*chunk_اندازه, chunk_اندازه);
                }
            }
        } اكسر;
        حالة GGML_BACKEND_وېشل_AXIS_PARTIAL: {
            GGML_ASSERT(value == 0);
            [[fallthrough]];
        }
        حالة GGML_BACKEND_وېشل_AXIS_MIRRORED: {
            لكل (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_بفر_simple_tensor(tensor, j);
                ggml_backend_tensor_memset(simple_tensor, value, دننه_واټن, size);
            }
        } اكسر;
        افتراضي: {
            GGML_ABORT("fatal error");
        }
    }
}

ساكن فراغ ggml_backend_meta_بفر_سېټ_tensor(ggml_backend_بفر_t بفر, ggml_tensor * tensor, ثابت فراغ * معلومات, size_t دننه_واټن, size_t size) {
    ثابت size_t n_bufs = ggml_backend_meta_بفر_n_bufs(بفر);
    ثابت ggml_backend_meta_وېشل_state وېشل_state = ggml_backend_meta_get_وېشل_state(tensor, /*assume_sync =*/ خطأ);
    GGML_ASSERT(ggml_is_contiguous(tensor) || وېشل_state.axis == GGML_BACKEND_وېشل_AXIS_MIRRORED);

    إذا (وېشل_state.n_segments != 1 || وېشل_state.nr[0] != 1) {
        GGML_ASSERT(وېشل_state.axis >= 0 && وېشل_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(وېشل_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        size_t دننه_واټن_معلومات = 0;
        std::vector<size_t> simple_offsets(n_bufs, 0);
        إذا (وېشل_state.axis == GGML_BACKEND_وېشل_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            ثابت size_t لیکه_stride = tensor->nb[1];
            GGML_ASSERT(دننه_واټن % لیکه_stride == 0);
            GGML_ASSERT(size   % لیکه_stride == 0);
            ثابت int64_t لیکه_پیل = دننه_واټن / لیکه_stride;
            ثابت int64_t لیکه_شمېر = size   / لیکه_stride;
            GGML_ASSERT(لیکه_پیل + لیکه_شمېر <= tensor->ne[1]);

            ثابت int64_t blck_اندازه = ggml_blck_اندازه(tensor->type);
            لكل (size_t s = 0; s < وېشل_state.n_segments; s++) {
                لكل (size_t r = 0; r < وېشل_state.nr[s]; r++) {
                    لكل (size_t j = 0; j < n_bufs; j++) {
                        ggml_tensor * simple_tensor = ggml_backend_meta_بفر_simple_tensor(tensor, j);
                        GGML_ASSERT(وېشل_state.ne[s*n_bufs + j] % blck_اندازه == 0);
                        ثابت size_t nbytes = وېشل_state.ne[s*n_bufs + j]/blck_اندازه * tensor->nb[0];
                        ggml_backend_tensor_سېټ_2d(simple_tensor, (ثابت حرف *) معلومات + دننه_واټن_معلومات,
                            simple_offsets[j] + لیکه_پیل * simple_tensor->nb[1], nbytes,
                            لیکه_شمېر, simple_tensor->nb[1], tensor->nb[1]);
                        دننه_واټن_معلومات       += nbytes;
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            GGML_ASSERT(دننه_واټن_معلومات*لیکه_شمېر == size);
            أرجع;
        }
        GGML_ASSERT(وېشل_state.axis == GGML_BACKEND_وېشل_AXIS_1);

        ثابت size_t لیکه_stride = tensor->nb[2];
        GGML_ASSERT(دننه_واټن % لیکه_stride == 0);
        GGML_ASSERT(size   % لیکه_stride == 0);
        ثابت int64_t لیکه_پیل = دننه_واټن / لیکه_stride;
        ثابت int64_t لیکه_شمېر = size   / لیکه_stride;
        GGML_ASSERT(لیکه_پیل + لیکه_شمېر <= tensor->ne[2]);

        لكل (size_t s = 0; s < وېشل_state.n_segments; s++) {
            لكل (size_t r = 0; r < وېشل_state.nr[s]; r++) {
                لكل (size_t j = 0; j < n_bufs; j++) {
                    ggml_tensor * simple_tensor = ggml_backend_meta_بفر_simple_tensor(tensor, j);
                    ثابت size_t nbytes = وېشل_state.ne[s*n_bufs + j] * tensor->nb[1];
                    ggml_backend_tensor_سېټ_2d(simple_tensor, (ثابت حرف *) معلومات + دننه_واټن_معلومات,
                        simple_offsets[j] + لیکه_پیل * simple_tensor->nb[2], nbytes,
                        لیکه_شمېر, simple_tensor->nb[2], tensor->nb[2]);
                    دننه_واټن_معلومات       += nbytes;
                    simple_offsets[j] += nbytes;
                }
            }
        }
        GGML_ASSERT(دننه_واټن_معلومات*لیکه_شمېر == size);
        أرجع;
    }

    بدل (وېشل_state.axis) {
        حالة GGML_BACKEND_وېشل_AXIS_0:
        حالة GGML_BACKEND_وېشل_AXIS_1:
        حالة GGML_BACKEND_وېشل_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            ثابت size_t chunk_اندازه_ډک = tensor->nb[وېشل_state.axis + 1];
            GGML_ASSERT(دننه_واټن % chunk_اندازه_ډک == 0);
            GGML_ASSERT(size   % chunk_اندازه_ډک == 0);
            ثابت int64_t i_پیل =  دننه_واټن        /chunk_اندازه_ډک;
            ثابت int64_t i_stop  = (دننه_واټن + size)/chunk_اندازه_ډک;
            size_t دننه_واټن_j = 0;
            لكل (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_بفر_simple_tensor(tensor, j);
                ثابت size_t chunk_اندازه_j = simple_tensor->nb[وېشل_state.axis + 1];
                إذا (chunk_اندازه_j == 0) {
                    استمر;
                }
                ثابت size_t simple_دننه_واټن = i_پیل * chunk_اندازه_j;
                ggml_backend_tensor_سېټ_2d(simple_tensor, (ثابت حرف *) معلومات + دننه_واټن_j, simple_دننه_واټن, chunk_اندازه_j, i_stop - i_پیل, chunk_اندازه_j, chunk_اندازه_ډک);
                دننه_واټن_j += chunk_اندازه_j;
            }
            GGML_ASSERT(دننه_واټن_j == chunk_اندازه_ډک);
        } اكسر;
        حالة GGML_BACKEND_وېشل_AXIS_MIRRORED: {
            لكل (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_بفر_simple_tensor(tensor, j);
                ggml_backend_tensor_سېټ(simple_tensor, معلومات, دننه_واټن, size);
            }
        } اكسر;
        حالة GGML_BACKEND_وېشل_AXIS_PARTIAL: {
            GGML_ASSERT(tensor->type == GGML_TYPE_F32);
            GGML_ASSERT(دننه_واټن % حجم(عائم) == 0);
            GGML_ASSERT(size   % حجم(عائم) == 0);
            ثابت size_t n_ارزښتونه = size / حجم(عائم);
            size_t n_contributors = 0;
            لكل (size_t j = 0; j < n_bufs; j++) {
                n_contributors += وېشل_state.ne[j] != 0;
            }
            ثابت منطقي has_contributor_ماسک = n_contributors != 0;
            إذا (!has_contributor_ماسک) {
                n_contributors = n_bufs;
            }
            std::vector<عائم> tmp(n_ارزښتونه);
            لكل (size_t i = 0; i < n_ارزښتونه; i++) {
                tmp[i] = ((ثابت عائم *) معلومات)[i] / n_contributors;
            }
            std::vector<عائم> zero;
            إذا (has_contributor_ماسک) {
                zero.resize(n_ارزښتونه, 0.0f);
            }
            لكل (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_بفر_simple_tensor(tensor, j);
                ثابت عائم * partial = has_contributor_ماسک && وېشل_state.ne[j] == 0 ? zero.معلومات() : tmp.معلومات();
                ggml_backend_tensor_سېټ(simple_tensor, partial, دننه_واټن, size);
            }
        } اكسر;
        افتراضي: {
            GGML_ABORT("fatal error");
        }
    }
}

ساكن فراغ ggml_backend_meta_بفر_get_tensor(ggml_backend_بفر_t بفر, ثابت ggml_tensor * tensor, فراغ * معلومات, size_t دننه_واټن, size_t size) {
    ثابت size_t n_bufs = ggml_backend_meta_بفر_n_bufs(بفر);
    ثابت ggml_backend_meta_وېشل_state وېشل_state = ggml_backend_meta_get_وېشل_state(tensor, /*assume_sync =*/ خطأ);
    GGML_ASSERT(ggml_is_contiguous(tensor) || وېشل_state.axis == GGML_BACKEND_وېشل_AXIS_MIRRORED);

    إذا (وېشل_state.n_segments != 1 || وېشل_state.nr[0] != 1) {
        GGML_ASSERT(وېشل_state.axis >= 0 && وېشل_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(وېشل_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        size_t دننه_واټن_معلومات = 0;
        std::vector<size_t> simple_offsets(n_bufs, 0);
        إذا (وېشل_state.axis == GGML_BACKEND_وېشل_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            ثابت size_t لیکه_stride = tensor->nb[1];
            GGML_ASSERT(دننه_واټن % لیکه_stride == 0);
            GGML_ASSERT(size   % لیکه_stride == 0);
            ثابت int64_t لیکه_پیل = دننه_واټن / لیکه_stride;
            ثابت int64_t لیکه_شمېر = size   / لیکه_stride;
            GGML_ASSERT(لیکه_پیل + لیکه_شمېر <= tensor->ne[1]);

            ثابت int64_t blck_اندازه = ggml_blck_اندازه(tensor->type);
            لكل (size_t s = 0; s < وېشل_state.n_segments; s++) {
                لكل (size_t r = 0; r < وېشل_state.nr[s]; r++) {
                    لكل (size_t j = 0; j < n_bufs; j++) {
                        ثابت ggml_tensor * simple_tensor = ggml_backend_meta_بفر_simple_tensor(tensor, j);
                        GGML_ASSERT(وېشل_state.ne[s*n_bufs + j] % blck_اندازه == 0);
                        ثابت size_t nbytes = وېشل_state.ne[s*n_bufs + j]/blck_اندازه * tensor->nb[0];
                        ggml_backend_tensor_get_2d(simple_tensor, (حرف *) معلومات + دننه_واټن_معلومات,
                            simple_offsets[j] + لیکه_پیل * simple_tensor->nb[1], nbytes,
                            لیکه_شمېر, simple_tensor->nb[1], tensor->nb[1]);
                        دننه_واټن_معلومات       += nbytes;
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            GGML_ASSERT(دننه_واټن_معلومات*لیکه_شمېر == size);
            أرجع;
        }
        GGML_ASSERT(وېشل_state.axis == GGML_BACKEND_وېشل_AXIS_1);

        ثابت size_t لیکه_stride = tensor->nb[2];
        GGML_ASSERT(دننه_واټن % لیکه_stride == 0);
        GGML_ASSERT(size   % لیکه_stride == 0);
        ثابت int64_t لیکه_پیل = دننه_واټن / لیکه_stride;
        ثابت int64_t لیکه_شمېر = size   / لیکه_stride;
        GGML_ASSERT(لیکه_پیل + لیکه_شمېر <= tensor->ne[2]);

        لكل (size_t s = 0; s < وېشل_state.n_segments; s++) {
            لكل (size_t r = 0; r < وېشل_state.nr[s]; r++) {
                لكل (size_t j = 0; j < n_bufs; j++) {
                    ثابت ggml_tensor * simple_tensor = ggml_backend_meta_بفر_simple_tensor(tensor, j);
                    ثابت size_t nbytes = وېشل_state.ne[s*n_bufs + j] * tensor->nb[1];
                    ggml_backend_tensor_get_2d(simple_tensor, (حرف *) معلومات + دننه_واټن_معلومات,
                        simple_offsets[j] + لیکه_پیل * simple_tensor->nb[2], nbytes,
                        لیکه_شمېر, simple_tensor->nb[2], tensor->nb[2]);
                    دننه_واټن_معلومات       += nbytes;
                    simple_offsets[j] += nbytes;
                }
            }
        }
        GGML_ASSERT(دننه_واټن_معلومات*لیکه_شمېر == size);
        أرجع;
    }

    بدل (وېشل_state.axis) {
        حالة GGML_BACKEND_وېشل_AXIS_0:
        حالة GGML_BACKEND_وېشل_AXIS_1:
        حالة GGML_BACKEND_وېشل_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            ثابت size_t chunk_اندازه_ډک = tensor->nb[وېشل_state.axis + 1];
            GGML_ASSERT(دننه_واټن % chunk_اندازه_ډک == 0);
            GGML_ASSERT(size   % chunk_اندازه_ډک == 0);
            ثابت int64_t i_پیل =  دننه_واټن        /chunk_اندازه_ډک;
            ثابت int64_t i_stop  = (دننه_واټن + size)/chunk_اندازه_ډک;
            size_t دننه_واټن_j = 0;
            لكل (size_t j = 0; j < n_bufs; j++){
                ثابت ggml_tensor * simple_tensor = ggml_backend_meta_بفر_simple_tensor(tensor, j);
                ثابت size_t chunk_اندازه_j = simple_tensor->nb[وېشل_state.axis + 1];
                إذا (chunk_اندازه_j == 0) {
                    استمر;
                }
                ثابت size_t simple_دننه_واټن = i_پیل * chunk_اندازه_j;
                ggml_backend_tensor_get_2d(simple_tensor, (حرف *) معلومات + دننه_واټن_j, simple_دننه_واټن, chunk_اندازه_j, i_stop - i_پیل, chunk_اندازه_j, chunk_اندازه_ډک);
                دننه_واټن_j += chunk_اندازه_j;
            }
            GGML_ASSERT(دننه_واټن_j == chunk_اندازه_ډک);
        } اكسر;
        حالة GGML_BACKEND_وېشل_AXIS_MIRRORED: {
            // TODO other simple backend may be better
            ثابت ggml_tensor * simple_tensor = ggml_backend_meta_بفر_simple_tensor(tensor, 0);
            ggml_backend_tensor_get(simple_tensor, معلومات, دننه_واټن, size);
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
    /* .get_base        = */ ggml_backend_meta_بفر_get_بنسټ,
    /* .init_tensor     = */ ggml_backend_meta_بفر_init_tensor,
    /* .memset_tensor   = */ ggml_backend_meta_بفر_memset_tensor,
    /* .set_tensor      = */ ggml_backend_meta_بفر_سېټ_tensor,
    /* .get_tensor      = */ ggml_backend_meta_بفر_get_tensor,
    /* .set_tensor_2d   = */ تش_اشاره,
    /* .get_tensor_2d   = */ تش_اشاره,
    /* .cpy_tensor      = */ تش_اشاره,
    /* .clear           = */ ggml_backend_meta_بفر_clear,
    /* .reset           = */ ggml_backend_meta_بفر_reset,
};

منطقي ggml_backend_بفر_is_meta(ggml_backend_بفر_t buf) {
    أرجع buf != تش_اشاره && buf->iface.free_بفر == ggml_backend_meta_بفر_iface.free_بفر;
}

فراغ ggml_backend_meta_بفر_سېټ_usage(ggml_backend_بفر_t بفر, تعداد ggml_backend_بفر_usage usage) {
    GGML_ASSERT(ggml_backend_بفر_is_meta(بفر));
    ggml_backend_meta_بفر_context * buf_ctx = (ggml_backend_meta_بفر_context *) بفر->context;
    لكل (size_t i = 0; i < buf_ctx->bufs.size(); i++) {
        إذا (buf_ctx->bufs[i]) {
            ggml_backend_بفر_سېټ_usage(buf_ctx->bufs[i].get(), usage);
        }
    }
}

ساكن ggml_backend_بفر_t ggml_backend_meta_بفر_type_alloc_بفر(ggml_backend_بفر_type_t buft, size_t size) {
    ثابت size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);

    ثابت ggml_init_params params = {
        /*.mem_size   =*/ 1024*1024*ggml_tensor_overhead(), // FIXME
        /*.mem_buffer =*/ تش_اشاره,
        /*.no_alloc   =*/ ریښتیا,
    };
    ggml_backend_meta_simple_tensor_container stc_static;
    ggml_backend_meta_simple_tensor_container stc_compute_0(params, n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_1(params, n_simple_bufts);

    size_t max_size = 0;
    std::vector<ggml_backend_بفر_t> bufs;
    bufs.reserve(n_simple_bufts);
    لكل (size_t i = 0; i < n_simple_bufts; i++) {
        bufs.push_back(ggml_backend_buft_alloc_بفر(ggml_backend_meta_buft_simple_buft(buft, i), size));
        GGML_ASSERT(bufs.back() != تش_اشاره);
        max_size = std::max(max_size, ggml_backend_بفر_get_اندازه(bufs.back()));
    }
    ggml_backend_meta_بفر_context * buf_ctx = جديد ggml_backend_meta_بفر_context(stc_static, stc_compute_0, stc_compute_1, bufs);

    أرجع ggml_backend_بفر_init(buft, ggml_backend_meta_بفر_iface, buf_ctx, max_size);
}

ساكن ggml_backend_بفر_t ggml_backend_meta_بفر_type_alloc_بفر_n(ggml_backend_بفر_type_t buft, ggml_tensor ** tensors, صحيح n_tensors) {
    ثابت size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);

    constexpr size_t compute_headroom = 16; // Maximum number of views per statically allocated tensor that can be created between evals.
    ثابت ggml_init_params params_static = {
        /*.mem_size   =*/ n_tensors * ggml_tensor_overhead(),
        /*.mem_buffer =*/ تش_اشاره,
        /*.no_alloc   =*/ ریښتیا,
    };
    ثابت ggml_init_params params_compute = {
        /*.mem_size   =*/ compute_headroom * n_tensors * ggml_tensor_overhead(),
        /*.mem_buffer =*/ تش_اشاره,
        /*.no_alloc   =*/ ریښتیا,
    };
    ggml_backend_meta_simple_tensor_container stc_static   (params_static,  n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_0(params_compute, n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_1(params_compute, n_simple_bufts);

    std::vector<ggml_backend_بفر_t> bufs(n_simple_bufts, تش_اشاره);
    ggml_backend_meta_بفر_context * meta_buf_ctx = جديد ggml_backend_meta_بفر_context(stc_static, stc_compute_0, stc_compute_1, bufs);

    ggml_backend_بفر_t meta_buf = ggml_backend_بفر_init(buft, ggml_backend_meta_بفر_iface, meta_buf_ctx, 0);
    لكل (صحيح i = 0; i < n_tensors; i++) {
        ggml_tensor * t = tensors[i];
        t->بفر = meta_buf;
        ggml_backend_meta_بفر_init_tensor_impl(meta_buf_ctx->stc_static, t);
        t->معلومات = (فراغ *) 0x2000000000000000; // FIXME
    }
    لكل (size_t i = 0; i < n_simple_bufts; i++) {
        ggml_context * ctx = meta_buf_ctx->stc_static.ctxs[i].get();
        ggml_backend_بفر_type_t simple_buft = ggml_backend_meta_buft_simple_buft(buft, i);

        // If a ggml_context only has zero-sized tensors, ggml_backend_alloc_ctx_tensors_from_buft returns NULL.
        // For those edge cases, allocate a dummy buffer instead.
        منطقي any_nonzero_ټوټه = خطأ;
        لكل (ggml_tensor * t = ggml_get_لومړی_tensor(ctx); t != تش_اشاره; t = ggml_get_بل_tensor(ctx, t)) {
            إذا (ggml_nelements(t) != 0) {
                any_nonzero_ټوټه = ریښتیا;
                اكسر;
            }
        }
        إذا (any_nonzero_ټوټه) {
            meta_buf_ctx->bufs[i].reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx, simple_buft));
        } وإلا {
            meta_buf_ctx->bufs[i].reset(ggml_backend_buft_alloc_بفر(simple_buft, 0));
            لكل (ggml_tensor * t = ggml_get_لومړی_tensor(ctx); t != تش_اشاره; t = ggml_get_بل_tensor(ctx, t)) {
                t->بفر = meta_buf_ctx->bufs[i].get();
            }
        }
        GGML_ASSERT(meta_buf_ctx->bufs[i]);
        meta_buf->size = std::max(meta_buf->size, ggml_backend_بفر_get_اندازه(meta_buf_ctx->bufs[i].get()));
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
        ggml_cgraph * cgraph_main = تش_اشاره;
        صحيح           دننه_واټن      = 0; // Node offset vs. original graph

        std::vector<ggml_cgraph *> cgraphs_aux;
    };
    بنية backend_config {
        ggml_backend_t backend;

        std::vector<cgraph_config>           cgraphs;
        std::vector<ggml_tensor *>           nodes;
        std::vector<ggml_backend_بفر_ptr> bufs;

        backend_config(ggml_backend_t backend, ثابت size_t n_reduce_ګامونه) : backend(backend) {
            bufs.resize(n_reduce_ګامونه);
        }
    };
    std::string                 name;
    std::vector<backend_config> backend_configs;
    ggml_context_ptr            ctx;
    std::vector<ggml_cgraph *>  cgraphs_aux;
    std::vector<ggml_tensor *>  nodes_aux;
    size_t                      n_reduce_ګامونه;
    صحيح                         max_nnodes    = 0;
    size_t                      max_tmp_اندازه  = 0;
    size_t                      max_subgraphs = 0;
    size_t                      n_subgraphs   = 0;
    uint64_t                    uid           = 0;

    فراغ *                               comm_ctx       = تش_اشاره;
    ggml_backend_comm_allreduce_tensor_t comm_allreduce = تش_اشاره;

    ggml_backend_meta_context(ggml_backend_dev_t meta_dev, ثابت حرف * params) {
        ثابت size_t n_devs = ggml_backend_meta_dev_n_devs(meta_dev);
        n_reduce_ګامونه = std::ceil(std::log2(n_devs));
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
            backend_configs.emplace_back(simple_backends.back(), n_reduce_ګامونه);
        }
        name += ")";

        إذا (n_devs > 1) {
            ggml_backend_comm_init_t comm_init = (ggml_backend_comm_init_t) ggml_backend_reg_get_proc_پته(
                ggml_backend_dev_backend_reg(ggml_backend_get_device(simple_backends[0])), "ggml_backend_comm_init");
            إذا (comm_init != تش_اشاره) {
                comm_ctx = comm_init(simple_backends.معلومات(), simple_backends.size());
            }
        }
        إذا (comm_ctx != تش_اشاره) {
            comm_allreduce = (ggml_backend_comm_allreduce_tensor_t)
                ggml_backend_reg_get_proc_پته(ggml_backend_dev_backend_reg(
                    ggml_backend_get_device(simple_backends[0])), "ggml_backend_comm_allreduce_tensor");
            GGML_ASSERT(comm_allreduce != تش_اشاره);
        }
    }

    ~ggml_backend_meta_context() {
        إذا (comm_ctx != تش_اشاره) {
            ggml_backend_comm_free_t comm_free = (ggml_backend_comm_free_t) ggml_backend_reg_get_proc_پته(
                ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_configs[0].backend)), "ggml_backend_comm_free");
            GGML_ASSERT(comm_free != تش_اشاره);
            comm_free(comm_ctx);
        }
        لكل (خپلکاری & bc : backend_configs) {
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
    ړنګول backend_ctx;
    ړنګول backend;
}

ساكن فراغ ggml_backend_meta_سېټ_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, ثابت فراغ * معلومات, size_t دننه_واټن, size_t size) {
    ثابت size_t n_backends = ggml_backend_meta_n_backends(backend);
    GGML_ASSERT(دننه_واټن == 0);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    ثابت ggml_backend_meta_وېشل_state وېشل_state = ggml_backend_meta_get_وېشل_state(tensor, /*assume_sync =*/ خطأ);
    GGML_ASSERT(وېشل_state.n_segments == 1);
    GGML_ASSERT(وېشل_state.nr[0]      == 1);

    بدل (وېشل_state.axis) {
        حالة GGML_BACKEND_وېشل_AXIS_0:
        حالة GGML_BACKEND_وېشل_AXIS_1:
        حالة GGML_BACKEND_وېشل_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            ثابت size_t chunk_اندازه_ډک = tensor->nb[وېشل_state.axis + 1];
            GGML_ASSERT(دننه_واټن % chunk_اندازه_ډک == 0);
            GGML_ASSERT(size   % chunk_اندازه_ډک == 0);
            ثابت int64_t i_پیل =  دننه_واټن        /chunk_اندازه_ډک;
            ثابت int64_t i_stop  = (دننه_واټن + size)/chunk_اندازه_ډک;
            size_t دننه_واټن_j = 0;
            لكل (size_t j = 0; j < n_backends; j++){
                ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
                ggml_tensor * simple_tensor = ggml_backend_meta_بفر_simple_tensor(tensor, j);
                ثابت size_t chunk_اندازه_j = simple_tensor->nb[وېشل_state.axis + 1];
                إذا (chunk_اندازه_j == 0) {
                    استمر;
                }
                ggml_backend_tensor_سېټ_2d_async(simple_backend, simple_tensor, (ثابت حرف *) معلومات + دننه_واټن_j, دننه_واټن, chunk_اندازه_j,
                    i_stop - i_پیل, chunk_اندازه_j, chunk_اندازه_ډک);
                دننه_واټن_j += chunk_اندازه_j;
            }
            GGML_ASSERT(دننه_واټن_j == chunk_اندازه_ډک);
        } اكسر;
        حالة GGML_BACKEND_وېشل_AXIS_MIRRORED: {
            لكل (size_t j = 0; j < n_backends; j++) {
                ggml_backend_tensor_سېټ_async(
                    ggml_backend_meta_simple_backend(backend, j), ggml_backend_meta_بفر_simple_tensor(tensor, j), معلومات, دننه_واټن, size);
            }
        } اكسر;
        افتراضي: {
            GGML_ABORT("fatal error");
        }
    }
}

ساكن فراغ ggml_backend_meta_get_tensor_async(ggml_backend_t backend, ثابت ggml_tensor * tensor, فراغ * معلومات, size_t دننه_واټن, size_t size) {
    ثابت size_t n_backends = ggml_backend_meta_n_backends(backend);
    GGML_ASSERT(دننه_واټن == 0);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    ثابت ggml_backend_meta_وېشل_state وېشل_state = ggml_backend_meta_get_وېشل_state(tensor, /*assume_sync =*/ خطأ);
    GGML_ASSERT(وېشل_state.n_segments == 1);
    GGML_ASSERT(وېشل_state.nr[0]      == 1);

    بدل (وېشل_state.axis) {
        حالة GGML_BACKEND_وېشل_AXIS_0:
        حالة GGML_BACKEND_وېشل_AXIS_1:
        حالة GGML_BACKEND_وېشل_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            ثابت size_t chunk_اندازه_ډک = tensor->nb[وېشل_state.axis + 1];
            GGML_ASSERT(دننه_واټن % chunk_اندازه_ډک == 0);
            GGML_ASSERT(size   % chunk_اندازه_ډک == 0);
            ثابت int64_t i_پیل =  دننه_واټن        /chunk_اندازه_ډک;
            ثابت int64_t i_stop  = (دننه_واټن + size)/chunk_اندازه_ډک;
            size_t دننه_واټن_j = 0;
            لكل (size_t j = 0; j < n_backends; j++){
                ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
                ثابت ggml_tensor * simple_tensor = ggml_backend_meta_بفر_simple_tensor(tensor, j);
                ثابت size_t chunk_اندازه_j = simple_tensor->nb[وېشل_state.axis + 1];
                إذا (chunk_اندازه_j == 0) {
                    استمر;
                }
                ggml_backend_tensor_get_2d_async(simple_backend, simple_tensor, (حرف *) معلومات + دننه_واټن_j, دننه_واټن, chunk_اندازه_j,
                    i_stop - i_پیل, chunk_اندازه_j, chunk_اندازه_ډک);
                دننه_واټن_j += chunk_اندازه_j;
            }
            GGML_ASSERT(دننه_واټن_j == chunk_اندازه_ډک);
        } اكسر;
        حالة GGML_BACKEND_وېشل_AXIS_MIRRORED: {
            // TODO other simple backend may be better
            ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, 0);
            ثابت ggml_tensor * simple_tensor = ggml_backend_meta_بفر_simple_tensor(tensor, 0);
            ggml_backend_tensor_get_async(simple_backend, simple_tensor, معلومات, دننه_واټن, size);
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

ساكن تعداد ggml_حالت ggml_backend_meta_ګراف_compute(ggml_backend_t backend, بنية ggml_cgraph * cgraph) {
    GGML_ASSERT(cgraph->grads == تش_اشاره);
    ثابت size_t n_backends = ggml_backend_meta_n_backends(backend);
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;

    // If the previous cgraph had a defined UID it can be used to skip rebuilding the subgraphs per simple backend.
    ثابت منطقي needs_rebuild = (cgraph->uid == 0) || (cgraph->uid != backend_ctx->uid);

    منطقي max_nnodes_raised = خطأ;
    إذا (cgraph->n_nodes > backend_ctx->max_nnodes) {
        لكل (size_t j = 0; j < n_backends; j++) {
            خپلکاری & bcj = backend_ctx->backend_configs[j];
            bcj.nodes.resize(cgraph->n_nodes);
            bcj.cgraphs.resize(cgraph->n_nodes);
        }
        backend_ctx->max_nnodes = cgraph->n_nodes;
        max_nnodes_raised = ریښتیا;
        أكد(needs_rebuild);
    }

    إذا (needs_rebuild) {
        std::set<ggml_backend_بفر_t> کارول_شوی_buffers;
        لكل (صحيح i = 0; i < cgraph->n_leafs; i++) {
            إذا (ggml_backend_بفر_is_meta(cgraph->leafs[i]->بفر)) {
                کارول_شوی_buffers.emplace(cgraph->leafs[i]->بفر);
            }
        }
        لكل (صحيح i = 0; i < cgraph->n_nodes; i++) {
            إذا (ggml_backend_بفر_is_meta(cgraph->nodes[i]->بفر)) {
                کارول_شوی_buffers.emplace(cgraph->nodes[i]->بفر);
            }
        }
        لكل (ggml_backend_بفر_t buf : کارول_شوی_buffers) {
            ggml_backend_meta_بفر_context * buf_ctx = (ggml_backend_meta_بفر_context *) buf->context;
            buf_ctx->stc_compute_شاخص_بل = buf_ctx->stc_compute_شاخص ^ 1;
            ggml_backend_meta_simple_tensor_container & stc = buf_ctx->stc_compute[buf_ctx->stc_compute_شاخص_بل];
            لكل (ggml_context_ptr & ctx : stc.ctxs) {
                ggml_reset(ctx.get());
            }
            stc.simple_tensors.clear();
        }
        size_t n_subgraphs  = 0;
        size_t max_tmp_اندازه = 0;

        لكل (size_t j = 0; j < n_backends; j++) {
            خپلکاری & bcj = backend_ctx->backend_configs[j];

            لكل (صحيح i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * غوټه = cgraph->nodes[i];
                إذا (!ggml_backend_بفر_is_meta(غوټه->بفر)) {
                    // [TAG_META_HOST_VIEWS]
                    GGML_ASSERT(ggml_backend_meta_is_host_view(غوټه));

                    // keep the node as is, mapping it to a simple tensor is not possible
                    bcj.nodes[i] = غوټه;
                    استمر;
                }
                bcj.nodes[i] = ggml_backend_meta_بفر_simple_tensor(غوټه, j);
                GGML_ASSERT(bcj.nodes[i]);
            }
        }

        {
            // For MoE models it may make sense to delay the AllReduce in order to reduce I/O:
            خپلکاری get_i_delayed_branch = [&](ثابت صحيح i) -> صحيح {
                صحيح id = i; // i_delayed
                صحيح idr = i; // i_delayed return, last safe return value

                ggml_tensor * غوټه = cgraph->nodes[id];
                int32_t n_کارول_شوی = ggml_غوټه_get_use_شمېر(cgraph, id);

                // Skip MIRRORED nodes that don't consume node
                خپلکاری skip_unrelated = [&]() {
                    طالما (id + 1 < cgraph->n_nodes) {
                        ggml_tensor * بل = cgraph->nodes[id+1];
                        إذا (ggml_backend_meta_get_وېشل_state(بل, خطأ).axis != GGML_BACKEND_وېشل_AXIS_MIRRORED) {
                            اكسر;
                        }
                        منطقي خوندي = ریښتیا;
                        لكل (صحيح s = 0; s < GGML_MAX_SRC; s++) {
                            إذا (بل->src[s] == تش_اشاره) {
                                استمر;
                            }
                            إذا (بل->src[s] == غوټه) {
                                خوندي = خطأ;
                                اكسر;
                            }
                            إذا (ggml_backend_meta_get_وېشل_state(بل->src[s], خطأ).axis != GGML_BACKEND_وېشل_AXIS_MIRRORED) {
                                خوندي = خطأ;
                                اكسر;
                            }
                        }
                        إذا (!خوندي) {
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
                    ggml_tensor * بل = cgraph->nodes[id+1];
                    إذا (بل->op == GGML_OP_جمع_کړه_ID && بل->src[0] == غوټه &&
                            ggml_backend_meta_get_وېشل_state(بل->src[1], خطأ).axis == GGML_BACKEND_وېشل_AXIS_PARTIAL &&
                            ggml_backend_meta_get_وېشل_state(بل->src[2], خطأ).axis == GGML_BACKEND_وېشل_AXIS_MIRRORED) {
                        غوټه = بل;
                        id++;
                        idr = id;
                        n_کارول_شوی = ggml_غوټه_get_use_شمېر(cgraph, id);
                    }
                }
                // Chain of MULs with MIRRORED src[1]
                طالما (ریښتیا) {
                    skip_unrelated();
                    إذا (id + 1 >= cgraph->n_nodes) {
                        أرجع idr;
                    }
                    ggml_tensor * بل = cgraph->nodes[id+1];
                    إذا (بل->op == GGML_OP_MUL && بل->src[0] == غوټه &&
                            ggml_backend_meta_get_وېشل_state(بل->src[1], خطأ).axis == GGML_BACKEND_وېشل_AXIS_MIRRORED) {
                        غوټه = بل;
                        id++;
                        idr = id;
                        n_کارول_شوی = ggml_غوټه_get_use_شمېر(cgraph, id);
                    } وإلا {
                        اكسر;
                    }
                }

                إذا (n_کارول_شوی != غوټه->ne[1] || id + 2*n_کارول_شوی-1 >= cgraph->n_nodes) {
                    أرجع idr;
                }
                لكل (int32_t k = 0; k < n_کارول_شوی; k++) {
                    ggml_tensor * بل = cgraph->nodes[id+1];
                    إذا (بل->op != GGML_OP_VIEW || بل->view_src != غوټه || بل->view_offs != k*غوټه->nb[1] ||
                            بل->ne[0] != غوټه->ne[0] || بل->ne[1] != غوټه->ne[2] || بل->nb[1] != غوټه->nb[2] ||
                            ggml_غوټه_get_use_شمېر(cgraph, id+1) != 1) {
                        أرجع idr;
                    }
                    id++;
                }
                {
                    ggml_tensor * بل = cgraph->nodes[id+1];
                    إذا (بل->op != GGML_OP_جمع_کړه || بل->src[0] != cgraph->nodes[id - (n_کارول_شوی-1)] ||
                            بل->src[1] != cgraph->nodes[id - (n_کارول_شوی-2)] || ggml_غوټه_get_use_شمېر(cgraph, id+1) != 1) {
                        أرجع idr;
                    }
                    id++;
                }
                لكل (int32_t k = 0; k < n_کارول_شوی - 2; k++) {
                    ggml_tensor * بل = cgraph->nodes[id+1];
                    إذا (بل->op != GGML_OP_جمع_کړه || بل->src[0] != cgraph->nodes[id] ||
                            بل->src[1] != cgraph->nodes[id - (n_کارول_شوی-2)] || ggml_غوټه_get_use_شمېر(cgraph, id+1) != 1) {
                        أرجع idr;
                    }
                    id++;
                }
                idr = id;
                أرجع idr;
            };

            // AllReduce(a) + AllReduce(b) == AllReduce(a + b) for independent partial branches.
            خپلکاری get_i_delayed = [&](ثابت صحيح i) -> صحيح {
                ثابت صحيح i_delayed = get_i_delayed_branch(i);
                ggml_tensor * غوټه = cgraph->nodes[i_delayed];

                إذا (ggml_غوټه_get_use_شمېر(cgraph, i_delayed) != 1) {
                    أرجع i_delayed;
                }

                لكل (صحيح id = i_delayed + 1; id < cgraph->n_nodes; id++) {
                    ggml_tensor * بل = cgraph->nodes[id];
                    إذا (بل->view_src == غوټه) {
                        أرجع i_delayed;
                    }
                    لكل (صحيح s = 0; s < GGML_MAX_SRC; s++) {
                        إذا (بل->src[s] == غوټه) {
                            أرجع i_delayed;
                        }
                    }

                    إذا (بل->view_src != تش_اشاره && بل->view_src->op == GGML_OP_NONE && ggml_backend_بفر_is_host(بل->view_src->بفر)) {
                        استمر;
                    }
                    إذا (ggml_backend_meta_get_وېشل_state(بل, خطأ).axis != GGML_BACKEND_وېشل_AXIS_PARTIAL) {
                        استمر;
                    }

                    ثابت صحيح i_other = id;
                    ثابت صحيح i_other_delayed = get_i_delayed_branch(i_other);
                    ggml_tensor * other = cgraph->nodes[i_other_delayed];
                    إذا (ggml_غوټه_get_use_شمېر(cgraph, i_other_delayed) != 1 || i_other_delayed + 1 >= cgraph->n_nodes) {
                        أرجع i_delayed;
                    }

                    ggml_tensor * مجموعه = cgraph->nodes[i_other_delayed + 1];
                    إذا (مجموعه->op != GGML_OP_جمع_کړه ||
                            !ggml_are_same_shape(غوټه, other) || غوټه->type != other->type || مجموعه->type != غوټه->type ||
                            !((مجموعه->src[0] == غوټه && مجموعه->src[1] == other) ||
                              (مجموعه->src[0] == other && مجموعه->src[1] == غوټه)) ||
                            ggml_backend_meta_get_وېشل_state(مجموعه, خطأ).axis != GGML_BACKEND_وېشل_AXIS_MIRRORED) {
                        أرجع i_delayed;
                    }

                    لكل (size_t j = 0; j < n_backends; j++) {
                        خپلکاری & bcj = backend_ctx->backend_configs[j];
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

            صحيح i_پیل = 0;
            لكل (صحيح i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * غوټه = cgraph->nodes[i];
                إذا (غوټه->view_src != تش_اشاره && غوټه->view_src->op == GGML_OP_NONE && ggml_backend_بفر_is_host(غوټه->view_src->بفر)) {
                    استمر;
                }
                ثابت ggml_backend_meta_وېشل_state وېشل_state = ggml_backend_meta_get_وېشل_state(غوټه, /*assume_sync =*/ خطأ);
                إذا (وېشل_state.axis == GGML_BACKEND_وېشل_AXIS_PARTIAL) {
                    max_tmp_اندازه = std::max(max_tmp_اندازه, ggml_nbytes(غوټه));
                }
                ثابت منطقي new_subgraph = i + 1 == cgraph->n_nodes || وېشل_state.axis == GGML_BACKEND_وېشل_AXIS_PARTIAL;
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
                        خپلکاری & bcj = backend_ctx->backend_configs[j];
                        إذا ((bcj.nodes[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                            لكل (صحيح ii = i + 1; ii <= i_delayed; ii++) {
                                bcj.nodes[ii]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
                            }
                        }
                    }
                }

                i = i_delayed;

                لكل (size_t j = 0; j < n_backends; j++) {
                    خپلکاری & bcj = backend_ctx->backend_configs[j];
                    bcj.cgraphs[n_subgraphs].دننه_واټن = i_پیل;
                }
                n_subgraphs++;
                i_پیل = i + 1;
            }
            GGML_ASSERT(i_پیل == cgraph->n_nodes);
        }

        backend_ctx->uid         = cgraph->uid;
        backend_ctx->n_subgraphs = n_subgraphs;

        إذا (max_tmp_اندازه > backend_ctx->max_tmp_اندازه) {
            لكل (size_t j = 0; j < n_backends; j++) {
                خپلکاری & bcj = backend_ctx->backend_configs[j];
                لكل (size_t i = 0; i < backend_ctx->n_reduce_ګامونه; i++) {
                    bcj.bufs[i].reset(ggml_backend_alloc_بفر(bcj.backend, max_tmp_اندازه));
                }
            }
            backend_ctx->max_tmp_اندازه = max_tmp_اندازه;
        }

        إذا (max_nnodes_raised || n_subgraphs > backend_ctx->max_subgraphs) {
            backend_ctx->max_subgraphs = std::max(backend_ctx->max_subgraphs, n_subgraphs);
            ثابت size_t n_nodes_فی_device = 3 * backend_ctx->n_reduce_ګامونه; // tmp + ADD (+zeroing) graph per step and device
            ثابت size_t n_cgraphs_فی_device = 2 * backend_ctx->n_reduce_ګامونه; // ADD ( + zeroing) graph per step and device
            ثابت size_t mem_فی_device_graphs_main = backend_ctx->max_subgraphs*ggml_ګراف_overhead_custom(backend_ctx->max_nnodes, cgraph->grads);
            ثابت size_t mem_فی_device_graphs_aux = n_cgraphs_فی_device*backend_ctx->max_subgraphs*ggml_ګراف_overhead_custom(1, cgraph->grads);
            ثابت size_t mem_فی_device_nodes_aux = n_nodes_فی_device*backend_ctx->max_subgraphs*ggml_tensor_overhead();
            ثابت ggml_init_params params = {
                /*.mem_size   =*/ n_backends * (mem_فی_device_graphs_main + mem_فی_device_graphs_aux + mem_فی_device_nodes_aux),
                /*.mem_buffer =*/ تش_اشاره,
                /*.no_alloc   =*/ ریښتیا,
            };
            backend_ctx->ctx.reset(ggml_init(params));
            لكل (size_t j = 0; j < n_backends; j++) {
                خپلکاری & bcj = backend_ctx->backend_configs[j];
                لكل (size_t i = 0; i < backend_ctx->max_subgraphs; i++) {
                    bcj.cgraphs[i].cgraph_main = ggml_new_ګراف_custom(backend_ctx->ctx.get(), backend_ctx->max_nnodes, /*grads =*/ خطأ);
                }
            }
            backend_ctx->cgraphs_aux.resize(n_backends*n_cgraphs_فی_device*backend_ctx->max_subgraphs);
            لكل (size_t k = 0; k < backend_ctx->cgraphs_aux.size(); k++) {
                backend_ctx->cgraphs_aux[k] = ggml_new_ګراف_custom(backend_ctx->ctx.get(), 1, cgraph->grads);
            }
            backend_ctx->nodes_aux.resize(n_backends*n_nodes_فی_device*backend_ctx->max_subgraphs);
            لكل (size_t k = 0; k < backend_ctx->nodes_aux.size(); k++) {
                backend_ctx->nodes_aux[k] = ggml_new_tensor_1d(backend_ctx->ctx.get(), GGML_TYPE_F32, 1);
            }
        }

        لكل (size_t j = 0; j < n_backends; j++) {
            خپلکاری & bcj = backend_ctx->backend_configs[j];
            لكل (size_t i_ګراف = 0; i_ګراف < n_subgraphs; i_ګراف++) {
                ggml_cgraph * cgraph_ij = bcj.cgraphs[i_ګراف].cgraph_main;
                ثابت size_t i_غوټه_پیل = bcj.cgraphs[i_ګراف].دننه_واټن;
                ثابت size_t i_غوټه_stop = i_ګراف + 1 < n_subgraphs ? bcj.cgraphs[i_ګراف + 1].دننه_واټن : cgraph->n_nodes;
                cgraph_ij->n_nodes = i_غوټه_stop - i_غوټه_پیل;
                ggml_hash_سېټ_reset(&cgraph_ij->لیدل_شوی_hash_سېټ);
                لكل (size_t i_غوټه = i_غوټه_پیل; i_غوټه < i_غوټه_stop; i_غوټه++) {
                    ggml_tensor * غوټه_ij = bcj.nodes[i_غوټه];
                    cgraph_ij->nodes[i_غوټه - i_غوټه_پیل] = غوټه_ij;
                    ثابت size_t hash_pos_orig = ggml_hash_پیدا_کړه(&cgraph->لیدل_شوی_hash_سېټ, cgraph->nodes[i_غوټه]);
                    ثابت size_t hash_pos_ij = ggml_hash_ورننباسه(&cgraph_ij->لیدل_شوی_hash_سېټ, غوټه_ij);
                    cgraph_ij->use_counts[hash_pos_ij] = cgraph->use_counts[hash_pos_orig];
                }
                cgraph_ij->uid = ggml_ګراف_بل_uid();
            }
        }
    }

    size_t iga = 0; // i graph aux
    size_t ina = 0; // i node aux

    خپلکاری get_غوټه_aux = [&](ggml_tensor * t) -> ggml_tensor * {
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
    خپلکاری سېټ_tmp_معلومات = [&](ggml_tensor * tensor, ثابت size_t j, ثابت size_t i_buf) {
        خپلکاری & bcj = backend_ctx->backend_configs[j];
        ggml_backend_بفر_ptr & buf_ptr = bcj.bufs[i_buf];
        إذا (!buf_ptr || ggml_backend_بفر_get_اندازه(buf_ptr.get()) < backend_ctx->max_tmp_اندازه) {
            buf_ptr.reset(ggml_backend_alloc_بفر(bcj.backend, backend_ctx->max_tmp_اندازه));
        }
        tensor->بفر = buf_ptr.get();
        tensor->معلومات   = ggml_backend_بفر_get_بنسټ(buf_ptr.get());
    };
    // FIXME usage_counts
    خپلکاری get_cgraph_aux = [&]() -> ggml_cgraph * {
        ggml_cgraph * ret = backend_ctx->cgraphs_aux[iga++];
        أرجع ret;
    };

    // Preferentially use backend-specific allreduce_tensor_async (e.g. NCCL for CUDA), use a generic fallback if unavailable:
    خپلکاری allreduce_fallback = [&](size_t i) -> ggml_حالت {
        std::vector<ggml_cgraph *> ګام_cgraphs(n_backends, تش_اشاره);

        // Zero out nodes that were disabled due to having a zero-sized slice:
        لكل (size_t j = 0; j < n_backends; j++) {
            خپلکاری & bcj = backend_ctx->backend_configs[j];
            ggml_tensor * غوټه = bcj.cgraphs[i].cgraph_main->nodes[bcj.cgraphs[i].cgraph_main->n_nodes - 1];
            إذا (غوټه->flags & GGML_TENSOR_FLAG_COMPUTE) {
                استمر;
            }
            إذا (ggml_nelements(غوټه) > 0) {
                ggml_tensor * غوټه_zero = get_غوټه_aux(غوټه);
                غوټه_zero->op = GGML_OP_FILL;
                غوټه_zero->src[0] = غوټه; // only used for the shape, the data is not read
                ggml_سېټ_op_params_f32(غوټه_zero, 0, 0.0f);
                غوټه_zero->معلومات = غوټه->معلومات;
                غوټه_zero->بفر = غوټه->بفر;
                غوټه_zero->flags |= GGML_TENSOR_FLAG_COMPUTE;

                ګام_cgraphs[j] = get_cgraph_aux();
                ګام_cgraphs[j]->nodes[0] = غوټه_zero;
                ګام_cgraphs[j]->n_nodes = 1;
                ثابت ggml_حالت حالت = ggml_backend_ګراف_compute_async(bcj.backend, ګام_cgraphs[j]);
                إذا (حالت != GGML_حالت_SUCCESS) {
                    أرجع حالت;
                }
            }
        }
        std::fill(ګام_cgraphs.begin(), ګام_cgraphs.end(), تش_اشاره);

        خپلکاری ټېل_معلومات = [&](ثابت size_t j_src, ثابت size_t j_dst, ثابت size_t i_buf) {
            أكد(ګام_cgraphs[j_dst] == تش_اشاره);
            خپلکاری & bcj_src = backend_ctx->backend_configs[j_src];
            خپلکاری & bcj_dst = backend_ctx->backend_configs[j_dst];

            ggml_tensor * غوټه_src = bcj_src.cgraphs[i].cgraph_main->nodes[bcj_src.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_tensor * غوټه_dst = bcj_dst.cgraphs[i].cgraph_main->nodes[bcj_dst.cgraphs[i].cgraph_main->n_nodes - 1];
            GGML_ASSERT(ggml_is_contiguous(غوټه_src));
            GGML_ASSERT(ggml_is_contiguous(غوټه_dst));

            ggml_tensor * غوټه_tmp = get_غوټه_aux(غوټه_dst);
            سېټ_tmp_معلومات(غوټه_tmp, j_dst, i_buf);

            ggml_backend_tensor_copy_async(bcj_src.backend, bcj_dst.backend, غوټه_src, غوټه_tmp);

            ggml_tensor * غوټه_red = get_غوټه_aux(غوټه_dst);
            غوټه_red->view_src = غوټه_dst->view_src == تش_اشاره ? غوټه_dst : غوټه_dst->view_src;
            غوټه_red->view_offs = غوټه_dst->view_offs;
            غوټه_red->op = GGML_OP_جمع_کړه;
            غوټه_red->src[0] = غوټه_dst;
            غوټه_red->src[1] = غوټه_tmp;
            غوټه_red->flags |= GGML_TENSOR_FLAG_COMPUTE;
            ggml_backend_view_init(غوټه_red);

            ggml_cgraph * cgraph_aux = get_cgraph_aux();
            cgraph_aux->nodes[0] = غوټه_red;
            cgraph_aux->n_nodes = 1;
            ګام_cgraphs[j_dst] = cgraph_aux;
        };

        size_t دننه_واټن_j = n_backends/2;
        طالما ((دننه_واټن_j & (دننه_واټن_j - 1)) != 0) {
            دننه_واټن_j--;
        }
        ثابت size_t دننه_واټن_j_max = دننه_واټن_j;
        size_t i_buf = 0;

        // If n_backends is not a power of 2, fold in the excess prior to butterfly reduction:
        لكل (size_t j_src = 2*دننه_واټن_j_max; j_src < n_backends; j_src++) {
            ثابت size_t j_dst = j_src - 2*دننه_واټن_j_max;
            ټېل_معلومات(j_src, j_dst, i_buf);
            ثابت ggml_حالت حالت = ggml_backend_ګراف_compute_async(backend_ctx->backend_configs[j_dst].backend, ګام_cgraphs[j_dst]);
            إذا (حالت != GGML_حالت_SUCCESS) {
                أرجع حالت;
            }
            i_buf = 1;
        }

        // Butterfly reduction:
        لكل (; دننه_واټن_j >= 1; دننه_واټن_j /= 2) {
            std::fill(ګام_cgraphs.begin(), ګام_cgraphs.end(), تش_اشاره);

            لكل (size_t j = 0; j < 2*دننه_واټن_j_max; j++) {
                ثابت size_t j_other = j ^ دننه_واټن_j;
                إذا (j_other >= n_backends) {
                    استمر;
                }
                ټېل_معلومات(j, j_other, i_buf);
            }

            لكل (size_t j = 0; j < 2*دننه_واټن_j_max; j++) {
                إذا (ګام_cgraphs[j] == تش_اشاره) {
                    استمر;
                }
                خپلکاری & bcj = backend_ctx->backend_configs[j];
                ثابت ggml_حالت حالت = ggml_backend_ګراف_compute_async(bcj.backend, ګام_cgraphs[j]);
                إذا (حالت != GGML_حالت_SUCCESS) {
                    أرجع حالت;
                }
            }
            i_buf++;
        }
        أكد(i_buf == backend_ctx->n_reduce_ګامونه);

        // If n_backends is not a power of 2, copy back the reduced tensors to the excess:
        لكل (size_t j = 2*دننه_واټن_j_max; j < n_backends; j++) {
            خپلکاری & bcj_src = backend_ctx->backend_configs[j - 2*دننه_واټن_j_max];
            خپلکاری & bcj_dst = backend_ctx->backend_configs[j];

            ggml_tensor * غوټه_src = bcj_src.cgraphs[i].cgraph_main->nodes[bcj_src.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_tensor * غوټه_dst = bcj_dst.cgraphs[i].cgraph_main->nodes[bcj_dst.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_backend_tensor_copy_async(bcj_src.backend, bcj_dst.backend, غوټه_src, غوټه_dst);
        }

        أرجع GGML_حالت_SUCCESS;
    };


    لكل (size_t i = 0; i < backend_ctx->n_subgraphs; i++) {
        لكل (size_t j = 0; j < n_backends; j++) {
            خپلکاری & bcj = backend_ctx->backend_configs[j];
            ثابت ggml_حالت حالت = ggml_backend_ګراف_compute_async(bcj.backend, bcj.cgraphs[i].cgraph_main);
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
                    خپلکاری & bcj = backend_ctx->backend_configs[j];
                    ggml_cgraph * cgraph_ij = bcj.cgraphs[i].cgraph_main;
                    nodes.push_back(cgraph_ij->nodes[cgraph_ij->n_nodes-1]);
                }
                backend_allreduce_success = backend_ctx->comm_allreduce(backend_ctx->comm_ctx, nodes.معلومات());
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
    /* .set_tensor_async        = */ ggml_backend_meta_سېټ_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_meta_get_tensor_async,
    /* .set_tensor_2d_async     = */ تش_اشاره,
    /* .get_tensor_2d_async     = */ تش_اشاره,
    /* .cpy_tensor_async        = */ تش_اشاره,
    /* .synchronize             = */ ggml_backend_meta_synchronize,
    /* .graph_plan_create       = */ تش_اشاره,
    /* .graph_plan_free         = */ تش_اشاره,
    /* .graph_plan_update       = */ تش_اشاره,
    /* .graph_plan_compute      = */ تش_اشاره,
    /* .graph_compute           = */ ggml_backend_meta_ګراف_compute,
    /* .event_record            = */ تش_اشاره,
    /* .event_wait              = */ تش_اشاره,
    /* .graph_optimize          = */ تش_اشاره,
};

منطقي ggml_backend_is_meta(ggml_backend_t backend) {
    أرجع backend != تش_اشاره && backend->iface.get_name == ggml_backend_meta_i.get_name;
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

ggml_backend_t ggml_backend_meta_simple_backend(ggml_backend_t meta_backend, size_t شاخص) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    ثابت ggml_backend_meta_context * backend_ctx = (ثابت ggml_backend_meta_context *) meta_backend->context;
    أرجع backend_ctx->backend_configs[شاخص].backend;
}
