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
بنية ggml_backend_meta_میانگیر_type;
بنية ggml_backend_meta_میانگیر;
بنية ggml_backend_meta;

ثابت حرف * ggml_backend_meta_تقسیم_axis_name(تعداد ggml_backend_meta_تقسیم_axis تقسیم_axis) {
    بدل (تقسیم_axis) {
        حالة GGML_BACKEND_تقسیم_AXIS_0:
            أرجع "0";
        حالة GGML_BACKEND_تقسیم_AXIS_1:
            أرجع "1";
        حالة GGML_BACKEND_تقسیم_AXIS_2:
            أرجع "2";
        حالة GGML_BACKEND_تقسیم_AXIS_3:
            أرجع "3";
        حالة GGML_BACKEND_تقسیم_AXIS_MIRRORED:
            أرجع "MIRRORED";
        حالة GGML_BACKEND_تقسیم_AXIS_PARTIAL:
            أرجع "PARTIAL";
        حالة GGML_BACKEND_تقسیم_AXIS_NONE:
            أرجع "NONE";
        حالة GGML_BACKEND_تقسیم_AXIS_UNKNOWN:
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
    ggml_backend_meta_get_تقسیم_state_t get_تقسیم_state;
    فراغ *                              get_تقسیم_state_ud;

    std::string name;
    std::string description;

    ggml_backend_meta_device_context(
            std::vector<ggml_backend_dev_t> simple_devs, ggml_backend_meta_get_تقسیم_state_t get_تقسیم_state, فراغ * get_تقسیم_state_ud) :
            simple_devs(std::move(simple_devs)), get_تقسیم_state(get_تقسیم_state), get_تقسیم_state_ud(get_تقسیم_state_ud) {
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

    منطقي عملگر<(ثابت ggml_backend_meta_device_context & other) ثابت {
        أرجع std::tie(simple_devs, get_تقسیم_state, get_تقسیم_state_ud)
            < std::tie(other.simple_devs, other.get_تقسیم_state, other.get_تقسیم_state_ud);
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

ساكن فراغ ggml_backend_meta_device_get_حافظه(ggml_backend_dev_t dev, size_t * حرر, size_t * مجموع) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ثابت ggml_backend_meta_device_context * meta_dev_ctx = (ثابت ggml_backend_meta_device_context *) dev->context;
    *حرر  = 0;
    *مجموع = 0;
    لكل (ggml_backend_dev_t dev : meta_dev_ctx->simple_devs) {
        size_t tmp_free, tmp_مجموع;
        ggml_backend_dev_حافظه(dev, &tmp_free, &tmp_مجموع);
        *حرر  += tmp_free;
        *مجموع += tmp_مجموع;
    }
}

ساكن تعداد ggml_backend_dev_type ggml_backend_meta_device_get_type(ggml_backend_dev_t dev) {
    أرجع GGML_BACKEND_DEVICE_TYPE_META;

    GGML_استفادهنشده(dev);
}

ساكن فراغ ggml_backend_meta_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ثابت ggml_backend_meta_device_context * meta_dev_ctx = (ثابت ggml_backend_meta_device_context *) dev->context;

    // TODO replace placeholders
    props->name        = ggml_backend_meta_device_get_name(dev);
    props->description = ggml_backend_meta_device_get_description(dev);
    props->type        = ggml_backend_meta_device_get_type(dev);
    props->device_id   = 0;

    ggml_backend_meta_device_get_حافظه(dev, &props->حافظه_free, &props->حافظه_مجموع);

    props->caps = {
        /* .async                 = */ درست,
        /* .host_buffer           = */ خطأ, // Not implemented.
        /* .buffer_from_host_ptr  = */ خطأ, // Not implemented.
        /* .events                = */ خطأ, // Not implemented.
        /* .mmap_support          = */ درست,
    };
    لكل (ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
        ggml_backend_dev_props tmp_props;
        ggml_backend_dev_get_props(simple_dev, &tmp_props);
        props->caps.async                = props->caps.async                && tmp_props.caps.async;
        props->caps.host_میانگیر          = props->caps.host_میانگیر          && tmp_props.caps.host_میانگیر;
        props->caps.میانگیر_from_host_ptr = props->caps.میانگیر_from_host_ptr && tmp_props.caps.میانگیر_from_host_ptr;
        props->caps.events               = props->caps.events               && tmp_props.caps.events;
        props->caps.mmap_support         = props->caps.mmap_support         && tmp_props.caps.mmap_support;
    }
}

ساكن ggml_backend_t ggml_backend_meta_device_init_backend(ggml_backend_dev_t dev, ثابت حرف * params);

ساكن ggml_backend_میانگیر_type_t ggml_backend_meta_device_get_میانگیر_type(ggml_backend_dev_t dev);

ساكن ggml_backend_میانگیر_type_t ggml_backend_meta_device_get_host_میانگیر_type(ggml_backend_dev_t dev);

ساكن منطقي ggml_backend_meta_device_supports_op(ggml_backend_dev_t dev, ثابت ggml_tensor * op) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ثابت ggml_backend_meta_device_context * meta_dev_ctx = (ثابت ggml_backend_meta_device_context *) dev->context;
    أرجع std::all_of(meta_dev_ctx->simple_devs.begin(), meta_dev_ctx->simple_devs.end(),
        [op](ggml_backend_dev_t simple_dev) { أرجع ggml_backend_dev_supports_op(simple_dev, op); });
}

ساكن منطقي ggml_backend_meta_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_میانگیر_type_t buft) {
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
    أرجع درست;
}

ساكن ثابت ggml_backend_device_i ggml_backend_meta_device_iface = {
    /* .get_name             = */ ggml_backend_meta_device_get_name,
    /* .get_description      = */ ggml_backend_meta_device_get_description,
    /* .get_memory           = */ ggml_backend_meta_device_get_حافظه,
    /* .get_type             = */ ggml_backend_meta_device_get_type,
    /* .get_props            = */ ggml_backend_meta_device_get_props,
    /* .init_backend         = */ ggml_backend_meta_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_meta_device_get_میانگیر_type,
    /* .get_host_buffer_type = */ ggml_backend_meta_device_get_host_میانگیر_type,
    /* .buffer_from_host_ptr = */ اشاره_تهی,
    /* .supports_op          = */ ggml_backend_meta_device_supports_op,
    /* .supports_buft        = */ ggml_backend_meta_device_supports_buft,
    /* .offload_op           = */ اشاره_تهی,
    /* .event_new            = */ اشاره_تهی,
    /* .event_free           = */ اشاره_تهی,
    /* .event_synchronize    = */ اشاره_تهی,
};

ساكن منطقي ggml_backend_dev_is_meta(ggml_backend_dev_t dev) {
    أرجع dev != اشاره_تهی && dev->iface.get_name == ggml_backend_meta_device_iface.get_name;
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
        ggml_backend_dev_t * devs, size_t n_devs, ggml_backend_meta_get_تقسیم_state_t get_تقسیم_state, فراغ * get_تقسیم_state_ud) {
    GGML_ASSERT(n_devs <= GGML_BACKEND_META_MAX_DEVICES);
    // TODO: this is not thread-safe - needs to be fixed
    ساكن std::vector<std::unique_ptr<ggml_backend_meta_device_context>>         ctxs;
    ساكن std::map<ggml_backend_meta_device_context, بنية ggml_backend_device> meta_devs;

    std::vector<ggml_backend_dev_t> simple_devs;
    simple_devs.reserve(n_devs);
    لكل (size_t i = 0; i < n_devs; i++) {
        simple_devs.push_back(devs[i]);
    }
    ggml_backend_meta_device_context ctx(simple_devs, get_تقسیم_state, get_تقسیم_state_ud);

    {
        خودکار it = meta_devs.find(ctx);
        إذا (it != meta_devs.end()) {
            أرجع &it->second;
        }
    }
    ctxs.push_back(std::make_unique<ggml_backend_meta_device_context>(ctx));

    بنية ggml_backend_device meta_dev = {
        /*iface  =*/ ggml_backend_meta_device_iface,
        /*reg    =*/ اشاره_تهی,
        /*ctx    =*/ ctxs.back().get(),
    };

    خودکار نتیجه = meta_devs.emplace(*ctxs.back(), meta_dev);
    أرجع &نتیجه.اولی->second;
}

//
// meta backend buffer type
//

بنية ggml_backend_meta_میانگیر_type_context {
    std::vector<ggml_backend_میانگیر_type_t> simple_bufts;

    std::string name;

    ggml_backend_meta_میانگیر_type_context(std::vector<ggml_backend_میانگیر_type_t> simple_bufts) : simple_bufts(std::move(simple_bufts)) {
        name = "Meta(";
        لكل (size_t i = 0; i < simple_bufts.size(); i++) {
            إذا (i > 0) {
                name += ",";
            }
            name += ggml_backend_buft_name(simple_bufts[i]);
        }
        name += ")";
    }

    منطقي عملگر<(ثابت ggml_backend_meta_میانگیر_type_context & other) ثابت {
        أرجع simple_bufts < other.simple_bufts;
    }
};

ساكن size_t ggml_backend_meta_buft_n_bufts(ggml_backend_میانگیر_type_t meta_buft) {
    GGML_ASSERT(ggml_backend_buft_is_meta(meta_buft));
    ثابت ggml_backend_meta_میانگیر_type_context * meta_buft_ctx = (ثابت ggml_backend_meta_میانگیر_type_context *) meta_buft->context;
    أرجع meta_buft_ctx->simple_bufts.size();
}

ساكن ثابت حرف * ggml_backend_meta_میانگیر_type_get_name(ggml_backend_میانگیر_type_t buft) {
    GGML_ASSERT(ggml_backend_buft_is_meta(buft));
    ثابت ggml_backend_meta_میانگیر_type_context * meta_buft_ctx = (ثابت ggml_backend_meta_میانگیر_type_context *) buft->context;
    أرجع meta_buft_ctx->name.c_str();
}

ساكن ggml_backend_میانگیر_type_t ggml_backend_meta_buft_simple_buft(ggml_backend_میانگیر_type_t meta_buft, size_t شاخص) {
    GGML_ASSERT(ggml_backend_buft_is_meta(meta_buft));
    ثابت ggml_backend_meta_میانگیر_type_context * meta_buft_ctx = (ثابت ggml_backend_meta_میانگیر_type_context *) meta_buft->context;
    GGML_ASSERT(شاخص < meta_buft_ctx->simple_bufts.size());
    أرجع meta_buft_ctx->simple_bufts[شاخص];
}

ساكن ggml_backend_میانگیر_t ggml_backend_meta_میانگیر_type_alloc_میانگیر(ggml_backend_میانگیر_type_t buft, size_t size);

ساكن ggml_backend_میانگیر_t ggml_backend_meta_میانگیر_type_alloc_میانگیر_n(ggml_backend_میانگیر_type_t buft, ggml_tensor ** tensors, صحيح n_tensors);

ساكن size_t ggml_backend_meta_میانگیر_type_get_alignment(ggml_backend_میانگیر_type_t buft) {
    ثابت size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_alignment = 1;
    لكل (size_t i = 0; i < n_simple_bufts; i++) {
        ثابت size_t alignment = ggml_backend_buft_get_alignment(ggml_backend_meta_buft_simple_buft(buft, i));
        max_alignment = std::max(max_alignment, alignment);
        GGML_ASSERT(max_alignment % alignment == 0);
    }
    أرجع max_alignment;
}

ساكن size_t ggml_backend_meta_میانگیر_type_get_max_اندازه(ggml_backend_میانگیر_type_t buft) {
    ثابت size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_size = SIZE_MAX;
    لكل (size_t i = 0; i < n_simple_bufts; i++) {
        max_size = std::min(max_size, ggml_backend_buft_get_max_اندازه(ggml_backend_meta_buft_simple_buft(buft, i)));
    }
    أرجع max_size;
}

ساكن size_t ggml_backend_meta_میانگیر_type_get_alloc_اندازه(ggml_backend_میانگیر_type_t buft, ثابت ggml_tensor * tensor) {
    ثابت size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_alloc_اندازه = 0;
    لكل (size_t i = 0; i < n_simple_bufts; i++) {
        ثابت size_t alloc_اندازه = ggml_backend_buft_get_alloc_اندازه(ggml_backend_meta_buft_simple_buft(buft, i), tensor);
        max_alloc_اندازه = std::max(max_alloc_اندازه, alloc_اندازه);
    }
    أرجع max_alloc_اندازه;
}

ساكن منطقي ggml_backend_meta_میانگیر_type_is_host(ggml_backend_میانگیر_type_t buft) {
    ثابت size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    لكل (size_t i = 0; i < n_simple_bufts; i++) {
        إذا (!ggml_backend_buft_is_host(ggml_backend_meta_buft_simple_buft(buft, i))) {
            أرجع خطأ;
        }
    }
    أرجع درست;
}

ساكن ثابت بنية ggml_backend_میانگیر_type_i ggml_backend_meta_میانگیر_type_iface = {
    /* .get_name            = */ ggml_backend_meta_میانگیر_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_meta_میانگیر_type_alloc_میانگیر,
    /* .alloc_buffer_n      = */ ggml_backend_meta_میانگیر_type_alloc_میانگیر_n,
    /* .get_alignment       = */ ggml_backend_meta_میانگیر_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_meta_میانگیر_type_get_max_اندازه,
    /* .get_alloc_size      = */ ggml_backend_meta_میانگیر_type_get_alloc_اندازه,
    /* .get_alloc_size_n    = */ عدم,
    /* .is_host             = */ ggml_backend_meta_میانگیر_type_is_host,
};

منطقي ggml_backend_buft_is_meta(ggml_backend_میانگیر_type_t buft) {
    أرجع buft != اشاره_تهی && buft->iface.get_name == ggml_backend_meta_میانگیر_type_iface.get_name;
}

ساكن ggml_backend_میانگیر_type_t ggml_backend_meta_device_get_میانگیر_type(ggml_backend_dev_t dev) {
    ساكن std::map<ggml_backend_dev_t, بنية ggml_backend_میانگیر_type> meta_bufts;
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    {
        خودکار it = meta_bufts.find(dev);
        إذا (it != meta_bufts.end()) {
            أرجع &it->second;
        }
    }

    ثابت size_t n_devs = ggml_backend_meta_dev_n_devs(dev);
    std::vector<ggml_backend_میانگیر_type_t> simple_bufts;
    simple_bufts.reserve(n_devs);
    لكل (size_t i = 0; i < n_devs; i++) {
        simple_bufts.push_back(ggml_backend_dev_میانگیر_type(ggml_backend_meta_dev_simple_dev(dev, i)));
    }
    ggml_backend_meta_میانگیر_type_context * buft_ctx = جديد ggml_backend_meta_میانگیر_type_context(simple_bufts);

    بنية ggml_backend_میانگیر_type meta_buft = {
        /*iface  =*/ ggml_backend_meta_میانگیر_type_iface,
        /*device =*/ dev,
        /*ctx    =*/ buft_ctx,
    };
    خودکار نتیجه = meta_bufts.emplace(dev, meta_buft);
    أرجع &نتیجه.اولی->second;
}

ساكن ggml_backend_میانگیر_type_t ggml_backend_meta_device_get_host_میانگیر_type(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ثابت ggml_backend_meta_device_context * meta_dev_ctx = (ثابت ggml_backend_meta_device_context *) dev->context;

    ggml_backend_میانگیر_type_t host_buft = اشاره_تهی;
    لكل (ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
        ggml_backend_میانگیر_type_t simple_host_buft = ggml_backend_dev_host_میانگیر_type(simple_dev);
        إذا (simple_host_buft == اشاره_تهی) {
            أرجع اشاره_تهی;
        }
        إذا (host_buft == اشاره_تهی) {
            host_buft = simple_host_buft;
        } وإلا إذا (host_buft != simple_host_buft) {
            // if different simple devices have different host buffer types,
            // we cannot provide a single host buffer type for the meta device
            أرجع اشاره_تهی;
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

بنية ggml_backend_meta_میانگیر_context {
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
    صحيح stc_compute_شاخص_بعدی = 0;
    std::vector<ggml_backend_میانگیر_ptr> bufs;

    // FIXME
    // The size of the split state cache is unbounded and can theoretically grow infinitely large.
    // However, it is also expensive to build and clearing it on every rebuild in ggml_backend_meta_graph_compute is too expensive.
    ساكن constexpr size_t nbtc = GGML_TENSOR_اندازه - حجم(ggml_tensor::padding);
    std::map<std::pair<ثابت ggml_tensor *, منطقي>, std::pair<ggml_backend_meta_تقسیم_state, حرف[nbtc]>> تقسیم_state_cache;

    صحيح debug;

    ggml_backend_meta_میانگیر_context(
            ggml_backend_meta_simple_tensor_container & stc_static,
            ggml_backend_meta_simple_tensor_container & stc_compute_0,
            ggml_backend_meta_simple_tensor_container & stc_compute_1,
            ثابت std::vector<ggml_backend_میانگیر_t> & bufs)
            : stc_static(std::move(stc_static)), stc_compute{std::move(stc_compute_0), std::move(stc_compute_1)} {
        هذا->bufs.reserve(bufs.size());
        لكل (ggml_backend_میانگیر_t buf : bufs) {
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

ساكن فراغ ggml_backend_meta_میانگیر_free_میانگیر(ggml_backend_میانگیر_t میانگیر) {
    GGML_ASSERT(ggml_backend_میانگیر_is_meta(میانگیر));
    ggml_backend_meta_میانگیر_context * buf_ctx = (ggml_backend_meta_میانگیر_context *) میانگیر->context;
    حذف buf_ctx;
}

ساكن size_t ggml_backend_meta_میانگیر_n_bufs(ggml_backend_میانگیر_t meta_buf) {
    GGML_ASSERT(ggml_backend_میانگیر_is_meta(meta_buf));
    ggml_backend_meta_میانگیر_context * buf_ctx = (ggml_backend_meta_میانگیر_context *) meta_buf->context;
    أرجع buf_ctx->bufs.size();
}

ساكن ggml_backend_میانگیر_t ggml_backend_meta_میانگیر_simple_میانگیر(ggml_backend_میانگیر_t meta_buf, size_t شاخص) {
    GGML_ASSERT(ggml_backend_میانگیر_is_meta(meta_buf));
    ggml_backend_meta_میانگیر_context * buf_ctx = (ggml_backend_meta_میانگیر_context *) meta_buf->context;
    GGML_ASSERT(شاخص < buf_ctx->bufs.size());
    أرجع buf_ctx->bufs[شاخص].get();
}

ساكن بنية ggml_tensor * ggml_backend_meta_میانگیر_simple_tensor(ثابت بنية ggml_tensor * tensor, size_t شاخص) {
    GGML_ASSERT(ggml_backend_میانگیر_is_meta(tensor->میانگیر));
    ggml_backend_meta_میانگیر_context * buf_ctx = (ggml_backend_meta_میانگیر_context *) tensor->میانگیر->context;
    GGML_ASSERT(شاخص < buf_ctx->bufs.size());

    ggml_backend_meta_simple_tensor_container & stc = buf_ctx->get_simple_tensor_container(tensor);
    خودکار it = stc.simple_tensors.find(tensor);
    إذا (it == stc.simple_tensors.end()) {
        أرجع اشاره_تهی;
    }
    أرجع it->second[شاخص];
}

ساكن بنية ggml_backend_meta_تقسیم_state ggml_backend_meta_get_تقسیم_state(ثابت بنية ggml_tensor * tensor, منطقي assume_sync);

ساكن بنية ggml_backend_meta_تقسیم_state ggml_backend_meta_get_تقسیم_state(
        ggml_backend_meta_simple_tensor_container & stc, ثابت بنية ggml_tensor * tensor, منطقي assume_sync) {
    // FIXME Currently this function preserves/erases the information in n_segments and nr in an inconsistent way.
    // Since the operations in question are developed specifically for llama.cpp this currently does not manifest as a bug there.
    // However, in a broader ggml context with arbitrary ggml graphs this can lead to unexpected results.
    ثابت size_t n_bufs = ggml_backend_meta_میانگیر_n_bufs(tensor->میانگیر);
    ggml_backend_meta_میانگیر_context * buf_ctx = (ggml_backend_meta_میانگیر_context *) tensor->میانگیر->context;

    خودکار تقسیم_حالتها_equal = [&](ثابت ggml_backend_meta_تقسیم_state & a, ثابت ggml_backend_meta_تقسیم_state & b) -> منطقي {
        إذا (a.axis != b.axis) {
            أرجع خطأ;
        }
        لكل (size_t j = 0; j < n_bufs; j++) {
            int64_t جمع_a = 0;
            لكل (size_t s = 0; s < a.n_segments; s++) {
                جمع_a += a.ne[s*n_bufs + j] * a.nr[s];
            }
            int64_t جمع_b = 0;
            لكل (size_t s = 0; s < b.n_segments; s++) {
                جمع_b += b.ne[s*n_bufs + j] * b.nr[s];
            }
            إذا (جمع_a != جمع_b) {
                أرجع خطأ;
            }
        }
        أرجع درست;
    };

    خودکار handle_generic = [&](ثابت std::vector<ggml_backend_meta_تقسیم_state> & src_ss, منطقي scalar_only) -> ggml_backend_meta_تقسیم_state {
        ggml_backend_meta_تقسیم_state ret = {GGML_BACKEND_تقسیم_AXIS_NONE, {0}, {1}, 1};
        لكل (size_t i = 0; i < GGML_MAX_SRC; i++) {
            إذا (tensor->src[i] == اشاره_تهی || tensor->src[i] == tensor) {
                استمر;
            }
            إذا (ret.axis == GGML_BACKEND_تقسیم_AXIS_NONE) {
                ret = src_ss[i];
            } وإلا إذا (!تقسیم_حالتها_equal(src_ss[i], ret)) {
                ret = {GGML_BACKEND_تقسیم_AXIS_UNKNOWN, {0}, {1}, 1};
                اكسر;
            }
        }
        إذا (ret.axis == GGML_BACKEND_تقسیم_AXIS_NONE) {
            ret = {GGML_BACKEND_تقسیم_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        إذا (scalar_only && ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
            ret = {GGML_BACKEND_تقسیم_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        GGML_ASSERT(ret.axis != GGML_BACKEND_تقسیم_AXIS_UNKNOWN);
        أرجع ret;
    };

    // Some ops process data on a per-row bases:
    خودکار handle_در_هر_ردیف = [&](ثابت std::vector<ggml_backend_meta_تقسیم_state> & src_ss) -> ggml_backend_meta_تقسیم_state {
        GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_تقسیم_AXIS_0);
        أرجع src_ss[0];
    };

    // Some ops broadcast the src1 data across src0:
    خودکار handle_bin_bcast = [&](ثابت std::vector<ggml_backend_meta_تقسیم_state> & src_ss) -> ggml_backend_meta_تقسیم_state {
        إذا (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS &&
                tensor->src[1]->ne[src_ss[0].axis] == 1 && src_ss[1].axis == GGML_BACKEND_تقسیم_AXIS_MIRRORED) {
            أرجع src_ss[0];
        }
        إذا (src_ss[2].axis == GGML_BACKEND_تقسیم_AXIS_MIRRORED && (src_ss[0].axis == src_ss[1].axis ||
           (src_ss[0].axis == GGML_BACKEND_تقسیم_AXIS_MIRRORED && (src_ss[1].axis == GGML_BACKEND_تقسیم_AXIS_PARTIAL)))) {
            أرجع src_ss[0]; // GGML_OP_ADD_ID
        }
        GGML_ASSERT(tensor->src[2] == اشاره_تهی || src_ss[2].axis == GGML_BACKEND_تقسیم_AXIS_MIRRORED);
        أرجع handle_generic(src_ss, /*scalar_only =*/ خطأ);
    };

    خودکار handle_concat = [&](ثابت std::vector<ggml_backend_meta_تقسیم_state> & src_ss) -> ggml_backend_meta_تقسیم_state {
        ثابت ggml_backend_meta_تقسیم_axis concat_axis = ggml_backend_meta_تقسیم_axis(ggml_get_op_params_i32(tensor, 0));
        إذا (src_ss[0].axis == GGML_BACKEND_تقسیم_AXIS_MIRRORED && src_ss[1].axis >= 0 && src_ss[1].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(concat_axis != src_ss[1].axis);
            أرجع src_ss[1];
        }
        إذا (src_ss[1].axis == GGML_BACKEND_تقسیم_AXIS_MIRRORED && src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(concat_axis != src_ss[0].axis);
            أرجع src_ss[0];
        }
        إذا (src_ss[0].axis == src_ss[1].axis && src_ss[0].axis != concat_axis) {
            أرجع src_ss[0];
        }
        أرجع handle_generic(src_ss, /*scalar_only =*/ درست);
    };

    خودکار handle_mul_mat = [&](ثابت std::vector<ggml_backend_meta_تقسیم_state> & src_ss) -> ggml_backend_meta_تقسیم_state {
        إذا (src_ss[0].axis == GGML_BACKEND_تقسیم_AXIS_MIRRORED && src_ss[1].axis == GGML_BACKEND_تقسیم_AXIS_MIRRORED) {
            أرجع {GGML_BACKEND_تقسیم_AXIS_MIRRORED, {0}, {1}, 1};
        }
        إذا (src_ss[0].axis == GGML_BACKEND_تقسیم_AXIS_1 && src_ss[1].axis == GGML_BACKEND_تقسیم_AXIS_MIRRORED) {
            ggml_backend_meta_تقسیم_state ret = src_ss[0];
            ret.axis = GGML_BACKEND_تقسیم_AXIS_0;
            ret.nr[0] = 1;
            ret.n_segments = 1;
            أرجع ret;
        }
        إذا (src_ss[1].axis == GGML_BACKEND_تقسیم_AXIS_1 && src_ss[0].axis == GGML_BACKEND_تقسیم_AXIS_MIRRORED) {
            أرجع src_ss[1];
        }
        إذا (src_ss[0].axis == GGML_BACKEND_تقسیم_AXIS_0 && src_ss[1].axis == GGML_BACKEND_تقسیم_AXIS_0) {
            GGML_ASSERT(تقسیم_حالتها_equal(src_ss[0], src_ss[1]));
            أرجع {assume_sync ? GGML_BACKEND_تقسیم_AXIS_MIRRORED : GGML_BACKEND_تقسیم_AXIS_PARTIAL, {0}, {1}, 1};
        }
        إذا (src_ss[0].axis == src_ss[1].axis && src_ss[0].axis >= GGML_BACKEND_تقسیم_AXIS_2 &&
                src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(تقسیم_حالتها_equal(src_ss[0], src_ss[1]));
            أرجع src_ss[0];
        }
        // batched matmul with the batches split across devices and a replicated activation
        إذا (src_ss[0].axis >= GGML_BACKEND_تقسیم_AXIS_2 && src_ss[0].axis < GGML_MAX_DIMS &&
                src_ss[1].axis == GGML_BACKEND_تقسیم_AXIS_MIRRORED) {
            أرجع src_ss[0];
        }
        GGML_ABORT("unsupported mul_mat split states: node=%s src0=%s axis=%d src1=%s axis=%d",
            tensor->name, tensor->src[0]->name, (صحيح) src_ss[0].axis, tensor->src[1]->name, (صحيح) src_ss[1].axis);
        //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
    };

    خودکار handle_reshape = [&](ثابت std::vector<ggml_backend_meta_تقسیم_state> & src_ss) -> ggml_backend_meta_تقسیم_state {
        بدل (src_ss[0].axis) {
            حالة GGML_BACKEND_تقسیم_AXIS_0:
            حالة GGML_BACKEND_تقسیم_AXIS_1:
            حالة GGML_BACKEND_تقسیم_AXIS_2:
            حالة GGML_BACKEND_تقسیم_AXIS_3: {
                int64_t پایه_ne_درون = 1;
                لكل (صحيح dim = 0; dim <= src_ss[0].axis; dim++) {
                    پایه_ne_درون *= tensor->src[0]->ne[dim];
                }
                إذا (src_ss[0].n_segments == 1) {
                    پایه_ne_درون /= src_ss[0].nr[0];
                    إذا (src_ss[0].axis == ggml_n_dims(tensor->src[0]) - 1 && src_ss[0].nr[0] == 1) {
                        أرجع {ggml_backend_meta_تقسیم_axis(ggml_n_dims(tensor) - 1), {0}, {1}, 1};
                    }
                    إذا (src_ss[0].axis == GGML_BACKEND_تقسیم_AXIS_0 && tensor->ne[0] == tensor->src[0]->ne[0] &&
                            tensor->ne[1] == 1 && src_ss[0].nr[0] == 1) {
                        منطقي complete_ردیفها = درست;
                        لكل (size_t j = 0; j < n_bufs; j++) {
                            ثابت int64_t ne = src_ss[0].ne[j];
                            complete_ردیفها = complete_ردیفها && (ne == 0 || ne == tensor->src[0]->ne[0]);
                        }
                        إذا (complete_ردیفها) {
                            // Move a complete dim-0 split to the following singleton dimension.
                            أرجع {GGML_BACKEND_تقسیم_AXIS_1, {0}, {1}, 1};
                        }
                    }
                }
                // Reshape outputs use one segment; split-state propagation merges source segments.
                int64_t پایه_ne_بیرون = 1;
                لكل (صحيح dim = 0; dim < GGML_MAX_DIMS; dim++) {
                    پایه_ne_بیرون *= tensor->ne[dim];
                    إذا (پایه_ne_بیرون % پایه_ne_درون == 0) {
                        أرجع {ggml_backend_meta_تقسیم_axis(dim), {0}, {uint32_t(پایه_ne_بیرون/پایه_ne_درون)}, 1};
                    }
                    إذا (پایه_ne_بیرون > پایه_ne_درون) {
                        GGML_ASSERT(src_ss[0].n_segments == 1);
                        GGML_ASSERT(src_ss[0].nr[0]      == 1);
                        أرجع {ggml_backend_meta_تقسیم_axis(dim), {0}, {1}, 1};
                    }
                }
                GGML_ABORT("shape mismatch for %s", ggml_op_name(tensor->op));
            }
            حالة GGML_BACKEND_تقسیم_AXIS_MIRRORED:
            حالة GGML_BACKEND_تقسیم_AXIS_PARTIAL: {
                أرجع src_ss[0];
            }
            افتراضي: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    خودکار handle_cpy = [&](ثابت std::vector<ggml_backend_meta_تقسیم_state> & src_ss) -> ggml_backend_meta_تقسیم_state {
        إذا (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            أرجع handle_reshape(src_ss);
        }
        أرجع handle_generic(src_ss, /*scalar_only =*/ خطأ);
    };

    خودکار handle_view = [&](ثابت std::vector<ggml_backend_meta_تقسیم_state> & src_ss) -> ggml_backend_meta_تقسیم_state {
        إذا (ggml_is_contiguous(tensor) && ggml_is_contiguous(tensor->src[0])) {
            أرجع handle_reshape(src_ss);
        }
        ثابت صحيح axis = src_ss[0].axis;
        {
            منطقي all_strides_the_same = درست;
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
                    أرجع {ggml_backend_meta_تقسیم_axis(dim), {0}, {1}, 1};
                }
            }
            GGML_ABORT("fatal error");
        }
        إذا (src_ss[0].axis == GGML_BACKEND_تقسیم_AXIS_MIRRORED || src_ss[0].axis == GGML_BACKEND_تقسیم_AXIS_PARTIAL) {
            أرجع src_ss[0];
        }
        GGML_ABORT("view of permuted tensor not implemented");
        //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
    };

    خودکار handle_جایگشت = [&](ثابت std::vector<ggml_backend_meta_تقسیم_state> & src_ss) -> ggml_backend_meta_تقسیم_state {
        بدل (src_ss[0].axis) {
            حالة GGML_BACKEND_تقسیم_AXIS_0:
            حالة GGML_BACKEND_تقسیم_AXIS_1:
            حالة GGML_BACKEND_تقسیم_AXIS_2:
            حالة GGML_BACKEND_تقسیم_AXIS_3: {
                GGML_ASSERT(src_ss[0].n_segments == 1 || src_ss[0].nr[0] == 1);
                أرجع {ggml_backend_meta_تقسیم_axis(tensor->op_params[src_ss[0].axis]), {0}, {src_ss[0].nr[0]}, 1};
            }
            حالة GGML_BACKEND_تقسیم_AXIS_MIRRORED:
            حالة GGML_BACKEND_تقسیم_AXIS_PARTIAL: {
                أرجع src_ss[0];
            }
            افتراضي: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    خودکار handle_transpose = [&](ثابت std::vector<ggml_backend_meta_تقسیم_state> & src_ss) -> ggml_backend_meta_تقسیم_state {
        بدل (src_ss[0].axis) {
            حالة GGML_BACKEND_تقسیم_AXIS_0:
            حالة GGML_BACKEND_تقسیم_AXIS_1: {
                GGML_ASSERT(src_ss[0].n_segments == 1 || src_ss[0].nr[0] == 1);
                أرجع {ggml_backend_meta_تقسیم_axis(صحيح(src_ss[0].axis) ^ 1), {0}, {src_ss[0].nr[0]}, 1};
            }
            حالة GGML_BACKEND_تقسیم_AXIS_2:
            حالة GGML_BACKEND_تقسیم_AXIS_3:
            حالة GGML_BACKEND_تقسیم_AXIS_MIRRORED:
            حالة GGML_BACKEND_تقسیم_AXIS_PARTIAL: {
                أرجع src_ss[0];
            }
            افتراضي: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    خودکار handle_get_ردیفها = [&](ثابت std::vector<ggml_backend_meta_تقسیم_state> & src_ss) -> ggml_backend_meta_تقسیم_state {
        إذا (src_ss[0].axis == GGML_BACKEND_تقسیم_AXIS_0 && src_ss[1].axis == GGML_BACKEND_تقسیم_AXIS_MIRRORED) {
            أرجع src_ss[0];
        }
        أرجع handle_generic(src_ss, /*scalar_only =*/ درست);
    };

    خودکار handle_مجموعه_ردیفها = [&](ثابت std::vector<ggml_backend_meta_تقسیم_state> & src_ss) -> ggml_backend_meta_تقسیم_state {
        GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_تقسیم_AXIS_1);
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_تقسیم_AXIS_MIRRORED);
        GGML_ASSERT(تقسیم_حالتها_equal(src_ss[0], src_ss[2]));
        أرجع src_ss[0];
    };

    خودکار handle_rope = [&](ثابت std::vector<ggml_backend_meta_تقسیم_state> & src_ss) -> ggml_backend_meta_تقسیم_state {
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_تقسیم_AXIS_MIRRORED);
        أرجع src_ss[0];
    };

    خودکار handle_pad = [&](ثابت std::vector<ggml_backend_meta_تقسیم_state> & src_ss) -> ggml_backend_meta_تقسیم_state {
        إذا (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(tensor->op_params[2*src_ss[0].axis + 0] == 0);
            GGML_ASSERT(tensor->op_params[2*src_ss[0].axis + 1] == 0);
        }
        أرجع src_ss[0];
    };

    خودکار handle_pool_1d = [&](ثابت std::vector<ggml_backend_meta_تقسیم_state> & src_ss) -> ggml_backend_meta_تقسیم_state {
        إذا (src_ss[0].axis == GGML_BACKEND_تقسیم_AXIS_0) {
            أرجع {GGML_BACKEND_تقسیم_AXIS_MIRRORED, {0}, {1}, 1};
        }
        أرجع src_ss[0];
    };

    خودکار handle_flash_attn_ext = [&](ثابت std::vector<ggml_backend_meta_تقسیم_state> & src_ss) -> ggml_backend_meta_تقسیم_state {
        GGML_ASSERT(tensor->src[3] == اشاره_تهی || src_ss[3].axis == GGML_BACKEND_تقسیم_AXIS_MIRRORED);

        إذا (src_ss[0].axis == GGML_BACKEND_تقسیم_AXIS_MIRRORED) {
            GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_تقسیم_AXIS_MIRRORED);
            GGML_ASSERT(src_ss[2].axis == GGML_BACKEND_تقسیم_AXIS_MIRRORED);
            GGML_ASSERT(tensor->src[4] == اشاره_تهی || src_ss[4].axis == GGML_BACKEND_تقسیم_AXIS_MIRRORED);
            أرجع {GGML_BACKEND_تقسیم_AXIS_MIRRORED, {0}, {1}, 1};
        }

        GGML_ASSERT(src_ss[0].axis == GGML_BACKEND_تقسیم_AXIS_2);
        ثابت منطقي kv_تقسیم = src_ss[1].axis == GGML_BACKEND_تقسیم_AXIS_2 &&
                src_ss[2].axis == GGML_BACKEND_تقسیم_AXIS_2;
        ثابت منطقي kv_mirrored = src_ss[1].axis == GGML_BACKEND_تقسیم_AXIS_MIRRORED &&
                src_ss[2].axis == GGML_BACKEND_تقسیم_AXIS_MIRRORED;
        GGML_ASSERT(kv_تقسیم || kv_mirrored);
        GGML_ASSERT(tensor->src[4] == اشاره_تهی || src_ss[4].axis == GGML_BACKEND_تقسیم_AXIS_0);
        أرجع {GGML_BACKEND_تقسیم_AXIS_1, {0}, {1}, 1};
    };

    خودکار handle_lightning_indexer = [&](
            ثابت std::vector<ggml_backend_meta_تقسیم_state> & src_ss) -> ggml_backend_meta_تقسیم_state {
        لكل (size_t i = 0; i < 4; i++) {
            GGML_ASSERT(src_ss[i].axis == GGML_BACKEND_تقسیم_AXIS_MIRRORED);
        }
        أرجع {GGML_BACKEND_تقسیم_AXIS_MIRRORED, {0}, {1}, 1};
    };

    خودکار handle_ssm_conv = [&](ثابت std::vector<ggml_backend_meta_تقسیم_state> & src_ss) -> ggml_backend_meta_تقسیم_state {
        إذا (src_ss[0].axis == src_ss[1].axis) {
            إذا (src_ss[0].axis == GGML_BACKEND_تقسیم_AXIS_0) {
                أرجع {GGML_BACKEND_تقسیم_AXIS_1, {0}, {1}, 1};
            }
            إذا (src_ss[0].axis == GGML_BACKEND_تقسیم_AXIS_1) {
                أرجع {GGML_BACKEND_تقسیم_AXIS_0, {0}, {1}, 1};
            }
        }
        أرجع handle_generic(src_ss, /*scalar_only =*/ خطأ);
    };

    خودکار handle_gated_delta_net = [&](ثابت std::vector<ggml_backend_meta_تقسیم_state> & src_ss) -> ggml_backend_meta_تقسیم_state {
        إذا (src_ss[0].axis == GGML_BACKEND_تقسیم_AXIS_MIRRORED && src_ss[1].axis == GGML_BACKEND_تقسیم_AXIS_MIRRORED &&
                src_ss[2].axis == GGML_BACKEND_تقسیم_AXIS_MIRRORED && src_ss[3].axis == GGML_BACKEND_تقسیم_AXIS_MIRRORED &&
                src_ss[4].axis == GGML_BACKEND_تقسیم_AXIS_MIRRORED && src_ss[5].axis == GGML_BACKEND_تقسیم_AXIS_MIRRORED) {
            أرجع src_ss[0];
        }
        GGML_ASSERT(src_ss[0].axis == GGML_BACKEND_تقسیم_AXIS_1);
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_تقسیم_AXIS_1);
        GGML_ASSERT(src_ss[2].axis == GGML_BACKEND_تقسیم_AXIS_1);
        GGML_ASSERT(src_ss[3].axis == GGML_BACKEND_تقسیم_AXIS_1);
        GGML_ASSERT(src_ss[4].axis == GGML_BACKEND_تقسیم_AXIS_1);
        // state shape is [S_v, S_v, H_v, n_seqs] (s0 only); the heads dim is its own axis 2,
        // so a head-aligned split on the input cache lands on axis 2 here.
        GGML_ASSERT(src_ss[5].axis == GGML_BACKEND_تقسیم_AXIS_2 || src_ss[5].axis == GGML_BACKEND_تقسیم_AXIS_1 || src_ss[5].axis == GGML_BACKEND_تقسیم_AXIS_0);
        أرجع {GGML_BACKEND_تقسیم_AXIS_0, {0}, {1}, 1};
    };

    خودکار calculate_تقسیم_state = [&]() -> ggml_backend_meta_تقسیم_state {
        إذا (ggml_nelements(tensor) == 0) {
            أرجع {GGML_BACKEND_تقسیم_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        إذا (ggml_backend_میانگیر_get_usage(tensor->میانگیر) != GGML_BACKEND_میانگیر_USAGE_COMPUTE && tensor->view_src == اشاره_تهی) {
            ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_میانگیر_get_type(tensor->میانگیر));
            ثابت ggml_backend_meta_device_context * dev_ctx = (ثابت ggml_backend_meta_device_context *) dev->context;
            ggml_backend_meta_تقسیم_state ret = dev_ctx->get_تقسیم_state(tensor, dev_ctx->get_تقسیم_state_ud);
            إذا (ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
                ثابت int64_t granularity = ret.axis == GGML_BACKEND_تقسیم_AXIS_0 ? ggml_blck_اندازه(tensor->type) : 1;
                int64_t ne_جمع = 0;
                لكل (size_t s = 0; s < ret.n_segments; s++) {
                    لكل (size_t j = 0; j < n_bufs; j++) {
                        GGML_ASSERT(ret.ne[s*n_bufs + j] % granularity == 0);
                        ne_جمع += ret.ne[s*n_bufs + j] * ret.nr[s];
                    }
                }
                GGML_ASSERT(ne_جمع == tensor->ne[ret.axis]);
            } وإلا إذا (ret.axis == GGML_BACKEND_تقسیم_AXIS_PARTIAL) {
                GGML_ASSERT(ret.n_segments == 1);
                GGML_ASSERT(ret.nr[0] == 1);
            }
            أرجع ret;
        }

        std::vector<ggml_backend_meta_تقسیم_state> src_ss(GGML_MAX_SRC, {GGML_BACKEND_تقسیم_AXIS_NONE, {0}, {1}, 1});
        لكل (size_t i = 0; i < GGML_MAX_SRC; i++) {
            إذا (tensor->src[i] == اشاره_تهی || tensor->src[i] == tensor) {
                src_ss[i] = {GGML_BACKEND_تقسیم_AXIS_UNKNOWN, {0}, {1}, 1};
                استمر;
            }
            src_ss[i] = ggml_backend_meta_get_تقسیم_state(stc, tensor->src[i], /*assume_sync =*/ درست);
            GGML_ASSERT(src_ss[i].axis != GGML_BACKEND_تقسیم_AXIS_UNKNOWN);
        }

        ggml_backend_meta_تقسیم_state تقسیم_state;
        بدل (tensor->op) {
            حالة GGML_OP_NONE: {
                إذا (tensor->view_src != اشاره_تهی) {
                    // full-tensor view created with ggml_view_tensor, transparent for the split state
                    تقسیم_state = ggml_backend_meta_get_تقسیم_state(stc, tensor->view_src, assume_sync);
                } وإلا {
                    تقسیم_state = {GGML_BACKEND_تقسیم_AXIS_MIRRORED, {0}, {1}, 1};
                }
            } اكسر;
            حالة GGML_OP_DUP: {
                تقسیم_state = handle_generic(src_ss, /*scalar_only =*/ درست);
            } اكسر;
            حالة GGML_OP_جمع_کن:
            حالة GGML_OP_جمع_کن_ID: {
                تقسیم_state = handle_bin_bcast(src_ss);
            } اكسر;
            حالة GGML_OP_ADD1:
            حالة GGML_OP_ACC: {
                تقسیم_state = handle_generic(src_ss, /*scalar_only =*/ درست);
            } اكسر;
            حالة GGML_OP_SUB:
            حالة GGML_OP_MUL:
            حالة GGML_OP_DIV: {
                تقسیم_state = handle_bin_bcast(src_ss);
            } اكسر;
            حالة GGML_OP_SQR:
            حالة GGML_OP_SQRT:
            حالة GGML_OP_LOG:
            حالة GGML_OP_SIN:
            حالة GGML_OP_COS: {
                تقسیم_state = handle_generic(src_ss, /*scalar_only =*/ خطأ);
            } اكسر;
            حالة GGML_OP_جمع: {
                تقسیم_state = handle_generic(src_ss, /*scalar_only =*/ درست);
            } اكسر;
            حالة GGML_OP_جمع_ردیفها:
            حالة GGML_OP_CUMSUM:
            حالة GGML_OP_میانگین:
            حالة GGML_OP_ARGMAX:
            حالة GGML_OP_شمار_EQUAL: {
                تقسیم_state = handle_در_هر_ردیف(src_ss);
            } اكسر;
            حالة GGML_OP_REPEAT:
            حالة GGML_OP_REPEAT_پشت: {
                تقسیم_state = handle_generic(src_ss, /*scalar_only =*/ خطأ);
            } اكسر;
            حالة GGML_OP_CONCAT: {
                تقسیم_state = handle_concat(src_ss);
            } اكسر;
            حالة GGML_OP_SILU_پشت: {
                تقسیم_state = handle_generic(src_ss, /*scalar_only =*/ خطأ);
            } اكسر;
            حالة GGML_OP_NORM:
            حالة GGML_OP_RMS_NORM:
            حالة GGML_OP_RMS_NORM_پشت:
            حالة GGML_OP_گروه_NORM:
            حالة GGML_OP_L2_NORM: {
                تقسیم_state = handle_در_هر_ردیف(src_ss);
            } اكسر;
            حالة GGML_OP_MUL_MAT:
            حالة GGML_OP_MUL_MAT_ID: {
                تقسیم_state = handle_mul_mat(src_ss);
            } اكسر;
            حالة GGML_OP_بیرون_PROD: {
                تقسیم_state = handle_generic(src_ss, /*scalar_only =*/ درست);
            } اكسر;
            حالة GGML_OP_SCALE: {
                تقسیم_state = handle_generic(src_ss, /*scalar_only =*/ خطأ);
            } اكسر;
            حالة GGML_OP_مجموعه: {
                تقسیم_state = handle_generic(src_ss, /*scalar_only =*/ درست);
            } اكسر;
            حالة GGML_OP_CPY: {
                تقسیم_state = handle_cpy(src_ss);
            } اكسر;
            حالة GGML_OP_CONT:
            حالة GGML_OP_RESHAPE: {
                تقسیم_state = handle_reshape(src_ss);
            } اكسر;
            حالة GGML_OP_VIEW: {
                تقسیم_state = handle_view(src_ss);
            } اكسر;
            حالة GGML_OP_جایگشت: {
                تقسیم_state = handle_جایگشت(src_ss);
            } اكسر;
            حالة GGML_OP_TRANSPOSE: {
                تقسیم_state = handle_transpose(src_ss);
            } اكسر;
            حالة GGML_OP_GET_ردیفها: {
                تقسیم_state = handle_get_ردیفها(src_ss);
            } اكسر;
            حالة GGML_OP_GET_ردیفها_پشت: {
                تقسیم_state = handle_generic(src_ss, /*scalar_only =*/ درست);
            } اكسر;
            حالة GGML_OP_مجموعه_ردیفها: {
                تقسیم_state = handle_مجموعه_ردیفها(src_ss);
            } اكسر;
            حالة GGML_OP_DIAG:
            حالة GGML_OP_DIAG_ماسک_INF:
            حالة GGML_OP_DIAG_ماسک_ZERO: {
                تقسیم_state = handle_generic(src_ss, /*scalar_only =*/ درست);
            } اكسر;
            حالة GGML_OP_SOFT_MAX:
            حالة GGML_OP_SOFT_MAX_پشت: {
                تقسیم_state = handle_generic(src_ss, /*scalar_only =*/ خطأ);
            } اكسر;
            حالة GGML_OP_ROPE: {
                تقسیم_state = handle_rope(src_ss);
            } اكسر;
            حالة GGML_OP_ROPE_پشت: {
                تقسیم_state = handle_rope(src_ss);
            } اكسر;
            حالة GGML_OP_CLAMP: {
                تقسیم_state = handle_generic(src_ss, /*scalar_only =*/ خطأ);
            } اكسر;
            حالة GGML_OP_CONV_TRANSPOSE_1D:
            حالة GGML_OP_IM2_ستون:
            حالة GGML_OP_IM2_ستون_پشت:
            حالة GGML_OP_IM2_ستون_3_D:
            حالة GGML_OP_CONV_2D:
            حالة GGML_OP_CONV_3D:
            حالة GGML_OP_CONV_2D_DW:
            حالة GGML_OP_CONV_TRANSPOSE_2D:
            حالة GGML_OP_POOL_2D:
            حالة GGML_OP_POOL_2_D_پشت:
            حالة GGML_OP_UPSCALE: {
                تقسیم_state = handle_generic(src_ss, /*scalar_only =*/ درست);
            } اكسر;
            حالة GGML_OP_POOL_1D: {
                تقسیم_state = handle_pool_1d(src_ss);
            } اكسر;
            حالة GGML_OP_PAD: {
                تقسیم_state = handle_pad(src_ss);
            } اكسر;
            حالة GGML_OP_PAD_REFLECT_1D:
            حالة GGML_OP_ROLL:
            حالة GGML_OP_ARANGE:
            حالة GGML_OP_TIMESTEP_EMBEDDING: {
                تقسیم_state = handle_generic(src_ss, /*scalar_only =*/ درست);
            } اكسر;
            حالة GGML_OP_ARGSORT:
            حالة GGML_OP_بالاترین_K: {
                تقسیم_state = handle_در_هر_ردیف(src_ss);
            } اكسر;
            حالة GGML_OP_LEAKY_RELU: {
                تقسیم_state = handle_generic(src_ss, /*scalar_only =*/ خطأ);
            } اكسر;
            حالة GGML_OP_TRI: {
                تقسیم_state = handle_generic(src_ss, /*scalar_only =*/ درست);
            } اكسر;
            حالة GGML_OP_FILL: {
                تقسیم_state = handle_generic(src_ss, /*scalar_only =*/ خطأ);
            } اكسر;
            حالة GGML_OP_FLASH_ATTN_EXT: {
                تقسیم_state = handle_flash_attn_ext(src_ss);
            } اكسر;
            حالة GGML_OP_FLASH_ATTN_پشت: {
                تقسیم_state = handle_generic(src_ss, /*scalar_only =*/ درست);
            } اكسر;
            حالة GGML_OP_SSM_CONV: {
                تقسیم_state = handle_ssm_conv(src_ss);
            } اكسر;
            حالة GGML_OP_SSM_پیمایش:
            حالة GGML_OP_WIN_PART:
            حالة GGML_OP_WIN_UNPART:
            حالة GGML_OP_GET_REL_POS:
            حالة GGML_OP_جمع_کن_REL_POS:
            حالة GGML_OP_RWKV_WKV6:
            حالة GGML_OP_GATED_خطی_ATTN:
            حالة GGML_OP_RWKV_WKV7:
            حالة GGML_OP_SOLVE_TRI: {
                تقسیم_state = handle_generic(src_ss, /*scalar_only =*/ درست);
            } اكسر;
            حالة GGML_OP_GATED_DELTA_NET: {
                تقسیم_state = handle_gated_delta_net(src_ss);
            } اكسر;
            حالة GGML_OP_LIGHTNING_INDEXER: {
                تقسیم_state = handle_lightning_indexer(src_ss);
            } اكسر;
            حالة GGML_OP_DSV4_HC_COMB:
            حالة GGML_OP_DSV4_HC_PRE:
            حالة GGML_OP_DSV4_HC_POST:
            حالة GGML_OP_بالاترین_K_ماسک: {
                تقسیم_state = handle_generic(src_ss, /*scalar_only =*/ درست);
            } اكسر;
            حالة GGML_OP_UNARY: {
                تقسیم_state = handle_generic(src_ss, /*scalar_only =*/ خطأ);
            } اكسر;
            حالة GGML_OP_MAP_CUSTOM1:
            حالة GGML_OP_MAP_CUSTOM2:
            حالة GGML_OP_MAP_CUSTOM3:
            حالة GGML_OP_CUSTOM: {
                تقسیم_state = handle_generic(src_ss, /*scalar_only =*/ درست);
            } اكسر;
            حالة GGML_OP_CROSS_ENTROPY_LOSS:
            حالة GGML_OP_CROSS_ENTROPY_LOSS_پشت: {
                تقسیم_state = handle_در_هر_ردیف(src_ss);
            } اكسر;
            حالة GGML_OP_OPT_گام_ADAMW:
            حالة GGML_OP_OPT_گام_SGD:
            حالة GGML_OP_GLU: {
                تقسیم_state = handle_generic(src_ss, /*scalar_only =*/ خطأ);
            } اكسر;
            افتراضي: {
                GGML_ABORT("ggml op not implemented: %s", ggml_op_name(tensor->op));
                تقسیم_state = {GGML_BACKEND_تقسیم_AXIS_UNKNOWN, {0}, {1}, 1};
            } اكسر;
        }
        إذا (تقسیم_state.axis >= 0 && تقسیم_state.axis < GGML_MAX_DIMS) {
            منطقي اولی_src_تقسیم_by_axis = درست;
            ثابت size_t n_bufs = ggml_backend_meta_میانگیر_n_bufs(tensor->میانگیر);

            لكل (size_t i = 0; i < GGML_MAX_SRC; i++) {
                إذا (tensor->src[i] == اشاره_تهی || src_ss[i].axis < 0 || src_ss[i].axis >= GGML_MAX_DIMS) {
                    استمر;
                }
                إذا (اولی_src_تقسیم_by_axis) {
                    لكل (size_t j = 0; j < n_bufs; j++) {
                        // Take over ratio from src:
                        لكل (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            تقسیم_state.ne[s*n_bufs + j] = 0;
                        }
                        لكل (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            تقسیم_state.ne[j] += src_ss[i].ne[s*n_bufs + j] * src_ss[i].nr[s];
                        }
                        تقسیم_state.ne[j] *= tensor->ne[تقسیم_state.axis];
                        إذا (تقسیم_state.ne[j] != 0 || tensor->src[i]->ne[src_ss[i].axis] != 0) {
                            ثابت int64_t div = tensor->src[i]->ne[src_ss[i].axis] * تقسیم_state.nr[0];
                            GGML_ASSERT(تقسیم_state.ne[j] % div == 0);
                            تقسیم_state.ne[j] /= div;
                        }
                    }
                } وإلا {
                    GGML_ASSERT(تقسیم_state.n_segments == 1);
                    لكل (size_t j = 0; j < n_bufs; j++) {
                        // Assert that ratio is consistent:
                        int64_t جمع = 0;
                        لكل (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            جمع += src_ss[i].ne[s*n_bufs + j] * src_ss[i].nr[s];
                        }
                        GGML_ASSERT(تقسیم_state.ne[j]*تقسیم_state.nr[0] * tensor->src[i]->ne[src_ss[i].axis]
                                                                 == جمع * tensor->ne[تقسیم_state.axis]);
                    }
                }
                اولی_src_تقسیم_by_axis = خطأ;
            }
            GGML_ASSERT(!اولی_src_تقسیم_by_axis);
        }
        أرجع تقسیم_state;
    };

    ثابت std::pair کلید = std::make_pair(tensor, assume_sync);
    خودکار it = buf_ctx->تقسیم_state_cache.find(کلید);
    إذا (it != buf_ctx->تقسیم_state_cache.end() && memcmp(it->second.second, (ثابت حرف *) tensor, حجم(it->second.second)) != 0) {
        buf_ctx->تقسیم_state_cache.clear();
        it = buf_ctx->تقسیم_state_cache.end();
    }

    إذا (it == buf_ctx->تقسیم_state_cache.end()) {
        buf_ctx->تقسیم_state_cache[کلید].اولی = calculate_تقسیم_state();
        کپی_حافظه(buf_ctx->تقسیم_state_cache[کلید].second, tensor, حجم(buf_ctx->تقسیم_state_cache[کلید].second));
        إذا (buf_ctx->debug > 0) {
            std::string srcs_info;
            لكل (size_t i = 0; i < GGML_MAX_SRC; i++) {
                إذا (tensor->src[i] == اشاره_تهی || tensor->src[i] == tensor) {
                    استمر;
                }
                إذا (!srcs_info.خالی()) {
                    srcs_info += ", ";
                }
                ثابت ggml_backend_meta_تقسیم_state تقسیم_state =
                        ggml_backend_meta_get_تقسیم_state(tensor->src[i], درست);
                GGML_ASSERT(تقسیم_state.n_segments == 1);
                ثابت حرف * axis_name = ggml_backend_meta_تقسیم_axis_name(تقسیم_state.axis);
                std::string ne_info;
                لكل (size_t j = 0; j < n_bufs; j++) {
                    إذا (!ne_info.خالی()) {
                        ne_info += ", ";
                    }
                    ne_info += std::to_string(تقسیم_state.ne[j]) + "x" + std::to_string(تقسیم_state.nr[0]);
                }
                srcs_info += std::string(tensor->src[i]->name) + "[" + ggml_op_name(tensor->src[i]->op) + ", " + axis_name + ", {" + ne_info + "}]";
            }
            std::string ne_info;
            لكل (size_t j = 0; j < n_bufs; j++) {
                إذا (!ne_info.خالی()) {
                    ne_info += ", ";
                }
                ثابت ggml_backend_meta_تقسیم_state & ss = buf_ctx->تقسیم_state_cache[کلید].اولی;
                ne_info += std::to_string(ss.ne[j]) + "x" + std::to_string(ss.nr[0]);
            }
            GGML_LOG_DEBUG("SPLIT_STATE: {%s} -> %s[%s, %s, {%s}]\n", srcs_info.c_str(), tensor->name, ggml_op_name(tensor->op),
                ggml_backend_meta_تقسیم_axis_name(buf_ctx->تقسیم_state_cache[کلید].اولی.axis), ne_info.c_str());
        }
    }

    ggml_backend_meta_تقسیم_state ret = buf_ctx->تقسیم_state_cache[کلید].اولی;
    GGML_ASSERT(ret.axis != GGML_BACKEND_تقسیم_AXIS_NONE);
#اگر_تعریف_نشده NDEBUG
    إذا (ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
        int64_t ne_ret = 0;
        لكل (size_t s = 0; s < ret.n_segments; s++) {
            لكل (size_t j = 0; j < n_bufs; j++) {
                ne_ret += ret.ne[s*n_bufs + j] * ret.nr[s];
            }
        }
        أكد(ne_ret == tensor->ne[صحيح(ret.axis)]);
    }
#پایان_اگر // NDEBUG
    أرجع ret;
}

ساكن منطقي ggml_backend_meta_is_host_view(ثابت بنية ggml_tensor * tensor) {
    أرجع ggml_is_view(tensor) && ggml_backend_میانگیر_is_host(tensor->view_src->میانگیر);
}

ساكن بنية ggml_backend_meta_تقسیم_state ggml_backend_meta_get_تقسیم_state(ثابت بنية ggml_tensor * tensor, منطقي assume_sync) {
    // [TAG_META_HOST_VIEWS]
    // TODO: technically, this check should not be needed if the backend scheduler correctly prevents assigning
    //       such host-buffer views to the meta backend. figure out how to update the scheduler logic to achieve that
    //       ref: https://github.com/ggml-org/llama.cpp/pull/30217
    إذا (!ggml_backend_میانگیر_is_meta(tensor->میانگیر)) {
        GGML_ASSERT(ggml_backend_meta_is_host_view(tensor));

        // the view is not allocated in the meta buffer, it is not split across the sub-devices
        أرجع { GGML_BACKEND_تقسیم_AXIS_NONE, {0}, {1}, 1 };
    }

    ggml_backend_meta_میانگیر_context * buf_ctx = (ggml_backend_meta_میانگیر_context *) tensor->میانگیر->context;
    أرجع ggml_backend_meta_get_تقسیم_state(buf_ctx->get_simple_tensor_container(tensor), tensor, assume_sync);
}

ساكن فراغ * ggml_backend_meta_میانگیر_get_پایه(ggml_backend_میانگیر_t میانگیر) {
    GGML_استفادهنشده(میانگیر);
    أرجع (فراغ *) 0x1000000000000000; // FIXME
}

ساكن تعداد ggml_وضعیت ggml_backend_meta_میانگیر_init_tensor_impl(ggml_backend_meta_simple_tensor_container & stc, ggml_tensor * tensor) {
    GGML_ASSERT(ggml_backend_میانگیر_is_meta(tensor->میانگیر));
    ggml_backend_meta_میانگیر_context * buf_ctx = (ggml_backend_meta_میانگیر_context *) tensor->میانگیر->context;
    ثابت size_t n_simple_bufs = ggml_backend_meta_میانگیر_n_bufs(tensor->میانگیر);

    ثابت ggml_backend_meta_تقسیم_state تقسیم_state = ggml_backend_meta_get_تقسیم_state(stc, tensor, /*assume_sync =*/ درست);
    GGML_ASSERT(ggml_nelements(tensor) == 0 || تقسیم_state.axis != GGML_BACKEND_تقسیم_AXIS_UNKNOWN);
    GGML_ASSERT(تقسیم_state.n_segments <= 16);

    صحيح تقسیم_dim = تقسیم_state.axis;
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
        ggml_backend_میانگیر_t   simple_buf = buf_ctx->bufs[j].get();

        إذا (تقسیم_dim >= 0 && تقسیم_dim < GGML_MAX_DIMS) {
            // TODO: the following assert fails for llama-parallel even though the results are correct:
            // GGML_ASSERT(ggml_is_contiguously_allocated(tensor));
            ne[تقسیم_dim] = 0;
            لكل (size_t s = 0; s < تقسیم_state.n_segments; s++) {
                ne[تقسیم_dim] += تقسیم_state.ne[s*n_simple_bufs + j] * تقسیم_state.nr[s];
            }
            لكل (صحيح i = 0; i < GGML_MAX_DIMS; i++) {
                إذا (tensor->nb[i] > tensor->nb[تقسیم_dim]) {
                    nb[i] = tensor->nb[i] * ne[تقسیم_dim]/tensor->ne[تقسیم_dim];
                }
            }
        }

        ggml_tensor * t_ij = ggml_new_tensor(simple_ctx, tensor->type, GGML_MAX_DIMS, ne);
        t_ij->op = tensor->op;
        لكل (صحيح i = 0; i < GGML_MAX_DIMS; i++) {
            t_ij->nb[i] = nb[i];
        }
        t_ij->flags = tensor->flags;
        کپی_حافظه(t_ij->op_params, tensor->op_params, حجم(tensor->op_params));
        ggml_مجموعه_name(t_ij, tensor->name);
        t_ij->میانگیر = simple_buf;
        t_ij->view_src = tensor->view_src;
        t_ij->view_offs = tensor->view_offs;
        إذا (t_ij->view_src != اشاره_تهی && ggml_backend_میانگیر_is_meta(t_ij->view_src->میانگیر)) {
            t_ij->view_src = ggml_backend_meta_میانگیر_simple_tensor(tensor->view_src, j);
            إذا (t_ij->view_offs > 0 && تقسیم_dim >= 0 && تقسیم_dim < GGML_MAX_DIMS) {
                GGML_ASSERT(tensor->ne[تقسیم_dim] != 0);
                ثابت صحيح تقسیم_dim_view_src = ggml_backend_meta_get_تقسیم_state(tensor->view_src, /*assume_sync =*/ درست).axis;
                GGML_ASSERT(تقسیم_dim_view_src >= 0 && تقسیم_dim_view_src < GGML_MAX_DIMS);

                // The offset can be internal to the data split, in those cases the view offset should not be scaled.
                // If however, the offset is larger than the data split then it needs to be scaled proportionally.
                منطقي تقسیم_داخلی_جابه_جایی = t_ij->view_offs <= tensor->view_src->nb[تقسیم_dim_view_src];
                لكل (صحيح i = 0; i < GGML_MAX_DIMS; i++) {
                    ثابت size_t dim_اندازه = tensor->ne[i] * tensor->nb[i];
                    إذا (tensor->view_offs <= dim_اندازه && dim_اندازه < tensor->nb[تقسیم_dim]) {
                        تقسیم_داخلی_جابه_جایی = درست;
                        اكسر;
                    }
                }
                إذا (!تقسیم_داخلی_جابه_جایی) {
                    t_ij->view_offs = t_ij->view_offs * ne[تقسیم_dim]/tensor->ne[تقسیم_dim];
                }
            }
        }
        // TODO: revisit once the graph allocator has been refactored, see https://github.com/ggml-org/llama.cpp/pull/25051#issuecomment-4842873396
        ggml_backend_میانگیر_t init_buf = simple_buf;
        إذا (t_ij->view_src != اشاره_تهی) {
            t_ij->داده = (حرف *) t_ij->view_src->داده + t_ij->view_offs;
            // views inherit the source slice's concrete sub-buffer (issue 22197)
            إذا (tensor->view_src != اشاره_تهی && ggml_backend_میانگیر_is_meta(tensor->view_src->میانگیر)
                    && t_ij->view_src->میانگیر != اشاره_تهی) {
                t_ij->میانگیر = t_ij->view_src->میانگیر;
                init_buf     = t_ij->view_src->میانگیر;
            }
        } وإلا إذا (simple_buf != اشاره_تهی) {
            إذا (ggml_backend_میانگیر_is_multi_میانگیر(simple_buf)) {
                GGML_ABORT("multi buffers are not supported by the meta backend");
            }
            t_ij->داده = (حرف *) ggml_backend_میانگیر_get_پایه(simple_buf)
                + size_t(tensor->داده) - size_t(ggml_backend_میانگیر_get_پایه(tensor->میانگیر));
        }

        إذا (init_buf) {
            // the backend that owns the buffer will set .extra
            ggml_backend_میانگیر_init_tensor(init_buf, t_ij);
        } وإلا {
            t_ij->extra = tensor->extra;
        }

        لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
            t_ij->src[i] = tensor->src[i];
            إذا (tensor->src[i] == tensor) {
                t_ij->src[i] = t_ij;
            } وإلا إذا (t_ij->src[i] != اشاره_تهی && ggml_backend_میانگیر_is_meta(t_ij->src[i]->میانگیر)) {
                t_ij->src[i] = ggml_backend_meta_میانگیر_simple_tensor(tensor->src[i], j);
            }
        }

        simple_tensors.push_back(t_ij);
    }

    // If one of the sources has a zero-sized slice, disable the computation:
    لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
        إذا (tensor->src[i] == اشاره_تهی || !ggml_backend_میانگیر_is_meta(tensor->src[i]->میانگیر)) {
            استمر;
        }

        ثابت ggml_backend_meta_تقسیم_state تقسیم_state_src = ggml_backend_meta_get_تقسیم_state(tensor->src[i], /*assume_sync =*/ درست);
        إذا (تقسیم_state_src.axis < 0 || تقسیم_state_src.axis >= GGML_MAX_DIMS) {
            استمر;
        }
        لكل (size_t j = 0; j < n_simple_bufs; j++) {
            int64_t ne_جمع = 0;
            لكل (size_t s = 0; s < تقسیم_state_src.n_segments; s++) {
                ne_جمع += تقسیم_state_src.ne[s*n_simple_bufs + j] * تقسیم_state_src.nr[s];
            }
            إذا (ne_جمع == 0) {
                simple_tensors[j]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
            }
        }
    }

    stc.simple_tensors[tensor] = simple_tensors;

    أرجع GGML_وضعیت_SUCCESS;
}

ساكن تعداد ggml_وضعیت ggml_backend_meta_میانگیر_init_tensor(ggml_backend_میانگیر_t میانگیر, ggml_tensor * tensor) {
    GGML_ASSERT(ggml_backend_میانگیر_is_meta(میانگیر));
    ggml_backend_meta_میانگیر_context * buf_ctx = (ggml_backend_meta_میانگیر_context *) میانگیر->context;
    buf_ctx->stc_compute_شاخص = buf_ctx->stc_compute_شاخص_بعدی;
    أرجع ggml_backend_meta_میانگیر_init_tensor_impl(buf_ctx->get_simple_tensor_container(tensor), tensor);
}

ساكن فراغ ggml_backend_meta_میانگیر_memset_tensor(
        ggml_backend_میانگیر_t میانگیر, ggml_tensor * tensor, uint8_t value, size_t جابه_جایی, size_t size) {
    ثابت size_t n_bufs = ggml_backend_meta_میانگیر_n_bufs(میانگیر);
    ثابت ggml_backend_meta_تقسیم_state تقسیم_state =
            ggml_backend_meta_get_تقسیم_state(tensor, /*assume_sync =*/ خطأ);
    GGML_ASSERT(ggml_is_contiguous(tensor) || تقسیم_state.axis == GGML_BACKEND_تقسیم_AXIS_MIRRORED);

    إذا (تقسیم_state.n_segments != 1 || تقسیم_state.nr[0] != 1) {
        GGML_ASSERT(تقسیم_state.axis >= 0 && تقسیم_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(تقسیم_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        std::vector<size_t> simple_offsets(n_bufs, 0);
        إذا (تقسیم_state.axis == GGML_BACKEND_تقسیم_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            ثابت size_t ردیف_stride = tensor->nb[1];
            GGML_ASSERT(جابه_جایی % ردیف_stride == 0);
            GGML_ASSERT(size   % ردیف_stride == 0);
            ثابت int64_t ردیف_شروع = جابه_جایی / ردیف_stride;
            ثابت int64_t ردیف_شمار = size   / ردیف_stride;
            GGML_ASSERT(ردیف_شروع + ردیف_شمار <= tensor->ne[1]);

            ثابت int64_t blck_اندازه = ggml_blck_اندازه(tensor->type);
            لكل (size_t s = 0; s < تقسیم_state.n_segments; s++) {
                لكل (size_t r = 0; r < تقسیم_state.nr[s]; r++) {
                    لكل (size_t j = 0; j < n_bufs; j++) {
                        ggml_tensor * simple_tensor = ggml_backend_meta_میانگیر_simple_tensor(tensor, j);
                        GGML_ASSERT(تقسیم_state.ne[s*n_bufs + j] % blck_اندازه == 0);
                        ثابت size_t nbytes = تقسیم_state.ne[s*n_bufs + j]/blck_اندازه * tensor->nb[0];
                        لكل (int64_t ردیف = 0; ردیف < ردیف_شمار; ردیف++) {
                            ggml_backend_tensor_memset(simple_tensor, value,
                                    simple_offsets[j] + (ردیف_شروع + ردیف)*simple_tensor->nb[1], nbytes);
                        }
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            أرجع;
        }

        GGML_ASSERT(تقسیم_state.axis == GGML_BACKEND_تقسیم_AXIS_1);

        ثابت size_t ردیف_stride = tensor->nb[2];
        GGML_ASSERT(جابه_جایی % ردیف_stride == 0);
        GGML_ASSERT(size   % ردیف_stride == 0);
        ثابت int64_t ردیف_شروع = جابه_جایی / ردیف_stride;
        ثابت int64_t ردیف_شمار = size   / ردیف_stride;
        GGML_ASSERT(ردیف_شروع + ردیف_شمار <= tensor->ne[2]);

        لكل (size_t s = 0; s < تقسیم_state.n_segments; s++) {
            لكل (size_t r = 0; r < تقسیم_state.nr[s]; r++) {
                لكل (size_t j = 0; j < n_bufs; j++) {
                    ggml_tensor * simple_tensor = ggml_backend_meta_میانگیر_simple_tensor(tensor, j);
                    ثابت size_t nbytes = تقسیم_state.ne[s*n_bufs + j] * tensor->nb[1];
                    لكل (int64_t ردیف = 0; ردیف < ردیف_شمار; ردیف++) {
                        ggml_backend_tensor_memset(simple_tensor, value,
                                simple_offsets[j] + (ردیف_شروع + ردیف)*simple_tensor->nb[2], nbytes);
                    }
                    simple_offsets[j] += nbytes;
                }
            }
        }
        أرجع;
    }

    بدل (تقسیم_state.axis) {
        حالة GGML_BACKEND_تقسیم_AXIS_0:
        حالة GGML_BACKEND_تقسیم_AXIS_1:
        حالة GGML_BACKEND_تقسیم_AXIS_2: {
            ثابت size_t chunk_اندازه_پر = tensor->nb[تقسیم_state.axis + 1];
            GGML_ASSERT(جابه_جایی % chunk_اندازه_پر == 0);
            GGML_ASSERT(size   % chunk_اندازه_پر == 0);
            ثابت int64_t i_شروع =  جابه_جایی        / chunk_اندازه_پر;
            ثابت int64_t i_stop  = (جابه_جایی + size) / chunk_اندازه_پر;
            لكل (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_میانگیر_simple_tensor(tensor, j);
                ثابت size_t chunk_اندازه = simple_tensor->nb[تقسیم_state.axis + 1];
                إذا (chunk_اندازه == 0) {
                    استمر;
                }
                لكل (int64_t i = i_شروع; i < i_stop; i++) {
                    ggml_backend_tensor_memset(simple_tensor, value, i*chunk_اندازه, chunk_اندازه);
                }
            }
        } اكسر;
        حالة GGML_BACKEND_تقسیم_AXIS_PARTIAL: {
            GGML_ASSERT(value == 0);
            [[fallthrough]];
        }
        حالة GGML_BACKEND_تقسیم_AXIS_MIRRORED: {
            لكل (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_میانگیر_simple_tensor(tensor, j);
                ggml_backend_tensor_memset(simple_tensor, value, جابه_جایی, size);
            }
        } اكسر;
        افتراضي: {
            GGML_ABORT("fatal error");
        }
    }
}

ساكن فراغ ggml_backend_meta_میانگیر_مجموعه_tensor(ggml_backend_میانگیر_t میانگیر, ggml_tensor * tensor, ثابت فراغ * داده, size_t جابه_جایی, size_t size) {
    ثابت size_t n_bufs = ggml_backend_meta_میانگیر_n_bufs(میانگیر);
    ثابت ggml_backend_meta_تقسیم_state تقسیم_state = ggml_backend_meta_get_تقسیم_state(tensor, /*assume_sync =*/ خطأ);
    GGML_ASSERT(ggml_is_contiguous(tensor) || تقسیم_state.axis == GGML_BACKEND_تقسیم_AXIS_MIRRORED);

    إذا (تقسیم_state.n_segments != 1 || تقسیم_state.nr[0] != 1) {
        GGML_ASSERT(تقسیم_state.axis >= 0 && تقسیم_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(تقسیم_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        size_t جابه_جایی_داده = 0;
        std::vector<size_t> simple_offsets(n_bufs, 0);
        إذا (تقسیم_state.axis == GGML_BACKEND_تقسیم_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            ثابت size_t ردیف_stride = tensor->nb[1];
            GGML_ASSERT(جابه_جایی % ردیف_stride == 0);
            GGML_ASSERT(size   % ردیف_stride == 0);
            ثابت int64_t ردیف_شروع = جابه_جایی / ردیف_stride;
            ثابت int64_t ردیف_شمار = size   / ردیف_stride;
            GGML_ASSERT(ردیف_شروع + ردیف_شمار <= tensor->ne[1]);

            ثابت int64_t blck_اندازه = ggml_blck_اندازه(tensor->type);
            لكل (size_t s = 0; s < تقسیم_state.n_segments; s++) {
                لكل (size_t r = 0; r < تقسیم_state.nr[s]; r++) {
                    لكل (size_t j = 0; j < n_bufs; j++) {
                        ggml_tensor * simple_tensor = ggml_backend_meta_میانگیر_simple_tensor(tensor, j);
                        GGML_ASSERT(تقسیم_state.ne[s*n_bufs + j] % blck_اندازه == 0);
                        ثابت size_t nbytes = تقسیم_state.ne[s*n_bufs + j]/blck_اندازه * tensor->nb[0];
                        ggml_backend_tensor_مجموعه_2d(simple_tensor, (ثابت حرف *) داده + جابه_جایی_داده,
                            simple_offsets[j] + ردیف_شروع * simple_tensor->nb[1], nbytes,
                            ردیف_شمار, simple_tensor->nb[1], tensor->nb[1]);
                        جابه_جایی_داده       += nbytes;
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            GGML_ASSERT(جابه_جایی_داده*ردیف_شمار == size);
            أرجع;
        }
        GGML_ASSERT(تقسیم_state.axis == GGML_BACKEND_تقسیم_AXIS_1);

        ثابت size_t ردیف_stride = tensor->nb[2];
        GGML_ASSERT(جابه_جایی % ردیف_stride == 0);
        GGML_ASSERT(size   % ردیف_stride == 0);
        ثابت int64_t ردیف_شروع = جابه_جایی / ردیف_stride;
        ثابت int64_t ردیف_شمار = size   / ردیف_stride;
        GGML_ASSERT(ردیف_شروع + ردیف_شمار <= tensor->ne[2]);

        لكل (size_t s = 0; s < تقسیم_state.n_segments; s++) {
            لكل (size_t r = 0; r < تقسیم_state.nr[s]; r++) {
                لكل (size_t j = 0; j < n_bufs; j++) {
                    ggml_tensor * simple_tensor = ggml_backend_meta_میانگیر_simple_tensor(tensor, j);
                    ثابت size_t nbytes = تقسیم_state.ne[s*n_bufs + j] * tensor->nb[1];
                    ggml_backend_tensor_مجموعه_2d(simple_tensor, (ثابت حرف *) داده + جابه_جایی_داده,
                        simple_offsets[j] + ردیف_شروع * simple_tensor->nb[2], nbytes,
                        ردیف_شمار, simple_tensor->nb[2], tensor->nb[2]);
                    جابه_جایی_داده       += nbytes;
                    simple_offsets[j] += nbytes;
                }
            }
        }
        GGML_ASSERT(جابه_جایی_داده*ردیف_شمار == size);
        أرجع;
    }

    بدل (تقسیم_state.axis) {
        حالة GGML_BACKEND_تقسیم_AXIS_0:
        حالة GGML_BACKEND_تقسیم_AXIS_1:
        حالة GGML_BACKEND_تقسیم_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            ثابت size_t chunk_اندازه_پر = tensor->nb[تقسیم_state.axis + 1];
            GGML_ASSERT(جابه_جایی % chunk_اندازه_پر == 0);
            GGML_ASSERT(size   % chunk_اندازه_پر == 0);
            ثابت int64_t i_شروع =  جابه_جایی        /chunk_اندازه_پر;
            ثابت int64_t i_stop  = (جابه_جایی + size)/chunk_اندازه_پر;
            size_t جابه_جایی_j = 0;
            لكل (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_میانگیر_simple_tensor(tensor, j);
                ثابت size_t chunk_اندازه_j = simple_tensor->nb[تقسیم_state.axis + 1];
                إذا (chunk_اندازه_j == 0) {
                    استمر;
                }
                ثابت size_t simple_جابه_جایی = i_شروع * chunk_اندازه_j;
                ggml_backend_tensor_مجموعه_2d(simple_tensor, (ثابت حرف *) داده + جابه_جایی_j, simple_جابه_جایی, chunk_اندازه_j, i_stop - i_شروع, chunk_اندازه_j, chunk_اندازه_پر);
                جابه_جایی_j += chunk_اندازه_j;
            }
            GGML_ASSERT(جابه_جایی_j == chunk_اندازه_پر);
        } اكسر;
        حالة GGML_BACKEND_تقسیم_AXIS_MIRRORED: {
            لكل (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_میانگیر_simple_tensor(tensor, j);
                ggml_backend_tensor_مجموعه(simple_tensor, داده, جابه_جایی, size);
            }
        } اكسر;
        حالة GGML_BACKEND_تقسیم_AXIS_PARTIAL: {
            GGML_ASSERT(tensor->type == GGML_TYPE_F32);
            GGML_ASSERT(جابه_جایی % حجم(عائم) == 0);
            GGML_ASSERT(size   % حجم(عائم) == 0);
            ثابت size_t n_مقادیر = size / حجم(عائم);
            size_t n_contributors = 0;
            لكل (size_t j = 0; j < n_bufs; j++) {
                n_contributors += تقسیم_state.ne[j] != 0;
            }
            ثابت منطقي has_contributor_ماسک = n_contributors != 0;
            إذا (!has_contributor_ماسک) {
                n_contributors = n_bufs;
            }
            std::vector<عائم> tmp(n_مقادیر);
            لكل (size_t i = 0; i < n_مقادیر; i++) {
                tmp[i] = ((ثابت عائم *) داده)[i] / n_contributors;
            }
            std::vector<عائم> zero;
            إذا (has_contributor_ماسک) {
                zero.resize(n_مقادیر, 0.0f);
            }
            لكل (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_میانگیر_simple_tensor(tensor, j);
                ثابت عائم * partial = has_contributor_ماسک && تقسیم_state.ne[j] == 0 ? zero.داده() : tmp.داده();
                ggml_backend_tensor_مجموعه(simple_tensor, partial, جابه_جایی, size);
            }
        } اكسر;
        افتراضي: {
            GGML_ABORT("fatal error");
        }
    }
}

ساكن فراغ ggml_backend_meta_میانگیر_get_tensor(ggml_backend_میانگیر_t میانگیر, ثابت ggml_tensor * tensor, فراغ * داده, size_t جابه_جایی, size_t size) {
    ثابت size_t n_bufs = ggml_backend_meta_میانگیر_n_bufs(میانگیر);
    ثابت ggml_backend_meta_تقسیم_state تقسیم_state = ggml_backend_meta_get_تقسیم_state(tensor, /*assume_sync =*/ خطأ);
    GGML_ASSERT(ggml_is_contiguous(tensor) || تقسیم_state.axis == GGML_BACKEND_تقسیم_AXIS_MIRRORED);

    إذا (تقسیم_state.n_segments != 1 || تقسیم_state.nr[0] != 1) {
        GGML_ASSERT(تقسیم_state.axis >= 0 && تقسیم_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(تقسیم_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        size_t جابه_جایی_داده = 0;
        std::vector<size_t> simple_offsets(n_bufs, 0);
        إذا (تقسیم_state.axis == GGML_BACKEND_تقسیم_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            ثابت size_t ردیف_stride = tensor->nb[1];
            GGML_ASSERT(جابه_جایی % ردیف_stride == 0);
            GGML_ASSERT(size   % ردیف_stride == 0);
            ثابت int64_t ردیف_شروع = جابه_جایی / ردیف_stride;
            ثابت int64_t ردیف_شمار = size   / ردیف_stride;
            GGML_ASSERT(ردیف_شروع + ردیف_شمار <= tensor->ne[1]);

            ثابت int64_t blck_اندازه = ggml_blck_اندازه(tensor->type);
            لكل (size_t s = 0; s < تقسیم_state.n_segments; s++) {
                لكل (size_t r = 0; r < تقسیم_state.nr[s]; r++) {
                    لكل (size_t j = 0; j < n_bufs; j++) {
                        ثابت ggml_tensor * simple_tensor = ggml_backend_meta_میانگیر_simple_tensor(tensor, j);
                        GGML_ASSERT(تقسیم_state.ne[s*n_bufs + j] % blck_اندازه == 0);
                        ثابت size_t nbytes = تقسیم_state.ne[s*n_bufs + j]/blck_اندازه * tensor->nb[0];
                        ggml_backend_tensor_get_2d(simple_tensor, (حرف *) داده + جابه_جایی_داده,
                            simple_offsets[j] + ردیف_شروع * simple_tensor->nb[1], nbytes,
                            ردیف_شمار, simple_tensor->nb[1], tensor->nb[1]);
                        جابه_جایی_داده       += nbytes;
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            GGML_ASSERT(جابه_جایی_داده*ردیف_شمار == size);
            أرجع;
        }
        GGML_ASSERT(تقسیم_state.axis == GGML_BACKEND_تقسیم_AXIS_1);

        ثابت size_t ردیف_stride = tensor->nb[2];
        GGML_ASSERT(جابه_جایی % ردیف_stride == 0);
        GGML_ASSERT(size   % ردیف_stride == 0);
        ثابت int64_t ردیف_شروع = جابه_جایی / ردیف_stride;
        ثابت int64_t ردیف_شمار = size   / ردیف_stride;
        GGML_ASSERT(ردیف_شروع + ردیف_شمار <= tensor->ne[2]);

        لكل (size_t s = 0; s < تقسیم_state.n_segments; s++) {
            لكل (size_t r = 0; r < تقسیم_state.nr[s]; r++) {
                لكل (size_t j = 0; j < n_bufs; j++) {
                    ثابت ggml_tensor * simple_tensor = ggml_backend_meta_میانگیر_simple_tensor(tensor, j);
                    ثابت size_t nbytes = تقسیم_state.ne[s*n_bufs + j] * tensor->nb[1];
                    ggml_backend_tensor_get_2d(simple_tensor, (حرف *) داده + جابه_جایی_داده,
                        simple_offsets[j] + ردیف_شروع * simple_tensor->nb[2], nbytes,
                        ردیف_شمار, simple_tensor->nb[2], tensor->nb[2]);
                    جابه_جایی_داده       += nbytes;
                    simple_offsets[j] += nbytes;
                }
            }
        }
        GGML_ASSERT(جابه_جایی_داده*ردیف_شمار == size);
        أرجع;
    }

    بدل (تقسیم_state.axis) {
        حالة GGML_BACKEND_تقسیم_AXIS_0:
        حالة GGML_BACKEND_تقسیم_AXIS_1:
        حالة GGML_BACKEND_تقسیم_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            ثابت size_t chunk_اندازه_پر = tensor->nb[تقسیم_state.axis + 1];
            GGML_ASSERT(جابه_جایی % chunk_اندازه_پر == 0);
            GGML_ASSERT(size   % chunk_اندازه_پر == 0);
            ثابت int64_t i_شروع =  جابه_جایی        /chunk_اندازه_پر;
            ثابت int64_t i_stop  = (جابه_جایی + size)/chunk_اندازه_پر;
            size_t جابه_جایی_j = 0;
            لكل (size_t j = 0; j < n_bufs; j++){
                ثابت ggml_tensor * simple_tensor = ggml_backend_meta_میانگیر_simple_tensor(tensor, j);
                ثابت size_t chunk_اندازه_j = simple_tensor->nb[تقسیم_state.axis + 1];
                إذا (chunk_اندازه_j == 0) {
                    استمر;
                }
                ثابت size_t simple_جابه_جایی = i_شروع * chunk_اندازه_j;
                ggml_backend_tensor_get_2d(simple_tensor, (حرف *) داده + جابه_جایی_j, simple_جابه_جایی, chunk_اندازه_j, i_stop - i_شروع, chunk_اندازه_j, chunk_اندازه_پر);
                جابه_جایی_j += chunk_اندازه_j;
            }
            GGML_ASSERT(جابه_جایی_j == chunk_اندازه_پر);
        } اكسر;
        حالة GGML_BACKEND_تقسیم_AXIS_MIRRORED: {
            // TODO other simple backend may be better
            ثابت ggml_tensor * simple_tensor = ggml_backend_meta_میانگیر_simple_tensor(tensor, 0);
            ggml_backend_tensor_get(simple_tensor, داده, جابه_جایی, size);
        } اكسر;
        افتراضي: {
            GGML_ABORT("fatal error");
        }
    }
}

ساكن فراغ ggml_backend_meta_میانگیر_clear(ggml_backend_میانگیر_t میانگیر, uint8_t value) {
    ثابت size_t n_buffers = ggml_backend_meta_میانگیر_n_bufs(میانگیر);
    لكل (size_t i = 0; i < n_buffers; i++) {
        ggml_backend_میانگیر_clear(ggml_backend_meta_میانگیر_simple_میانگیر(میانگیر, i), value);
    }
}

ساكن فراغ ggml_backend_meta_میانگیر_reset(ggml_backend_میانگیر_t میانگیر) {
    GGML_ASSERT(ggml_backend_میانگیر_is_meta(میانگیر));
    ggml_backend_meta_میانگیر_context * buf_ctx = (ggml_backend_meta_میانگیر_context *) میانگیر->context;
    لكل (size_t i = 0; i < buf_ctx->bufs.size(); i++) {
        ggml_backend_میانگیر_reset(ggml_backend_meta_میانگیر_simple_میانگیر(میانگیر, i));
    }
}

ساكن ثابت ggml_backend_میانگیر_i ggml_backend_meta_میانگیر_iface = {
    /* .free_buffer     = */ ggml_backend_meta_میانگیر_free_میانگیر,
    /* .get_base        = */ ggml_backend_meta_میانگیر_get_پایه,
    /* .init_tensor     = */ ggml_backend_meta_میانگیر_init_tensor,
    /* .memset_tensor   = */ ggml_backend_meta_میانگیر_memset_tensor,
    /* .set_tensor      = */ ggml_backend_meta_میانگیر_مجموعه_tensor,
    /* .get_tensor      = */ ggml_backend_meta_میانگیر_get_tensor,
    /* .set_tensor_2d   = */ اشاره_تهی,
    /* .get_tensor_2d   = */ اشاره_تهی,
    /* .cpy_tensor      = */ اشاره_تهی,
    /* .clear           = */ ggml_backend_meta_میانگیر_clear,
    /* .reset           = */ ggml_backend_meta_میانگیر_reset,
};

منطقي ggml_backend_میانگیر_is_meta(ggml_backend_میانگیر_t buf) {
    أرجع buf != اشاره_تهی && buf->iface.free_میانگیر == ggml_backend_meta_میانگیر_iface.free_میانگیر;
}

فراغ ggml_backend_meta_میانگیر_مجموعه_usage(ggml_backend_میانگیر_t میانگیر, تعداد ggml_backend_میانگیر_usage usage) {
    GGML_ASSERT(ggml_backend_میانگیر_is_meta(میانگیر));
    ggml_backend_meta_میانگیر_context * buf_ctx = (ggml_backend_meta_میانگیر_context *) میانگیر->context;
    لكل (size_t i = 0; i < buf_ctx->bufs.size(); i++) {
        إذا (buf_ctx->bufs[i]) {
            ggml_backend_میانگیر_مجموعه_usage(buf_ctx->bufs[i].get(), usage);
        }
    }
}

ساكن ggml_backend_میانگیر_t ggml_backend_meta_میانگیر_type_alloc_میانگیر(ggml_backend_میانگیر_type_t buft, size_t size) {
    ثابت size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);

    ثابت ggml_init_params params = {
        /*.mem_size   =*/ 1024*1024*ggml_tensor_overhead(), // FIXME
        /*.mem_buffer =*/ اشاره_تهی,
        /*.no_alloc   =*/ درست,
    };
    ggml_backend_meta_simple_tensor_container stc_static;
    ggml_backend_meta_simple_tensor_container stc_compute_0(params, n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_1(params, n_simple_bufts);

    size_t max_size = 0;
    std::vector<ggml_backend_میانگیر_t> bufs;
    bufs.reserve(n_simple_bufts);
    لكل (size_t i = 0; i < n_simple_bufts; i++) {
        bufs.push_back(ggml_backend_buft_alloc_میانگیر(ggml_backend_meta_buft_simple_buft(buft, i), size));
        GGML_ASSERT(bufs.back() != اشاره_تهی);
        max_size = std::max(max_size, ggml_backend_میانگیر_get_اندازه(bufs.back()));
    }
    ggml_backend_meta_میانگیر_context * buf_ctx = جديد ggml_backend_meta_میانگیر_context(stc_static, stc_compute_0, stc_compute_1, bufs);

    أرجع ggml_backend_میانگیر_init(buft, ggml_backend_meta_میانگیر_iface, buf_ctx, max_size);
}

ساكن ggml_backend_میانگیر_t ggml_backend_meta_میانگیر_type_alloc_میانگیر_n(ggml_backend_میانگیر_type_t buft, ggml_tensor ** tensors, صحيح n_tensors) {
    ثابت size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);

    constexpr size_t compute_headroom = 16; // Maximum number of views per statically allocated tensor that can be created between evals.
    ثابت ggml_init_params params_static = {
        /*.mem_size   =*/ n_tensors * ggml_tensor_overhead(),
        /*.mem_buffer =*/ اشاره_تهی,
        /*.no_alloc   =*/ درست,
    };
    ثابت ggml_init_params params_compute = {
        /*.mem_size   =*/ compute_headroom * n_tensors * ggml_tensor_overhead(),
        /*.mem_buffer =*/ اشاره_تهی,
        /*.no_alloc   =*/ درست,
    };
    ggml_backend_meta_simple_tensor_container stc_static   (params_static,  n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_0(params_compute, n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_1(params_compute, n_simple_bufts);

    std::vector<ggml_backend_میانگیر_t> bufs(n_simple_bufts, اشاره_تهی);
    ggml_backend_meta_میانگیر_context * meta_buf_ctx = جديد ggml_backend_meta_میانگیر_context(stc_static, stc_compute_0, stc_compute_1, bufs);

    ggml_backend_میانگیر_t meta_buf = ggml_backend_میانگیر_init(buft, ggml_backend_meta_میانگیر_iface, meta_buf_ctx, 0);
    لكل (صحيح i = 0; i < n_tensors; i++) {
        ggml_tensor * t = tensors[i];
        t->میانگیر = meta_buf;
        ggml_backend_meta_میانگیر_init_tensor_impl(meta_buf_ctx->stc_static, t);
        t->داده = (فراغ *) 0x2000000000000000; // FIXME
    }
    لكل (size_t i = 0; i < n_simple_bufts; i++) {
        ggml_context * ctx = meta_buf_ctx->stc_static.ctxs[i].get();
        ggml_backend_میانگیر_type_t simple_buft = ggml_backend_meta_buft_simple_buft(buft, i);

        // If a ggml_context only has zero-sized tensors, ggml_backend_alloc_ctx_tensors_from_buft returns NULL.
        // For those edge cases, allocate a dummy buffer instead.
        منطقي any_nonzero_برش = خطأ;
        لكل (ggml_tensor * t = ggml_get_اولی_tensor(ctx); t != اشاره_تهی; t = ggml_get_بعدی_tensor(ctx, t)) {
            إذا (ggml_nelements(t) != 0) {
                any_nonzero_برش = درست;
                اكسر;
            }
        }
        إذا (any_nonzero_برش) {
            meta_buf_ctx->bufs[i].reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx, simple_buft));
        } وإلا {
            meta_buf_ctx->bufs[i].reset(ggml_backend_buft_alloc_میانگیر(simple_buft, 0));
            لكل (ggml_tensor * t = ggml_get_اولی_tensor(ctx); t != اشاره_تهی; t = ggml_get_بعدی_tensor(ctx, t)) {
                t->میانگیر = meta_buf_ctx->bufs[i].get();
            }
        }
        GGML_ASSERT(meta_buf_ctx->bufs[i]);
        meta_buf->size = std::max(meta_buf->size, ggml_backend_میانگیر_get_اندازه(meta_buf_ctx->bufs[i].get()));
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
        ggml_cgraph * cgraph_main = اشاره_تهی;
        صحيح           جابه_جایی      = 0; // Node offset vs. original graph

        std::vector<ggml_cgraph *> cgraphs_aux;
    };
    بنية backend_config {
        ggml_backend_t backend;

        std::vector<cgraph_config>           cgraphs;
        std::vector<ggml_tensor *>           nodes;
        std::vector<ggml_backend_میانگیر_ptr> bufs;

        backend_config(ggml_backend_t backend, ثابت size_t n_reduce_گامها) : backend(backend) {
            bufs.resize(n_reduce_گامها);
        }
    };
    std::string                 name;
    std::vector<backend_config> backend_configs;
    ggml_context_ptr            ctx;
    std::vector<ggml_cgraph *>  cgraphs_aux;
    std::vector<ggml_tensor *>  nodes_aux;
    size_t                      n_reduce_گامها;
    صحيح                         max_nnodes    = 0;
    size_t                      max_tmp_اندازه  = 0;
    size_t                      max_subgraphs = 0;
    size_t                      n_subgraphs   = 0;
    uint64_t                    uid           = 0;

    فراغ *                               comm_ctx       = اشاره_تهی;
    ggml_backend_comm_allreduce_tensor_t comm_allreduce = اشاره_تهی;

    ggml_backend_meta_context(ggml_backend_dev_t meta_dev, ثابت حرف * params) {
        ثابت size_t n_devs = ggml_backend_meta_dev_n_devs(meta_dev);
        n_reduce_گامها = std::ceil(std::log2(n_devs));
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
            backend_configs.emplace_back(simple_backends.back(), n_reduce_گامها);
        }
        name += ")";

        إذا (n_devs > 1) {
            ggml_backend_comm_init_t comm_init = (ggml_backend_comm_init_t) ggml_backend_reg_get_proc_آدرس(
                ggml_backend_dev_backend_reg(ggml_backend_get_device(simple_backends[0])), "ggml_backend_comm_init");
            إذا (comm_init != اشاره_تهی) {
                comm_ctx = comm_init(simple_backends.داده(), simple_backends.size());
            }
        }
        إذا (comm_ctx != اشاره_تهی) {
            comm_allreduce = (ggml_backend_comm_allreduce_tensor_t)
                ggml_backend_reg_get_proc_آدرس(ggml_backend_dev_backend_reg(
                    ggml_backend_get_device(simple_backends[0])), "ggml_backend_comm_allreduce_tensor");
            GGML_ASSERT(comm_allreduce != اشاره_تهی);
        }
    }

    ~ggml_backend_meta_context() {
        إذا (comm_ctx != اشاره_تهی) {
            ggml_backend_comm_free_t comm_free = (ggml_backend_comm_free_t) ggml_backend_reg_get_proc_آدرس(
                ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_configs[0].backend)), "ggml_backend_comm_free");
            GGML_ASSERT(comm_free != اشاره_تهی);
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
    حذف backend_ctx;
    حذف backend;
}

ساكن فراغ ggml_backend_meta_مجموعه_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, ثابت فراغ * داده, size_t جابه_جایی, size_t size) {
    ثابت size_t n_backends = ggml_backend_meta_n_backends(backend);
    GGML_ASSERT(جابه_جایی == 0);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    ثابت ggml_backend_meta_تقسیم_state تقسیم_state = ggml_backend_meta_get_تقسیم_state(tensor, /*assume_sync =*/ خطأ);
    GGML_ASSERT(تقسیم_state.n_segments == 1);
    GGML_ASSERT(تقسیم_state.nr[0]      == 1);

    بدل (تقسیم_state.axis) {
        حالة GGML_BACKEND_تقسیم_AXIS_0:
        حالة GGML_BACKEND_تقسیم_AXIS_1:
        حالة GGML_BACKEND_تقسیم_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            ثابت size_t chunk_اندازه_پر = tensor->nb[تقسیم_state.axis + 1];
            GGML_ASSERT(جابه_جایی % chunk_اندازه_پر == 0);
            GGML_ASSERT(size   % chunk_اندازه_پر == 0);
            ثابت int64_t i_شروع =  جابه_جایی        /chunk_اندازه_پر;
            ثابت int64_t i_stop  = (جابه_جایی + size)/chunk_اندازه_پر;
            size_t جابه_جایی_j = 0;
            لكل (size_t j = 0; j < n_backends; j++){
                ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
                ggml_tensor * simple_tensor = ggml_backend_meta_میانگیر_simple_tensor(tensor, j);
                ثابت size_t chunk_اندازه_j = simple_tensor->nb[تقسیم_state.axis + 1];
                إذا (chunk_اندازه_j == 0) {
                    استمر;
                }
                ggml_backend_tensor_مجموعه_2d_async(simple_backend, simple_tensor, (ثابت حرف *) داده + جابه_جایی_j, جابه_جایی, chunk_اندازه_j,
                    i_stop - i_شروع, chunk_اندازه_j, chunk_اندازه_پر);
                جابه_جایی_j += chunk_اندازه_j;
            }
            GGML_ASSERT(جابه_جایی_j == chunk_اندازه_پر);
        } اكسر;
        حالة GGML_BACKEND_تقسیم_AXIS_MIRRORED: {
            لكل (size_t j = 0; j < n_backends; j++) {
                ggml_backend_tensor_مجموعه_async(
                    ggml_backend_meta_simple_backend(backend, j), ggml_backend_meta_میانگیر_simple_tensor(tensor, j), داده, جابه_جایی, size);
            }
        } اكسر;
        افتراضي: {
            GGML_ABORT("fatal error");
        }
    }
}

ساكن فراغ ggml_backend_meta_get_tensor_async(ggml_backend_t backend, ثابت ggml_tensor * tensor, فراغ * داده, size_t جابه_جایی, size_t size) {
    ثابت size_t n_backends = ggml_backend_meta_n_backends(backend);
    GGML_ASSERT(جابه_جایی == 0);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    ثابت ggml_backend_meta_تقسیم_state تقسیم_state = ggml_backend_meta_get_تقسیم_state(tensor, /*assume_sync =*/ خطأ);
    GGML_ASSERT(تقسیم_state.n_segments == 1);
    GGML_ASSERT(تقسیم_state.nr[0]      == 1);

    بدل (تقسیم_state.axis) {
        حالة GGML_BACKEND_تقسیم_AXIS_0:
        حالة GGML_BACKEND_تقسیم_AXIS_1:
        حالة GGML_BACKEND_تقسیم_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            ثابت size_t chunk_اندازه_پر = tensor->nb[تقسیم_state.axis + 1];
            GGML_ASSERT(جابه_جایی % chunk_اندازه_پر == 0);
            GGML_ASSERT(size   % chunk_اندازه_پر == 0);
            ثابت int64_t i_شروع =  جابه_جایی        /chunk_اندازه_پر;
            ثابت int64_t i_stop  = (جابه_جایی + size)/chunk_اندازه_پر;
            size_t جابه_جایی_j = 0;
            لكل (size_t j = 0; j < n_backends; j++){
                ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
                ثابت ggml_tensor * simple_tensor = ggml_backend_meta_میانگیر_simple_tensor(tensor, j);
                ثابت size_t chunk_اندازه_j = simple_tensor->nb[تقسیم_state.axis + 1];
                إذا (chunk_اندازه_j == 0) {
                    استمر;
                }
                ggml_backend_tensor_get_2d_async(simple_backend, simple_tensor, (حرف *) داده + جابه_جایی_j, جابه_جایی, chunk_اندازه_j,
                    i_stop - i_شروع, chunk_اندازه_j, chunk_اندازه_پر);
                جابه_جایی_j += chunk_اندازه_j;
            }
            GGML_ASSERT(جابه_جایی_j == chunk_اندازه_پر);
        } اكسر;
        حالة GGML_BACKEND_تقسیم_AXIS_MIRRORED: {
            // TODO other simple backend may be better
            ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, 0);
            ثابت ggml_tensor * simple_tensor = ggml_backend_meta_میانگیر_simple_tensor(tensor, 0);
            ggml_backend_tensor_get_async(simple_backend, simple_tensor, داده, جابه_جایی, size);
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

ساكن تعداد ggml_وضعیت ggml_backend_meta_گراف_compute(ggml_backend_t backend, بنية ggml_cgraph * cgraph) {
    GGML_ASSERT(cgraph->grads == اشاره_تهی);
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
        max_nnodes_raised = درست;
        أكد(needs_rebuild);
    }

    إذا (needs_rebuild) {
        std::set<ggml_backend_میانگیر_t> استفاده_شده_buffers;
        لكل (صحيح i = 0; i < cgraph->n_leafs; i++) {
            إذا (ggml_backend_میانگیر_is_meta(cgraph->leafs[i]->میانگیر)) {
                استفاده_شده_buffers.emplace(cgraph->leafs[i]->میانگیر);
            }
        }
        لكل (صحيح i = 0; i < cgraph->n_nodes; i++) {
            إذا (ggml_backend_میانگیر_is_meta(cgraph->nodes[i]->میانگیر)) {
                استفاده_شده_buffers.emplace(cgraph->nodes[i]->میانگیر);
            }
        }
        لكل (ggml_backend_میانگیر_t buf : استفاده_شده_buffers) {
            ggml_backend_meta_میانگیر_context * buf_ctx = (ggml_backend_meta_میانگیر_context *) buf->context;
            buf_ctx->stc_compute_شاخص_بعدی = buf_ctx->stc_compute_شاخص ^ 1;
            ggml_backend_meta_simple_tensor_container & stc = buf_ctx->stc_compute[buf_ctx->stc_compute_شاخص_بعدی];
            لكل (ggml_context_ptr & ctx : stc.ctxs) {
                ggml_reset(ctx.get());
            }
            stc.simple_tensors.clear();
        }
        size_t n_subgraphs  = 0;
        size_t max_tmp_اندازه = 0;

        لكل (size_t j = 0; j < n_backends; j++) {
            خودکار & bcj = backend_ctx->backend_configs[j];

            لكل (صحيح i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * گره = cgraph->nodes[i];
                إذا (!ggml_backend_میانگیر_is_meta(گره->میانگیر)) {
                    // [TAG_META_HOST_VIEWS]
                    GGML_ASSERT(ggml_backend_meta_is_host_view(گره));

                    // keep the node as is, mapping it to a simple tensor is not possible
                    bcj.nodes[i] = گره;
                    استمر;
                }
                bcj.nodes[i] = ggml_backend_meta_میانگیر_simple_tensor(گره, j);
                GGML_ASSERT(bcj.nodes[i]);
            }
        }

        {
            // For MoE models it may make sense to delay the AllReduce in order to reduce I/O:
            خودکار get_i_delayed_branch = [&](ثابت صحيح i) -> صحيح {
                صحيح id = i; // i_delayed
                صحيح idr = i; // i_delayed return, last safe return value

                ggml_tensor * گره = cgraph->nodes[id];
                int32_t n_استفاده_شده = ggml_گره_get_use_شمار(cgraph, id);

                // Skip MIRRORED nodes that don't consume node
                خودکار skip_unrelated = [&]() {
                    طالما (id + 1 < cgraph->n_nodes) {
                        ggml_tensor * بعدی = cgraph->nodes[id+1];
                        إذا (ggml_backend_meta_get_تقسیم_state(بعدی, خطأ).axis != GGML_BACKEND_تقسیم_AXIS_MIRRORED) {
                            اكسر;
                        }
                        منطقي امن = درست;
                        لكل (صحيح s = 0; s < GGML_MAX_SRC; s++) {
                            إذا (بعدی->src[s] == اشاره_تهی) {
                                استمر;
                            }
                            إذا (بعدی->src[s] == گره) {
                                امن = خطأ;
                                اكسر;
                            }
                            إذا (ggml_backend_meta_get_تقسیم_state(بعدی->src[s], خطأ).axis != GGML_BACKEND_تقسیم_AXIS_MIRRORED) {
                                امن = خطأ;
                                اكسر;
                            }
                        }
                        إذا (!امن) {
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
                    ggml_tensor * بعدی = cgraph->nodes[id+1];
                    إذا (بعدی->op == GGML_OP_جمع_کن_ID && بعدی->src[0] == گره &&
                            ggml_backend_meta_get_تقسیم_state(بعدی->src[1], خطأ).axis == GGML_BACKEND_تقسیم_AXIS_PARTIAL &&
                            ggml_backend_meta_get_تقسیم_state(بعدی->src[2], خطأ).axis == GGML_BACKEND_تقسیم_AXIS_MIRRORED) {
                        گره = بعدی;
                        id++;
                        idr = id;
                        n_استفاده_شده = ggml_گره_get_use_شمار(cgraph, id);
                    }
                }
                // Chain of MULs with MIRRORED src[1]
                طالما (درست) {
                    skip_unrelated();
                    إذا (id + 1 >= cgraph->n_nodes) {
                        أرجع idr;
                    }
                    ggml_tensor * بعدی = cgraph->nodes[id+1];
                    إذا (بعدی->op == GGML_OP_MUL && بعدی->src[0] == گره &&
                            ggml_backend_meta_get_تقسیم_state(بعدی->src[1], خطأ).axis == GGML_BACKEND_تقسیم_AXIS_MIRRORED) {
                        گره = بعدی;
                        id++;
                        idr = id;
                        n_استفاده_شده = ggml_گره_get_use_شمار(cgraph, id);
                    } وإلا {
                        اكسر;
                    }
                }

                إذا (n_استفاده_شده != گره->ne[1] || id + 2*n_استفاده_شده-1 >= cgraph->n_nodes) {
                    أرجع idr;
                }
                لكل (int32_t k = 0; k < n_استفاده_شده; k++) {
                    ggml_tensor * بعدی = cgraph->nodes[id+1];
                    إذا (بعدی->op != GGML_OP_VIEW || بعدی->view_src != گره || بعدی->view_offs != k*گره->nb[1] ||
                            بعدی->ne[0] != گره->ne[0] || بعدی->ne[1] != گره->ne[2] || بعدی->nb[1] != گره->nb[2] ||
                            ggml_گره_get_use_شمار(cgraph, id+1) != 1) {
                        أرجع idr;
                    }
                    id++;
                }
                {
                    ggml_tensor * بعدی = cgraph->nodes[id+1];
                    إذا (بعدی->op != GGML_OP_جمع_کن || بعدی->src[0] != cgraph->nodes[id - (n_استفاده_شده-1)] ||
                            بعدی->src[1] != cgraph->nodes[id - (n_استفاده_شده-2)] || ggml_گره_get_use_شمار(cgraph, id+1) != 1) {
                        أرجع idr;
                    }
                    id++;
                }
                لكل (int32_t k = 0; k < n_استفاده_شده - 2; k++) {
                    ggml_tensor * بعدی = cgraph->nodes[id+1];
                    إذا (بعدی->op != GGML_OP_جمع_کن || بعدی->src[0] != cgraph->nodes[id] ||
                            بعدی->src[1] != cgraph->nodes[id - (n_استفاده_شده-2)] || ggml_گره_get_use_شمار(cgraph, id+1) != 1) {
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
                ggml_tensor * گره = cgraph->nodes[i_delayed];

                إذا (ggml_گره_get_use_شمار(cgraph, i_delayed) != 1) {
                    أرجع i_delayed;
                }

                لكل (صحيح id = i_delayed + 1; id < cgraph->n_nodes; id++) {
                    ggml_tensor * بعدی = cgraph->nodes[id];
                    إذا (بعدی->view_src == گره) {
                        أرجع i_delayed;
                    }
                    لكل (صحيح s = 0; s < GGML_MAX_SRC; s++) {
                        إذا (بعدی->src[s] == گره) {
                            أرجع i_delayed;
                        }
                    }

                    إذا (بعدی->view_src != اشاره_تهی && بعدی->view_src->op == GGML_OP_NONE && ggml_backend_میانگیر_is_host(بعدی->view_src->میانگیر)) {
                        استمر;
                    }
                    إذا (ggml_backend_meta_get_تقسیم_state(بعدی, خطأ).axis != GGML_BACKEND_تقسیم_AXIS_PARTIAL) {
                        استمر;
                    }

                    ثابت صحيح i_other = id;
                    ثابت صحيح i_other_delayed = get_i_delayed_branch(i_other);
                    ggml_tensor * other = cgraph->nodes[i_other_delayed];
                    إذا (ggml_گره_get_use_شمار(cgraph, i_other_delayed) != 1 || i_other_delayed + 1 >= cgraph->n_nodes) {
                        أرجع i_delayed;
                    }

                    ggml_tensor * جمع = cgraph->nodes[i_other_delayed + 1];
                    إذا (جمع->op != GGML_OP_جمع_کن ||
                            !ggml_are_same_shape(گره, other) || گره->type != other->type || جمع->type != گره->type ||
                            !((جمع->src[0] == گره && جمع->src[1] == other) ||
                              (جمع->src[0] == other && جمع->src[1] == گره)) ||
                            ggml_backend_meta_get_تقسیم_state(جمع, خطأ).axis != GGML_BACKEND_تقسیم_AXIS_MIRRORED) {
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
                ggml_tensor * گره = cgraph->nodes[i];
                إذا (گره->view_src != اشاره_تهی && گره->view_src->op == GGML_OP_NONE && ggml_backend_میانگیر_is_host(گره->view_src->میانگیر)) {
                    استمر;
                }
                ثابت ggml_backend_meta_تقسیم_state تقسیم_state = ggml_backend_meta_get_تقسیم_state(گره, /*assume_sync =*/ خطأ);
                إذا (تقسیم_state.axis == GGML_BACKEND_تقسیم_AXIS_PARTIAL) {
                    max_tmp_اندازه = std::max(max_tmp_اندازه, ggml_nbytes(گره));
                }
                ثابت منطقي new_subgraph = i + 1 == cgraph->n_nodes || تقسیم_state.axis == GGML_BACKEND_تقسیم_AXIS_PARTIAL;
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
                    bcj.cgraphs[n_subgraphs].جابه_جایی = i_شروع;
                }
                n_subgraphs++;
                i_شروع = i + 1;
            }
            GGML_ASSERT(i_شروع == cgraph->n_nodes);
        }

        backend_ctx->uid         = cgraph->uid;
        backend_ctx->n_subgraphs = n_subgraphs;

        إذا (max_tmp_اندازه > backend_ctx->max_tmp_اندازه) {
            لكل (size_t j = 0; j < n_backends; j++) {
                خودکار & bcj = backend_ctx->backend_configs[j];
                لكل (size_t i = 0; i < backend_ctx->n_reduce_گامها; i++) {
                    bcj.bufs[i].reset(ggml_backend_alloc_میانگیر(bcj.backend, max_tmp_اندازه));
                }
            }
            backend_ctx->max_tmp_اندازه = max_tmp_اندازه;
        }

        إذا (max_nnodes_raised || n_subgraphs > backend_ctx->max_subgraphs) {
            backend_ctx->max_subgraphs = std::max(backend_ctx->max_subgraphs, n_subgraphs);
            ثابت size_t n_nodes_در_هر_device = 3 * backend_ctx->n_reduce_گامها; // tmp + ADD (+zeroing) graph per step and device
            ثابت size_t n_cgraphs_در_هر_device = 2 * backend_ctx->n_reduce_گامها; // ADD ( + zeroing) graph per step and device
            ثابت size_t mem_در_هر_device_graphs_main = backend_ctx->max_subgraphs*ggml_گراف_overhead_custom(backend_ctx->max_nnodes, cgraph->grads);
            ثابت size_t mem_در_هر_device_graphs_aux = n_cgraphs_در_هر_device*backend_ctx->max_subgraphs*ggml_گراف_overhead_custom(1, cgraph->grads);
            ثابت size_t mem_در_هر_device_nodes_aux = n_nodes_در_هر_device*backend_ctx->max_subgraphs*ggml_tensor_overhead();
            ثابت ggml_init_params params = {
                /*.mem_size   =*/ n_backends * (mem_در_هر_device_graphs_main + mem_در_هر_device_graphs_aux + mem_در_هر_device_nodes_aux),
                /*.mem_buffer =*/ اشاره_تهی,
                /*.no_alloc   =*/ درست,
            };
            backend_ctx->ctx.reset(ggml_init(params));
            لكل (size_t j = 0; j < n_backends; j++) {
                خودکار & bcj = backend_ctx->backend_configs[j];
                لكل (size_t i = 0; i < backend_ctx->max_subgraphs; i++) {
                    bcj.cgraphs[i].cgraph_main = ggml_new_گراف_custom(backend_ctx->ctx.get(), backend_ctx->max_nnodes, /*grads =*/ خطأ);
                }
            }
            backend_ctx->cgraphs_aux.resize(n_backends*n_cgraphs_در_هر_device*backend_ctx->max_subgraphs);
            لكل (size_t k = 0; k < backend_ctx->cgraphs_aux.size(); k++) {
                backend_ctx->cgraphs_aux[k] = ggml_new_گراف_custom(backend_ctx->ctx.get(), 1, cgraph->grads);
            }
            backend_ctx->nodes_aux.resize(n_backends*n_nodes_در_هر_device*backend_ctx->max_subgraphs);
            لكل (size_t k = 0; k < backend_ctx->nodes_aux.size(); k++) {
                backend_ctx->nodes_aux[k] = ggml_new_tensor_1d(backend_ctx->ctx.get(), GGML_TYPE_F32, 1);
            }
        }

        لكل (size_t j = 0; j < n_backends; j++) {
            خودکار & bcj = backend_ctx->backend_configs[j];
            لكل (size_t i_گراف = 0; i_گراف < n_subgraphs; i_گراف++) {
                ggml_cgraph * cgraph_ij = bcj.cgraphs[i_گراف].cgraph_main;
                ثابت size_t i_گره_شروع = bcj.cgraphs[i_گراف].جابه_جایی;
                ثابت size_t i_گره_stop = i_گراف + 1 < n_subgraphs ? bcj.cgraphs[i_گراف + 1].جابه_جایی : cgraph->n_nodes;
                cgraph_ij->n_nodes = i_گره_stop - i_گره_شروع;
                ggml_hash_مجموعه_reset(&cgraph_ij->دیدهشده_hash_مجموعه);
                لكل (size_t i_گره = i_گره_شروع; i_گره < i_گره_stop; i_گره++) {
                    ggml_tensor * گره_ij = bcj.nodes[i_گره];
                    cgraph_ij->nodes[i_گره - i_گره_شروع] = گره_ij;
                    ثابت size_t hash_pos_orig = ggml_hash_پیدا_کن(&cgraph->دیدهشده_hash_مجموعه, cgraph->nodes[i_گره]);
                    ثابت size_t hash_pos_ij = ggml_hash_درج(&cgraph_ij->دیدهشده_hash_مجموعه, گره_ij);
                    cgraph_ij->use_counts[hash_pos_ij] = cgraph->use_counts[hash_pos_orig];
                }
                cgraph_ij->uid = ggml_گراف_بعدی_uid();
            }
        }
    }

    size_t iga = 0; // i graph aux
    size_t ina = 0; // i node aux

    خودکار get_گره_aux = [&](ggml_tensor * t) -> ggml_tensor * {
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
    خودکار مجموعه_tmp_داده = [&](ggml_tensor * tensor, ثابت size_t j, ثابت size_t i_buf) {
        خودکار & bcj = backend_ctx->backend_configs[j];
        ggml_backend_میانگیر_ptr & buf_ptr = bcj.bufs[i_buf];
        إذا (!buf_ptr || ggml_backend_میانگیر_get_اندازه(buf_ptr.get()) < backend_ctx->max_tmp_اندازه) {
            buf_ptr.reset(ggml_backend_alloc_میانگیر(bcj.backend, backend_ctx->max_tmp_اندازه));
        }
        tensor->میانگیر = buf_ptr.get();
        tensor->داده   = ggml_backend_میانگیر_get_پایه(buf_ptr.get());
    };
    // FIXME usage_counts
    خودکار get_cgraph_aux = [&]() -> ggml_cgraph * {
        ggml_cgraph * ret = backend_ctx->cgraphs_aux[iga++];
        أرجع ret;
    };

    // Preferentially use backend-specific allreduce_tensor_async (e.g. NCCL for CUDA), use a generic fallback if unavailable:
    خودکار allreduce_fallback = [&](size_t i) -> ggml_وضعیت {
        std::vector<ggml_cgraph *> گام_cgraphs(n_backends, اشاره_تهی);

        // Zero out nodes that were disabled due to having a zero-sized slice:
        لكل (size_t j = 0; j < n_backends; j++) {
            خودکار & bcj = backend_ctx->backend_configs[j];
            ggml_tensor * گره = bcj.cgraphs[i].cgraph_main->nodes[bcj.cgraphs[i].cgraph_main->n_nodes - 1];
            إذا (گره->flags & GGML_TENSOR_FLAG_COMPUTE) {
                استمر;
            }
            إذا (ggml_nelements(گره) > 0) {
                ggml_tensor * گره_zero = get_گره_aux(گره);
                گره_zero->op = GGML_OP_FILL;
                گره_zero->src[0] = گره; // only used for the shape, the data is not read
                ggml_مجموعه_op_params_f32(گره_zero, 0, 0.0f);
                گره_zero->داده = گره->داده;
                گره_zero->میانگیر = گره->میانگیر;
                گره_zero->flags |= GGML_TENSOR_FLAG_COMPUTE;

                گام_cgraphs[j] = get_cgraph_aux();
                گام_cgraphs[j]->nodes[0] = گره_zero;
                گام_cgraphs[j]->n_nodes = 1;
                ثابت ggml_وضعیت وضعیت = ggml_backend_گراف_compute_async(bcj.backend, گام_cgraphs[j]);
                إذا (وضعیت != GGML_وضعیت_SUCCESS) {
                    أرجع وضعیت;
                }
            }
        }
        std::fill(گام_cgraphs.begin(), گام_cgraphs.end(), اشاره_تهی);

        خودکار افزودن_داده = [&](ثابت size_t j_src, ثابت size_t j_dst, ثابت size_t i_buf) {
            أكد(گام_cgraphs[j_dst] == اشاره_تهی);
            خودکار & bcj_src = backend_ctx->backend_configs[j_src];
            خودکار & bcj_dst = backend_ctx->backend_configs[j_dst];

            ggml_tensor * گره_src = bcj_src.cgraphs[i].cgraph_main->nodes[bcj_src.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_tensor * گره_dst = bcj_dst.cgraphs[i].cgraph_main->nodes[bcj_dst.cgraphs[i].cgraph_main->n_nodes - 1];
            GGML_ASSERT(ggml_is_contiguous(گره_src));
            GGML_ASSERT(ggml_is_contiguous(گره_dst));

            ggml_tensor * گره_tmp = get_گره_aux(گره_dst);
            مجموعه_tmp_داده(گره_tmp, j_dst, i_buf);

            ggml_backend_tensor_copy_async(bcj_src.backend, bcj_dst.backend, گره_src, گره_tmp);

            ggml_tensor * گره_red = get_گره_aux(گره_dst);
            گره_red->view_src = گره_dst->view_src == اشاره_تهی ? گره_dst : گره_dst->view_src;
            گره_red->view_offs = گره_dst->view_offs;
            گره_red->op = GGML_OP_جمع_کن;
            گره_red->src[0] = گره_dst;
            گره_red->src[1] = گره_tmp;
            گره_red->flags |= GGML_TENSOR_FLAG_COMPUTE;
            ggml_backend_view_init(گره_red);

            ggml_cgraph * cgraph_aux = get_cgraph_aux();
            cgraph_aux->nodes[0] = گره_red;
            cgraph_aux->n_nodes = 1;
            گام_cgraphs[j_dst] = cgraph_aux;
        };

        size_t جابه_جایی_j = n_backends/2;
        طالما ((جابه_جایی_j & (جابه_جایی_j - 1)) != 0) {
            جابه_جایی_j--;
        }
        ثابت size_t جابه_جایی_j_max = جابه_جایی_j;
        size_t i_buf = 0;

        // If n_backends is not a power of 2, fold in the excess prior to butterfly reduction:
        لكل (size_t j_src = 2*جابه_جایی_j_max; j_src < n_backends; j_src++) {
            ثابت size_t j_dst = j_src - 2*جابه_جایی_j_max;
            افزودن_داده(j_src, j_dst, i_buf);
            ثابت ggml_وضعیت وضعیت = ggml_backend_گراف_compute_async(backend_ctx->backend_configs[j_dst].backend, گام_cgraphs[j_dst]);
            إذا (وضعیت != GGML_وضعیت_SUCCESS) {
                أرجع وضعیت;
            }
            i_buf = 1;
        }

        // Butterfly reduction:
        لكل (; جابه_جایی_j >= 1; جابه_جایی_j /= 2) {
            std::fill(گام_cgraphs.begin(), گام_cgraphs.end(), اشاره_تهی);

            لكل (size_t j = 0; j < 2*جابه_جایی_j_max; j++) {
                ثابت size_t j_other = j ^ جابه_جایی_j;
                إذا (j_other >= n_backends) {
                    استمر;
                }
                افزودن_داده(j, j_other, i_buf);
            }

            لكل (size_t j = 0; j < 2*جابه_جایی_j_max; j++) {
                إذا (گام_cgraphs[j] == اشاره_تهی) {
                    استمر;
                }
                خودکار & bcj = backend_ctx->backend_configs[j];
                ثابت ggml_وضعیت وضعیت = ggml_backend_گراف_compute_async(bcj.backend, گام_cgraphs[j]);
                إذا (وضعیت != GGML_وضعیت_SUCCESS) {
                    أرجع وضعیت;
                }
            }
            i_buf++;
        }
        أكد(i_buf == backend_ctx->n_reduce_گامها);

        // If n_backends is not a power of 2, copy back the reduced tensors to the excess:
        لكل (size_t j = 2*جابه_جایی_j_max; j < n_backends; j++) {
            خودکار & bcj_src = backend_ctx->backend_configs[j - 2*جابه_جایی_j_max];
            خودکار & bcj_dst = backend_ctx->backend_configs[j];

            ggml_tensor * گره_src = bcj_src.cgraphs[i].cgraph_main->nodes[bcj_src.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_tensor * گره_dst = bcj_dst.cgraphs[i].cgraph_main->nodes[bcj_dst.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_backend_tensor_copy_async(bcj_src.backend, bcj_dst.backend, گره_src, گره_dst);
        }

        أرجع GGML_وضعیت_SUCCESS;
    };


    لكل (size_t i = 0; i < backend_ctx->n_subgraphs; i++) {
        لكل (size_t j = 0; j < n_backends; j++) {
            خودکار & bcj = backend_ctx->backend_configs[j];
            ثابت ggml_وضعیت وضعیت = ggml_backend_گراف_compute_async(bcj.backend, bcj.cgraphs[i].cgraph_main);
            إذا (وضعیت != GGML_وضعیت_SUCCESS) {
                أرجع وضعیت;
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
                backend_allreduce_success = backend_ctx->comm_allreduce(backend_ctx->comm_ctx, nodes.داده());
            }

            إذا (!backend_allreduce_success) {
                ثابت ggml_وضعیت وضعیت = allreduce_fallback(i);
                إذا (وضعیت != GGML_وضعیت_SUCCESS) {
                    أرجع وضعیت;
                }
            }
        }
    }
    أرجع GGML_وضعیت_SUCCESS;
}

ساكن ثابت ggml_backend_i ggml_backend_meta_i = {
    /* .get_name                = */ ggml_backend_meta_get_name,
    /* .free                    = */ ggml_backend_meta_free,
    /* .set_tensor_async        = */ ggml_backend_meta_مجموعه_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_meta_get_tensor_async,
    /* .set_tensor_2d_async     = */ اشاره_تهی,
    /* .get_tensor_2d_async     = */ اشاره_تهی,
    /* .cpy_tensor_async        = */ اشاره_تهی,
    /* .synchronize             = */ ggml_backend_meta_synchronize,
    /* .graph_plan_create       = */ اشاره_تهی,
    /* .graph_plan_free         = */ اشاره_تهی,
    /* .graph_plan_update       = */ اشاره_تهی,
    /* .graph_plan_compute      = */ اشاره_تهی,
    /* .graph_compute           = */ ggml_backend_meta_گراف_compute,
    /* .event_record            = */ اشاره_تهی,
    /* .event_wait              = */ اشاره_تهی,
    /* .graph_optimize          = */ اشاره_تهی,
};

منطقي ggml_backend_is_meta(ggml_backend_t backend) {
    أرجع backend != اشاره_تهی && backend->iface.get_name == ggml_backend_meta_i.get_name;
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
