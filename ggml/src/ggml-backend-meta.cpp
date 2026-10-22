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
بنية ggml_backend_meta_buffer_type;
بنية ggml_backend_meta_buffer;
بنية ggml_backend_meta;

ثابت حرف * ggml_backend_meta_split_axis_name(تعداد ggml_backend_meta_split_axis split_axis) {
    بدل (split_axis) {
        حالة GGML_BACKEND_SPLIT_AXIS_0:
            أرجع "0";
        حالة GGML_BACKEND_SPLIT_AXIS_1:
            أرجع "1";
        حالة GGML_BACKEND_SPLIT_AXIS_2:
            أرجع "2";
        حالة GGML_BACKEND_SPLIT_AXIS_3:
            أرجع "3";
        حالة GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            أرجع "MIRRORED";
        حالة GGML_BACKEND_SPLIT_AXIS_PARTIAL:
            أرجع "PARTIAL";
        حالة GGML_BACKEND_SPLIT_AXIS_NONE:
            أرجع "NONE";
        حالة GGML_BACKEND_SPLIT_AXIS_UNKNOWN:
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
    ggml_backend_meta_get_split_state_t get_split_state;
    فراغ *                              get_split_state_ud;

    std::string name;
    std::string description;

    ggml_backend_meta_device_context(
            std::vector<ggml_backend_dev_t> simple_devs, ggml_backend_meta_get_split_state_t get_split_state, فراغ * get_split_state_ud) :
            simple_devs(std::move(simple_devs)), get_split_state(get_split_state), get_split_state_ud(get_split_state_ud) {
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

    منطقي operator<(ثابت ggml_backend_meta_device_context & other) ثابت {
        أرجع std::tie(simple_devs, get_split_state, get_split_state_ud)
            < std::tie(other.simple_devs, other.get_split_state, other.get_split_state_ud);
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

ساكن فراغ ggml_backend_meta_device_get_memory(ggml_backend_dev_t dev, size_t * حرر, size_t * total) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ثابت ggml_backend_meta_device_context * meta_dev_ctx = (ثابت ggml_backend_meta_device_context *) dev->context;
    *حرر  = 0;
    *total = 0;
    لكل (ggml_backend_dev_t dev : meta_dev_ctx->simple_devs) {
        size_t tmp_free, tmp_total;
        ggml_backend_dev_memory(dev, &tmp_free, &tmp_total);
        *حرر  += tmp_free;
        *total += tmp_total;
    }
}

ساكن تعداد ggml_backend_dev_type ggml_backend_meta_device_get_type(ggml_backend_dev_t dev) {
    أرجع GGML_BACKEND_DEVICE_TYPE_META;

    GGML_UNUSED(dev);
}

ساكن فراغ ggml_backend_meta_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ثابت ggml_backend_meta_device_context * meta_dev_ctx = (ثابت ggml_backend_meta_device_context *) dev->context;

    // TODO replace placeholders
    props->name        = ggml_backend_meta_device_get_name(dev);
    props->description = ggml_backend_meta_device_get_description(dev);
    props->type        = ggml_backend_meta_device_get_type(dev);
    props->device_id   = 0;

    ggml_backend_meta_device_get_memory(dev, &props->memory_free, &props->memory_total);

    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ خطأ, // Not implemented.
        /* .buffer_from_host_ptr  = */ خطأ, // Not implemented.
        /* .events                = */ خطأ, // Not implemented.
        /* .mmap_support          = */ true,
    };
    لكل (ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
        ggml_backend_dev_props tmp_props;
        ggml_backend_dev_get_props(simple_dev, &tmp_props);
        props->caps.async                = props->caps.async                && tmp_props.caps.async;
        props->caps.host_buffer          = props->caps.host_buffer          && tmp_props.caps.host_buffer;
        props->caps.buffer_from_host_ptr = props->caps.buffer_from_host_ptr && tmp_props.caps.buffer_from_host_ptr;
        props->caps.events               = props->caps.events               && tmp_props.caps.events;
        props->caps.mmap_support         = props->caps.mmap_support         && tmp_props.caps.mmap_support;
    }
}

ساكن ggml_backend_t ggml_backend_meta_device_init_backend(ggml_backend_dev_t dev, ثابت حرف * params);

ساكن ggml_backend_buffer_type_t ggml_backend_meta_device_get_buffer_type(ggml_backend_dev_t dev);

ساكن ggml_backend_buffer_type_t ggml_backend_meta_device_get_host_buffer_type(ggml_backend_dev_t dev);

ساكن منطقي ggml_backend_meta_device_supports_op(ggml_backend_dev_t dev, ثابت ggml_tensor * op) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ثابت ggml_backend_meta_device_context * meta_dev_ctx = (ثابت ggml_backend_meta_device_context *) dev->context;
    أرجع std::all_of(meta_dev_ctx->simple_devs.begin(), meta_dev_ctx->simple_devs.end(),
        [op](ggml_backend_dev_t simple_dev) { أرجع ggml_backend_dev_supports_op(simple_dev, op); });
}

ساكن منطقي ggml_backend_meta_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
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
    أرجع true;
}

ساكن ثابت ggml_backend_device_i ggml_backend_meta_device_iface = {
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

ساكن منطقي ggml_backend_dev_is_meta(ggml_backend_dev_t dev) {
    أرجع dev != nullptr && dev->iface.get_name == ggml_backend_meta_device_iface.get_name;
}

ساكن size_t ggml_backend_meta_dev_n_devs(ggml_backend_dev_t meta_dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(meta_dev));
    ثابت ggml_backend_meta_device_context * meta_dev_ctx = (ثابت ggml_backend_meta_device_context *) meta_dev->context;
    أرجع meta_dev_ctx->simple_devs.size();
}

ساكن ggml_backend_dev_t ggml_backend_meta_dev_simple_dev(ggml_backend_dev_t meta_dev, size_t index) {
    GGML_ASSERT(ggml_backend_dev_is_meta(meta_dev));
    ثابت ggml_backend_meta_device_context * meta_dev_ctx = (ثابت ggml_backend_meta_device_context *) meta_dev->context;
    GGML_ASSERT(index < meta_dev_ctx->simple_devs.size());
    أرجع meta_dev_ctx->simple_devs[index];
}

ggml_backend_dev_t ggml_backend_meta_device(
        ggml_backend_dev_t * devs, size_t n_devs, ggml_backend_meta_get_split_state_t get_split_state, فراغ * get_split_state_ud) {
    GGML_ASSERT(n_devs <= GGML_BACKEND_META_MAX_DEVICES);
    // TODO: this is not thread-safe - needs to be fixed
    ساكن std::vector<std::unique_ptr<ggml_backend_meta_device_context>>         ctxs;
    ساكن std::map<ggml_backend_meta_device_context, بنية ggml_backend_device> meta_devs;

    std::vector<ggml_backend_dev_t> simple_devs;
    simple_devs.reserve(n_devs);
    لكل (size_t i = 0; i < n_devs; i++) {
        simple_devs.push_back(devs[i]);
    }
    ggml_backend_meta_device_context ctx(simple_devs, get_split_state, get_split_state_ud);

    {
        auto it = meta_devs.find(ctx);
        إذا (it != meta_devs.end()) {
            أرجع &it->second;
        }
    }
    ctxs.push_back(std::make_unique<ggml_backend_meta_device_context>(ctx));

    بنية ggml_backend_device meta_dev = {
        /*iface  =*/ ggml_backend_meta_device_iface,
        /*reg    =*/ nullptr,
        /*ctx    =*/ ctxs.back().get(),
    };

    auto result = meta_devs.emplace(*ctxs.back(), meta_dev);
    أرجع &result.first->second;
}

//
// meta backend buffer type
//

بنية ggml_backend_meta_buffer_type_context {
    std::vector<ggml_backend_buffer_type_t> simple_bufts;

    std::string name;

    ggml_backend_meta_buffer_type_context(std::vector<ggml_backend_buffer_type_t> simple_bufts) : simple_bufts(std::move(simple_bufts)) {
        name = "Meta(";
        لكل (size_t i = 0; i < simple_bufts.size(); i++) {
            إذا (i > 0) {
                name += ",";
            }
            name += ggml_backend_buft_name(simple_bufts[i]);
        }
        name += ")";
    }

    منطقي operator<(ثابت ggml_backend_meta_buffer_type_context & other) ثابت {
        أرجع simple_bufts < other.simple_bufts;
    }
};

ساكن size_t ggml_backend_meta_buft_n_bufts(ggml_backend_buffer_type_t meta_buft) {
    GGML_ASSERT(ggml_backend_buft_is_meta(meta_buft));
    ثابت ggml_backend_meta_buffer_type_context * meta_buft_ctx = (ثابت ggml_backend_meta_buffer_type_context *) meta_buft->context;
    أرجع meta_buft_ctx->simple_bufts.size();
}

ساكن ثابت حرف * ggml_backend_meta_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_backend_buft_is_meta(buft));
    ثابت ggml_backend_meta_buffer_type_context * meta_buft_ctx = (ثابت ggml_backend_meta_buffer_type_context *) buft->context;
    أرجع meta_buft_ctx->name.c_str();
}

ساكن ggml_backend_buffer_type_t ggml_backend_meta_buft_simple_buft(ggml_backend_buffer_type_t meta_buft, size_t index) {
    GGML_ASSERT(ggml_backend_buft_is_meta(meta_buft));
    ثابت ggml_backend_meta_buffer_type_context * meta_buft_ctx = (ثابت ggml_backend_meta_buffer_type_context *) meta_buft->context;
    GGML_ASSERT(index < meta_buft_ctx->simple_bufts.size());
    أرجع meta_buft_ctx->simple_bufts[index];
}

ساكن ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size);

ساكن ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer_n(ggml_backend_buffer_type_t buft, ggml_tensor ** tensors, صحيح n_tensors);

ساكن size_t ggml_backend_meta_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ثابت size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_alignment = 1;
    لكل (size_t i = 0; i < n_simple_bufts; i++) {
        ثابت size_t alignment = ggml_backend_buft_get_alignment(ggml_backend_meta_buft_simple_buft(buft, i));
        max_alignment = std::max(max_alignment, alignment);
        GGML_ASSERT(max_alignment % alignment == 0);
    }
    أرجع max_alignment;
}

ساكن size_t ggml_backend_meta_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    ثابت size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_size = SIZE_MAX;
    لكل (size_t i = 0; i < n_simple_bufts; i++) {
        max_size = std::min(max_size, ggml_backend_buft_get_max_size(ggml_backend_meta_buft_simple_buft(buft, i)));
    }
    أرجع max_size;
}

ساكن size_t ggml_backend_meta_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, ثابت ggml_tensor * tensor) {
    ثابت size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_alloc_size = 0;
    لكل (size_t i = 0; i < n_simple_bufts; i++) {
        ثابت size_t alloc_size = ggml_backend_buft_get_alloc_size(ggml_backend_meta_buft_simple_buft(buft, i), tensor);
        max_alloc_size = std::max(max_alloc_size, alloc_size);
    }
    أرجع max_alloc_size;
}

ساكن منطقي ggml_backend_meta_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    ثابت size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    لكل (size_t i = 0; i < n_simple_bufts; i++) {
        إذا (!ggml_backend_buft_is_host(ggml_backend_meta_buft_simple_buft(buft, i))) {
            أرجع خطأ;
        }
    }
    أرجع true;
}

ساكن ثابت بنية ggml_backend_buffer_type_i ggml_backend_meta_buffer_type_iface = {
    /* .get_name            = */ ggml_backend_meta_buffer_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_meta_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ ggml_backend_meta_buffer_type_alloc_buffer_n,
    /* .get_alignment       = */ ggml_backend_meta_buffer_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_meta_buffer_type_get_max_size,
    /* .get_alloc_size      = */ ggml_backend_meta_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ عدم,
    /* .is_host             = */ ggml_backend_meta_buffer_type_is_host,
};

منطقي ggml_backend_buft_is_meta(ggml_backend_buffer_type_t buft) {
    أرجع buft != nullptr && buft->iface.get_name == ggml_backend_meta_buffer_type_iface.get_name;
}

ساكن ggml_backend_buffer_type_t ggml_backend_meta_device_get_buffer_type(ggml_backend_dev_t dev) {
    ساكن std::map<ggml_backend_dev_t, بنية ggml_backend_buffer_type> meta_bufts;
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    {
        auto it = meta_bufts.find(dev);
        إذا (it != meta_bufts.end()) {
            أرجع &it->second;
        }
    }

    ثابت size_t n_devs = ggml_backend_meta_dev_n_devs(dev);
    std::vector<ggml_backend_buffer_type_t> simple_bufts;
    simple_bufts.reserve(n_devs);
    لكل (size_t i = 0; i < n_devs; i++) {
        simple_bufts.push_back(ggml_backend_dev_buffer_type(ggml_backend_meta_dev_simple_dev(dev, i)));
    }
    ggml_backend_meta_buffer_type_context * buft_ctx = جديد ggml_backend_meta_buffer_type_context(simple_bufts);

    بنية ggml_backend_buffer_type meta_buft = {
        /*iface  =*/ ggml_backend_meta_buffer_type_iface,
        /*device =*/ dev,
        /*ctx    =*/ buft_ctx,
    };
    auto result = meta_bufts.emplace(dev, meta_buft);
    أرجع &result.first->second;
}

ساكن ggml_backend_buffer_type_t ggml_backend_meta_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ثابت ggml_backend_meta_device_context * meta_dev_ctx = (ثابت ggml_backend_meta_device_context *) dev->context;

    ggml_backend_buffer_type_t host_buft = nullptr;
    لكل (ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
        ggml_backend_buffer_type_t simple_host_buft = ggml_backend_dev_host_buffer_type(simple_dev);
        إذا (simple_host_buft == nullptr) {
            أرجع nullptr;
        }
        إذا (host_buft == nullptr) {
            host_buft = simple_host_buft;
        } وإلا إذا (host_buft != simple_host_buft) {
            // if different simple devices have different host buffer types,
            // we cannot provide a single host buffer type for the meta device
            أرجع nullptr;
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

بنية ggml_backend_meta_buffer_context {
    // FIXME
    // Most tensors can simply be stored statically in their own buffer.
    // Externally created views however also need a mapping to simple tensors but they use the buffer of the view source.
    // If external views are simply using that buffer they will slowly deplete its memory.
    // Current solution: rotating set of 2 "compute" containers to hold external views, works correctly for llama.cpp.
    // Long-term: tie the lifetime of external views to the meta backend executing the graph instead,
    //     currently not possible due to graph-external operations in the backend scheduler.
    ggml_backend_meta_simple_tensor_container stc_static;
    ggml_backend_meta_simple_tensor_container stc_compute[2];
    صحيح stc_compute_index      = 0;
    صحيح stc_compute_index_next = 0;
    std::vector<ggml_backend_buffer_ptr> bufs;

    // FIXME
    // The size of the split state cache is unbounded and can theoretically grow infinitely large.
    // However, it is also expensive to build and clearing it on every rebuild in ggml_backend_meta_graph_compute is too expensive.
    ساكن constexpr size_t nbtc = GGML_TENSOR_SIZE - حجم(ggml_tensor::padding);
    std::map<std::pair<ثابت ggml_tensor *, منطقي>, std::pair<ggml_backend_meta_split_state, حرف[nbtc]>> split_state_cache;

    صحيح debug;

    ggml_backend_meta_buffer_context(
            ggml_backend_meta_simple_tensor_container & stc_static,
            ggml_backend_meta_simple_tensor_container & stc_compute_0,
            ggml_backend_meta_simple_tensor_container & stc_compute_1,
            ثابت std::vector<ggml_backend_buffer_t> & bufs)
            : stc_static(std::move(stc_static)), stc_compute{std::move(stc_compute_0), std::move(stc_compute_1)} {
        هذا->bufs.reserve(bufs.size());
        لكل (ggml_backend_buffer_t buf : bufs) {
            هذا->bufs.emplace_back(buf);
        }
        ثابت حرف * GGML_META_DEBUG = getenv("GGML_META_DEBUG");
        debug = GGML_META_DEBUG ? atoi(GGML_META_DEBUG) : 0;
    }

    ggml_backend_meta_simple_tensor_container & get_simple_tensor_container(ثابت ggml_tensor * tensor) {
        إذا (stc_static.simple_tensors.find(tensor) != stc_static.simple_tensors.end()) {
            أرجع stc_static;
        }
        أرجع stc_compute[stc_compute_index];
    }
};

ساكن فراغ ggml_backend_meta_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    delete buf_ctx;
}

ساكن size_t ggml_backend_meta_buffer_n_bufs(ggml_backend_buffer_t meta_buf) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(meta_buf));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) meta_buf->context;
    أرجع buf_ctx->bufs.size();
}

ساكن ggml_backend_buffer_t ggml_backend_meta_buffer_simple_buffer(ggml_backend_buffer_t meta_buf, size_t index) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(meta_buf));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) meta_buf->context;
    GGML_ASSERT(index < buf_ctx->bufs.size());
    أرجع buf_ctx->bufs[index].get();
}

ساكن بنية ggml_tensor * ggml_backend_meta_buffer_simple_tensor(ثابت بنية ggml_tensor * tensor, size_t index) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(tensor->buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    GGML_ASSERT(index < buf_ctx->bufs.size());

    ggml_backend_meta_simple_tensor_container & stc = buf_ctx->get_simple_tensor_container(tensor);
    auto it = stc.simple_tensors.find(tensor);
    إذا (it == stc.simple_tensors.end()) {
        أرجع nullptr;
    }
    أرجع it->second[index];
}

ساكن بنية ggml_backend_meta_split_state ggml_backend_meta_get_split_state(ثابت بنية ggml_tensor * tensor, منطقي assume_sync);

ساكن بنية ggml_backend_meta_split_state ggml_backend_meta_get_split_state(
        ggml_backend_meta_simple_tensor_container & stc, ثابت بنية ggml_tensor * tensor, منطقي assume_sync) {
    // FIXME Currently this function preserves/erases the information in n_segments and nr in an inconsistent way.
    // Since the operations in question are developed specifically for llama.cpp this currently does not manifest as a bug there.
    // However, in a broader ggml context with arbitrary ggml graphs this can lead to unexpected results.
    ثابت size_t n_bufs = ggml_backend_meta_buffer_n_bufs(tensor->buffer);
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;

    auto split_states_equal = [&](ثابت ggml_backend_meta_split_state & a, ثابت ggml_backend_meta_split_state & b) -> منطقي {
        إذا (a.axis != b.axis) {
            أرجع خطأ;
        }
        لكل (size_t j = 0; j < n_bufs; j++) {
            int64_t sum_a = 0;
            لكل (size_t s = 0; s < a.n_segments; s++) {
                sum_a += a.ne[s*n_bufs + j] * a.nr[s];
            }
            int64_t sum_b = 0;
            لكل (size_t s = 0; s < b.n_segments; s++) {
                sum_b += b.ne[s*n_bufs + j] * b.nr[s];
            }
            إذا (sum_a != sum_b) {
                أرجع خطأ;
            }
        }
        أرجع true;
    };

    auto handle_generic = [&](ثابت std::vector<ggml_backend_meta_split_state> & src_ss, منطقي scalar_only) -> ggml_backend_meta_split_state {
        ggml_backend_meta_split_state ret = {GGML_BACKEND_SPLIT_AXIS_NONE, {0}, {1}, 1};
        لكل (size_t i = 0; i < GGML_MAX_SRC; i++) {
            إذا (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                استمر;
            }
            إذا (ret.axis == GGML_BACKEND_SPLIT_AXIS_NONE) {
                ret = src_ss[i];
            } وإلا إذا (!split_states_equal(src_ss[i], ret)) {
                ret = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
                اكسر;
            }
        }
        إذا (ret.axis == GGML_BACKEND_SPLIT_AXIS_NONE) {
            ret = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        إذا (scalar_only && ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
            ret = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        GGML_ASSERT(ret.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
        أرجع ret;
    };

    // Some ops process data on a per-row bases:
    auto handle_per_row = [&](ثابت std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0);
        أرجع src_ss[0];
    };

    // Some ops broadcast the src1 data across src0:
    auto handle_bin_bcast = [&](ثابت std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        إذا (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS &&
                tensor->src[1]->ne[src_ss[0].axis] == 1 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            أرجع src_ss[0];
        }
        إذا (src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && (src_ss[0].axis == src_ss[1].axis ||
           (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL)))) {
            أرجع src_ss[0]; // GGML_OP_ADD_ID
        }
        GGML_ASSERT(tensor->src[2] == nullptr || src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        أرجع handle_generic(src_ss, /*scalar_only =*/ خطأ);
    };

    auto handle_concat = [&](ثابت std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        ثابت ggml_backend_meta_split_axis concat_axis = ggml_backend_meta_split_axis(ggml_get_op_params_i32(tensor, 0));
        إذا (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis >= 0 && src_ss[1].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(concat_axis != src_ss[1].axis);
            أرجع src_ss[1];
        }
        إذا (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(concat_axis != src_ss[0].axis);
            أرجع src_ss[0];
        }
        إذا (src_ss[0].axis == src_ss[1].axis && src_ss[0].axis != concat_axis) {
            أرجع src_ss[0];
        }
        أرجع handle_generic(src_ss, /*scalar_only =*/ true);
    };

    auto handle_mul_mat = [&](ثابت std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        إذا (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            أرجع {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
        }
        إذا (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            ggml_backend_meta_split_state ret = src_ss[0];
            ret.axis = GGML_BACKEND_SPLIT_AXIS_0;
            ret.nr[0] = 1;
            ret.n_segments = 1;
            أرجع ret;
        }
        إذا (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_1 && src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            أرجع src_ss[1];
        }
        إذا (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(split_states_equal(src_ss[0], src_ss[1]));
            أرجع {assume_sync ? GGML_BACKEND_SPLIT_AXIS_MIRRORED : GGML_BACKEND_SPLIT_AXIS_PARTIAL, {0}, {1}, 1};
        }
        إذا (src_ss[0].axis == src_ss[1].axis && src_ss[0].axis >= GGML_BACKEND_SPLIT_AXIS_2 &&
                src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(split_states_equal(src_ss[0], src_ss[1]));
            أرجع src_ss[0];
        }
        // batched matmul with the batches split across devices and a replicated activation
        إذا (src_ss[0].axis >= GGML_BACKEND_SPLIT_AXIS_2 && src_ss[0].axis < GGML_MAX_DIMS &&
                src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            أرجع src_ss[0];
        }
        GGML_ABORT("unsupported mul_mat split states: node=%s src0=%s axis=%d src1=%s axis=%d",
            tensor->name, tensor->src[0]->name, (صحيح) src_ss[0].axis, tensor->src[1]->name, (صحيح) src_ss[1].axis);
        //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
    };

    auto handle_reshape = [&](ثابت std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        بدل (src_ss[0].axis) {
            حالة GGML_BACKEND_SPLIT_AXIS_0:
            حالة GGML_BACKEND_SPLIT_AXIS_1:
            حالة GGML_BACKEND_SPLIT_AXIS_2:
            حالة GGML_BACKEND_SPLIT_AXIS_3: {
                int64_t base_ne_in = 1;
                لكل (صحيح dim = 0; dim <= src_ss[0].axis; dim++) {
                    base_ne_in *= tensor->src[0]->ne[dim];
                }
                إذا (src_ss[0].n_segments == 1) {
                    base_ne_in /= src_ss[0].nr[0];
                    إذا (src_ss[0].axis == ggml_n_dims(tensor->src[0]) - 1 && src_ss[0].nr[0] == 1) {
                        أرجع {ggml_backend_meta_split_axis(ggml_n_dims(tensor) - 1), {0}, {1}, 1};
                    }
                    إذا (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && tensor->ne[0] == tensor->src[0]->ne[0] &&
                            tensor->ne[1] == 1 && src_ss[0].nr[0] == 1) {
                        منطقي complete_rows = true;
                        لكل (size_t j = 0; j < n_bufs; j++) {
                            ثابت int64_t ne = src_ss[0].ne[j];
                            complete_rows = complete_rows && (ne == 0 || ne == tensor->src[0]->ne[0]);
                        }
                        إذا (complete_rows) {
                            // Move a complete dim-0 split to the following singleton dimension.
                            أرجع {GGML_BACKEND_SPLIT_AXIS_1, {0}, {1}, 1};
                        }
                    }
                }
                // Reshape outputs use one segment; split-state propagation merges source segments.
                int64_t base_ne_out = 1;
                لكل (صحيح dim = 0; dim < GGML_MAX_DIMS; dim++) {
                    base_ne_out *= tensor->ne[dim];
                    إذا (base_ne_out % base_ne_in == 0) {
                        أرجع {ggml_backend_meta_split_axis(dim), {0}, {uint32_t(base_ne_out/base_ne_in)}, 1};
                    }
                    إذا (base_ne_out > base_ne_in) {
                        GGML_ASSERT(src_ss[0].n_segments == 1);
                        GGML_ASSERT(src_ss[0].nr[0]      == 1);
                        أرجع {ggml_backend_meta_split_axis(dim), {0}, {1}, 1};
                    }
                }
                GGML_ABORT("shape mismatch for %s", ggml_op_name(tensor->op));
            }
            حالة GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            حالة GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
                أرجع src_ss[0];
            }
            افتراضي: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    auto handle_cpy = [&](ثابت std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        إذا (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            أرجع handle_reshape(src_ss);
        }
        أرجع handle_generic(src_ss, /*scalar_only =*/ خطأ);
    };

    auto handle_view = [&](ثابت std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        إذا (ggml_is_contiguous(tensor) && ggml_is_contiguous(tensor->src[0])) {
            أرجع handle_reshape(src_ss);
        }
        ثابت صحيح axis = src_ss[0].axis;
        {
            منطقي all_strides_the_same = true;
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
                    أرجع {ggml_backend_meta_split_axis(dim), {0}, {1}, 1};
                }
            }
            GGML_ABORT("fatal error");
        }
        إذا (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED || src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
            أرجع src_ss[0];
        }
        GGML_ABORT("view of permuted tensor not implemented");
        //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
    };

    auto handle_permute = [&](ثابت std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        بدل (src_ss[0].axis) {
            حالة GGML_BACKEND_SPLIT_AXIS_0:
            حالة GGML_BACKEND_SPLIT_AXIS_1:
            حالة GGML_BACKEND_SPLIT_AXIS_2:
            حالة GGML_BACKEND_SPLIT_AXIS_3: {
                GGML_ASSERT(src_ss[0].n_segments == 1 || src_ss[0].nr[0] == 1);
                أرجع {ggml_backend_meta_split_axis(tensor->op_params[src_ss[0].axis]), {0}, {src_ss[0].nr[0]}, 1};
            }
            حالة GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            حالة GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
                أرجع src_ss[0];
            }
            افتراضي: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    auto handle_transpose = [&](ثابت std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        بدل (src_ss[0].axis) {
            حالة GGML_BACKEND_SPLIT_AXIS_0:
            حالة GGML_BACKEND_SPLIT_AXIS_1: {
                GGML_ASSERT(src_ss[0].n_segments == 1 || src_ss[0].nr[0] == 1);
                أرجع {ggml_backend_meta_split_axis(صحيح(src_ss[0].axis) ^ 1), {0}, {src_ss[0].nr[0]}, 1};
            }
            حالة GGML_BACKEND_SPLIT_AXIS_2:
            حالة GGML_BACKEND_SPLIT_AXIS_3:
            حالة GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            حالة GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
                أرجع src_ss[0];
            }
            افتراضي: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    auto handle_get_rows = [&](ثابت std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        إذا (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            أرجع src_ss[0];
        }
        أرجع handle_generic(src_ss, /*scalar_only =*/ true);
    };

    auto handle_set_rows = [&](ثابت std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        GGML_ASSERT(split_states_equal(src_ss[0], src_ss[2]));
        أرجع src_ss[0];
    };

    auto handle_rope = [&](ثابت std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        أرجع src_ss[0];
    };

    auto handle_pad = [&](ثابت std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        إذا (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(tensor->op_params[2*src_ss[0].axis + 0] == 0);
            GGML_ASSERT(tensor->op_params[2*src_ss[0].axis + 1] == 0);
        }
        أرجع src_ss[0];
    };

    auto handle_pool_1d = [&](ثابت std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        إذا (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0) {
            أرجع {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
        }
        أرجع src_ss[0];
    };

    auto handle_flash_attn_ext = [&](ثابت std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(tensor->src[3] == nullptr || src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

        إذا (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            GGML_ASSERT(src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            GGML_ASSERT(tensor->src[4] == nullptr || src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            أرجع {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
        }

        GGML_ASSERT(src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_2);
        ثابت منطقي kv_split = src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_2 &&
                src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_2;
        ثابت منطقي kv_mirrored = src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED;
        GGML_ASSERT(kv_split || kv_mirrored);
        GGML_ASSERT(tensor->src[4] == nullptr || src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_0);
        أرجع {GGML_BACKEND_SPLIT_AXIS_1, {0}, {1}, 1};
    };

    auto handle_lightning_indexer = [&](
            ثابت std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        لكل (size_t i = 0; i < 4; i++) {
            GGML_ASSERT(src_ss[i].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        }
        أرجع {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
    };

    auto handle_ssm_conv = [&](ثابت std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        إذا (src_ss[0].axis == src_ss[1].axis) {
            إذا (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0) {
                أرجع {GGML_BACKEND_SPLIT_AXIS_1, {0}, {1}, 1};
            }
            إذا (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1) {
                أرجع {GGML_BACKEND_SPLIT_AXIS_0, {0}, {1}, 1};
            }
        }
        أرجع handle_generic(src_ss, /*scalar_only =*/ خطأ);
    };

    auto handle_gated_delta_net = [&](ثابت std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        إذا (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            أرجع src_ss[0];
        }
        GGML_ASSERT(src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_1);
        // state shape is [S_v, S_v, H_v, n_seqs] (s0 only); the heads dim is its own axis 2,
        // so a head-aligned split on the input cache lands on axis 2 here.
        GGML_ASSERT(src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_2 || src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_1 || src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_0);
        أرجع {GGML_BACKEND_SPLIT_AXIS_0, {0}, {1}, 1};
    };

    auto calculate_split_state = [&]() -> ggml_backend_meta_split_state {
        إذا (ggml_nelements(tensor) == 0) {
            أرجع {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        إذا (ggml_backend_buffer_get_usage(tensor->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE && tensor->view_src == nullptr) {
            ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(tensor->buffer));
            ثابت ggml_backend_meta_device_context * dev_ctx = (ثابت ggml_backend_meta_device_context *) dev->context;
            ggml_backend_meta_split_state ret = dev_ctx->get_split_state(tensor, dev_ctx->get_split_state_ud);
            إذا (ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
                ثابت int64_t granularity = ret.axis == GGML_BACKEND_SPLIT_AXIS_0 ? ggml_blck_size(tensor->type) : 1;
                int64_t ne_sum = 0;
                لكل (size_t s = 0; s < ret.n_segments; s++) {
                    لكل (size_t j = 0; j < n_bufs; j++) {
                        GGML_ASSERT(ret.ne[s*n_bufs + j] % granularity == 0);
                        ne_sum += ret.ne[s*n_bufs + j] * ret.nr[s];
                    }
                }
                GGML_ASSERT(ne_sum == tensor->ne[ret.axis]);
            } وإلا إذا (ret.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                GGML_ASSERT(ret.n_segments == 1);
                GGML_ASSERT(ret.nr[0] == 1);
            }
            أرجع ret;
        }

        std::vector<ggml_backend_meta_split_state> src_ss(GGML_MAX_SRC, {GGML_BACKEND_SPLIT_AXIS_NONE, {0}, {1}, 1});
        لكل (size_t i = 0; i < GGML_MAX_SRC; i++) {
            إذا (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                src_ss[i] = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
                استمر;
            }
            src_ss[i] = ggml_backend_meta_get_split_state(stc, tensor->src[i], /*assume_sync =*/ true);
            GGML_ASSERT(src_ss[i].axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
        }

        ggml_backend_meta_split_state split_state;
        بدل (tensor->op) {
            حالة GGML_OP_NONE: {
                إذا (tensor->view_src != nullptr) {
                    // full-tensor view created with ggml_view_tensor, transparent for the split state
                    split_state = ggml_backend_meta_get_split_state(stc, tensor->view_src, assume_sync);
                } وإلا {
                    split_state = {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
                }
            } اكسر;
            حالة GGML_OP_DUP: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } اكسر;
            حالة GGML_OP_ADD:
            حالة GGML_OP_ADD_ID: {
                split_state = handle_bin_bcast(src_ss);
            } اكسر;
            حالة GGML_OP_ADD1:
            حالة GGML_OP_ACC: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } اكسر;
            حالة GGML_OP_SUB:
            حالة GGML_OP_MUL:
            حالة GGML_OP_DIV: {
                split_state = handle_bin_bcast(src_ss);
            } اكسر;
            حالة GGML_OP_SQR:
            حالة GGML_OP_SQRT:
            حالة GGML_OP_LOG:
            حالة GGML_OP_SIN:
            حالة GGML_OP_COS: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ خطأ);
            } اكسر;
            حالة GGML_OP_SUM: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } اكسر;
            حالة GGML_OP_SUM_ROWS:
            حالة GGML_OP_CUMSUM:
            حالة GGML_OP_MEAN:
            حالة GGML_OP_ARGMAX:
            حالة GGML_OP_گنتی_EQUAL: {
                split_state = handle_per_row(src_ss);
            } اكسر;
            حالة GGML_OP_REPEAT:
            حالة GGML_OP_REPEAT_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ خطأ);
            } اكسر;
            حالة GGML_OP_CONCAT: {
                split_state = handle_concat(src_ss);
            } اكسر;
            حالة GGML_OP_SILU_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ خطأ);
            } اكسر;
            حالة GGML_OP_NORM:
            حالة GGML_OP_RMS_NORM:
            حالة GGML_OP_RMS_NORM_BACK:
            حالة GGML_OP_GROUP_NORM:
            حالة GGML_OP_L2_NORM: {
                split_state = handle_per_row(src_ss);
            } اكسر;
            حالة GGML_OP_MUL_MAT:
            حالة GGML_OP_MUL_MAT_ID: {
                split_state = handle_mul_mat(src_ss);
            } اكسر;
            حالة GGML_OP_OUT_PROD: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } اكسر;
            حالة GGML_OP_SCALE: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ خطأ);
            } اكسر;
            حالة GGML_OP_SET: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } اكسر;
            حالة GGML_OP_CPY: {
                split_state = handle_cpy(src_ss);
            } اكسر;
            حالة GGML_OP_CONT:
            حالة GGML_OP_RESHAPE: {
                split_state = handle_reshape(src_ss);
            } اكسر;
            حالة GGML_OP_VIEW: {
                split_state = handle_view(src_ss);
            } اكسر;
            حالة GGML_OP_PERMUTE: {
                split_state = handle_permute(src_ss);
            } اكسر;
            حالة GGML_OP_TRANSPOSE: {
                split_state = handle_transpose(src_ss);
            } اكسر;
            حالة GGML_OP_GET_ROWS: {
                split_state = handle_get_rows(src_ss);
            } اكسر;
            حالة GGML_OP_GET_ROWS_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } اكسر;
            حالة GGML_OP_SET_ROWS: {
                split_state = handle_set_rows(src_ss);
            } اكسر;
            حالة GGML_OP_DIAG:
            حالة GGML_OP_DIAG_MASK_INF:
            حالة GGML_OP_DIAG_MASK_ZERO: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } اكسر;
            حالة GGML_OP_SOFT_MAX:
            حالة GGML_OP_SOFT_MAX_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ خطأ);
            } اكسر;
            حالة GGML_OP_ROPE: {
                split_state = handle_rope(src_ss);
            } اكسر;
            حالة GGML_OP_ROPE_BACK: {
                split_state = handle_rope(src_ss);
            } اكسر;
            حالة GGML_OP_CLAMP: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ خطأ);
            } اكسر;
            حالة GGML_OP_CONV_TRANSPOSE_1D:
            حالة GGML_OP_IM2COL:
            حالة GGML_OP_IM2COL_BACK:
            حالة GGML_OP_IM2COL_3D:
            حالة GGML_OP_CONV_2D:
            حالة GGML_OP_CONV_3D:
            حالة GGML_OP_CONV_2D_DW:
            حالة GGML_OP_CONV_TRANSPOSE_2D:
            حالة GGML_OP_POOL_2D:
            حالة GGML_OP_POOL_2D_BACK:
            حالة GGML_OP_UPSCALE: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } اكسر;
            حالة GGML_OP_POOL_1D: {
                split_state = handle_pool_1d(src_ss);
            } اكسر;
            حالة GGML_OP_PAD: {
                split_state = handle_pad(src_ss);
            } اكسر;
            حالة GGML_OP_PAD_REFLECT_1D:
            حالة GGML_OP_ROLL:
            حالة GGML_OP_ARANGE:
            حالة GGML_OP_TIMESTEP_EMBEDDING: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } اكسر;
            حالة GGML_OP_ARGSORT:
            حالة GGML_OP_چوٹی_K: {
                split_state = handle_per_row(src_ss);
            } اكسر;
            حالة GGML_OP_LEAKY_RELU: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ خطأ);
            } اكسر;
            حالة GGML_OP_TRI: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } اكسر;
            حالة GGML_OP_FILL: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ خطأ);
            } اكسر;
            حالة GGML_OP_FLASH_ATTN_EXT: {
                split_state = handle_flash_attn_ext(src_ss);
            } اكسر;
            حالة GGML_OP_FLASH_ATTN_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } اكسر;
            حالة GGML_OP_SSM_CONV: {
                split_state = handle_ssm_conv(src_ss);
            } اكسر;
            حالة GGML_OP_SSM_SCAN:
            حالة GGML_OP_WIN_PART:
            حالة GGML_OP_WIN_UNPART:
            حالة GGML_OP_GET_REL_POS:
            حالة GGML_OP_ADD_REL_POS:
            حالة GGML_OP_RWKV_WKV6:
            حالة GGML_OP_GATED_LINEAR_ATTN:
            حالة GGML_OP_RWKV_WKV7:
            حالة GGML_OP_SOLVE_TRI: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } اكسر;
            حالة GGML_OP_GATED_DELTA_NET: {
                split_state = handle_gated_delta_net(src_ss);
            } اكسر;
            حالة GGML_OP_LIGHTNING_INDEXER: {
                split_state = handle_lightning_indexer(src_ss);
            } اكسر;
            حالة GGML_OP_DSV4_HC_COMB:
            حالة GGML_OP_DSV4_HC_PRE:
            حالة GGML_OP_DSV4_HC_POST:
            حالة GGML_OP_چوٹی_K_MASK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } اكسر;
            حالة GGML_OP_UNARY: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ خطأ);
            } اكسر;
            حالة GGML_OP_MAP_CUSTOM1:
            حالة GGML_OP_MAP_CUSTOM2:
            حالة GGML_OP_MAP_CUSTOM3:
            حالة GGML_OP_CUSTOM: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } اكسر;
            حالة GGML_OP_CROSS_ENTROPY_LOSS:
            حالة GGML_OP_CROSS_ENTROPY_LOSS_BACK: {
                split_state = handle_per_row(src_ss);
            } اكسر;
            حالة GGML_OP_OPT_STEP_ADAMW:
            حالة GGML_OP_OPT_STEP_SGD:
            حالة GGML_OP_GLU: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ خطأ);
            } اكسر;
            افتراضي: {
                GGML_ABORT("ggml op not implemented: %s", ggml_op_name(tensor->op));
                split_state = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            } اكسر;
        }
        إذا (split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS) {
            منطقي first_src_split_by_axis = true;
            ثابت size_t n_bufs = ggml_backend_meta_buffer_n_bufs(tensor->buffer);

            لكل (size_t i = 0; i < GGML_MAX_SRC; i++) {
                إذا (tensor->src[i] == nullptr || src_ss[i].axis < 0 || src_ss[i].axis >= GGML_MAX_DIMS) {
                    استمر;
                }
                إذا (first_src_split_by_axis) {
                    لكل (size_t j = 0; j < n_bufs; j++) {
                        // Take over ratio from src:
                        لكل (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            split_state.ne[s*n_bufs + j] = 0;
                        }
                        لكل (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            split_state.ne[j] += src_ss[i].ne[s*n_bufs + j] * src_ss[i].nr[s];
                        }
                        split_state.ne[j] *= tensor->ne[split_state.axis];
                        إذا (split_state.ne[j] != 0 || tensor->src[i]->ne[src_ss[i].axis] != 0) {
                            ثابت int64_t div = tensor->src[i]->ne[src_ss[i].axis] * split_state.nr[0];
                            GGML_ASSERT(split_state.ne[j] % div == 0);
                            split_state.ne[j] /= div;
                        }
                    }
                } وإلا {
                    GGML_ASSERT(split_state.n_segments == 1);
                    لكل (size_t j = 0; j < n_bufs; j++) {
                        // Assert that ratio is consistent:
                        int64_t sum = 0;
                        لكل (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            sum += src_ss[i].ne[s*n_bufs + j] * src_ss[i].nr[s];
                        }
                        GGML_ASSERT(split_state.ne[j]*split_state.nr[0] * tensor->src[i]->ne[src_ss[i].axis]
                                                                 == sum * tensor->ne[split_state.axis]);
                    }
                }
                first_src_split_by_axis = خطأ;
            }
            GGML_ASSERT(!first_src_split_by_axis);
        }
        أرجع split_state;
    };

    ثابت std::pair کلید = std::make_pair(tensor, assume_sync);
    auto it = buf_ctx->split_state_cache.find(کلید);
    إذا (it != buf_ctx->split_state_cache.end() && memcmp(it->second.second, (ثابت حرف *) tensor, حجم(it->second.second)) != 0) {
        buf_ctx->split_state_cache.clear();
        it = buf_ctx->split_state_cache.end();
    }

    إذا (it == buf_ctx->split_state_cache.end()) {
        buf_ctx->split_state_cache[کلید].first = calculate_split_state();
        memcpy(buf_ctx->split_state_cache[کلید].second, tensor, حجم(buf_ctx->split_state_cache[کلید].second));
        إذا (buf_ctx->debug > 0) {
            std::string srcs_info;
            لكل (size_t i = 0; i < GGML_MAX_SRC; i++) {
                إذا (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                    استمر;
                }
                إذا (!srcs_info.empty()) {
                    srcs_info += ", ";
                }
                ثابت ggml_backend_meta_split_state split_state =
                        ggml_backend_meta_get_split_state(tensor->src[i], true);
                GGML_ASSERT(split_state.n_segments == 1);
                ثابت حرف * axis_name = ggml_backend_meta_split_axis_name(split_state.axis);
                std::string ne_info;
                لكل (size_t j = 0; j < n_bufs; j++) {
                    إذا (!ne_info.empty()) {
                        ne_info += ", ";
                    }
                    ne_info += std::to_string(split_state.ne[j]) + "x" + std::to_string(split_state.nr[0]);
                }
                srcs_info += std::string(tensor->src[i]->name) + "[" + ggml_op_name(tensor->src[i]->op) + ", " + axis_name + ", {" + ne_info + "}]";
            }
            std::string ne_info;
            لكل (size_t j = 0; j < n_bufs; j++) {
                إذا (!ne_info.empty()) {
                    ne_info += ", ";
                }
                ثابت ggml_backend_meta_split_state & ss = buf_ctx->split_state_cache[کلید].first;
                ne_info += std::to_string(ss.ne[j]) + "x" + std::to_string(ss.nr[0]);
            }
            GGML_LOG_DEBUG("SPLIT_STATE: {%s} -> %s[%s, %s, {%s}]\n", srcs_info.c_str(), tensor->name, ggml_op_name(tensor->op),
                ggml_backend_meta_split_axis_name(buf_ctx->split_state_cache[کلید].first.axis), ne_info.c_str());
        }
    }

    ggml_backend_meta_split_state ret = buf_ctx->split_state_cache[کلید].first;
    GGML_ASSERT(ret.axis != GGML_BACKEND_SPLIT_AXIS_NONE);
#ifndef NDEBUG
    إذا (ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
        int64_t ne_ret = 0;
        لكل (size_t s = 0; s < ret.n_segments; s++) {
            لكل (size_t j = 0; j < n_bufs; j++) {
                ne_ret += ret.ne[s*n_bufs + j] * ret.nr[s];
            }
        }
        أكد(ne_ret == tensor->ne[صحيح(ret.axis)]);
    }
#endif // NDEBUG
    أرجع ret;
}

ساكن منطقي ggml_backend_meta_is_host_view(ثابت بنية ggml_tensor * tensor) {
    أرجع ggml_is_view(tensor) && ggml_backend_buffer_is_host(tensor->view_src->buffer);
}

ساكن بنية ggml_backend_meta_split_state ggml_backend_meta_get_split_state(ثابت بنية ggml_tensor * tensor, منطقي assume_sync) {
    // [TAG_META_HOST_VIEWS]
    // TODO: technically, this check should not be needed if the backend scheduler correctly prevents assigning
    //       such host-buffer views to the meta backend. figure out how to update the scheduler logic to achieve that
    //       ref: https://github.com/ggml-org/llama.cpp/pull/30217
    إذا (!ggml_backend_buffer_is_meta(tensor->buffer)) {
        GGML_ASSERT(ggml_backend_meta_is_host_view(tensor));

        // the view is not allocated in the meta buffer, it is not split across the sub-devices
        أرجع { GGML_BACKEND_SPLIT_AXIS_NONE, {0}, {1}, 1 };
    }

    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    أرجع ggml_backend_meta_get_split_state(buf_ctx->get_simple_tensor_container(tensor), tensor, assume_sync);
}

ساكن فراغ * ggml_backend_meta_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_UNUSED(buffer);
    أرجع (فراغ *) 0x1000000000000000; // FIXME
}

ساكن تعداد ggml_status ggml_backend_meta_buffer_init_tensor_impl(ggml_backend_meta_simple_tensor_container & stc, ggml_tensor * tensor) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(tensor->buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    ثابت size_t n_simple_bufs = ggml_backend_meta_buffer_n_bufs(tensor->buffer);

    ثابت ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(stc, tensor, /*assume_sync =*/ true);
    GGML_ASSERT(ggml_nelements(tensor) == 0 || split_state.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
    GGML_ASSERT(split_state.n_segments <= 16);

    صحيح split_dim = split_state.axis;
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
        ggml_backend_buffer_t   simple_buf = buf_ctx->bufs[j].get();

        إذا (split_dim >= 0 && split_dim < GGML_MAX_DIMS) {
            // TODO: the following assert fails for llama-parallel even though the results are correct:
            // GGML_ASSERT(ggml_is_contiguously_allocated(tensor));
            ne[split_dim] = 0;
            لكل (size_t s = 0; s < split_state.n_segments; s++) {
                ne[split_dim] += split_state.ne[s*n_simple_bufs + j] * split_state.nr[s];
            }
            لكل (صحيح i = 0; i < GGML_MAX_DIMS; i++) {
                إذا (tensor->nb[i] > tensor->nb[split_dim]) {
                    nb[i] = tensor->nb[i] * ne[split_dim]/tensor->ne[split_dim];
                }
            }
        }

        ggml_tensor * t_ij = ggml_new_tensor(simple_ctx, tensor->type, GGML_MAX_DIMS, ne);
        t_ij->op = tensor->op;
        لكل (صحيح i = 0; i < GGML_MAX_DIMS; i++) {
            t_ij->nb[i] = nb[i];
        }
        t_ij->flags = tensor->flags;
        memcpy(t_ij->op_params, tensor->op_params, حجم(tensor->op_params));
        ggml_set_name(t_ij, tensor->name);
        t_ij->buffer = simple_buf;
        t_ij->view_src = tensor->view_src;
        t_ij->view_offs = tensor->view_offs;
        إذا (t_ij->view_src != nullptr && ggml_backend_buffer_is_meta(t_ij->view_src->buffer)) {
            t_ij->view_src = ggml_backend_meta_buffer_simple_tensor(tensor->view_src, j);
            إذا (t_ij->view_offs > 0 && split_dim >= 0 && split_dim < GGML_MAX_DIMS) {
                GGML_ASSERT(tensor->ne[split_dim] != 0);
                ثابت صحيح split_dim_view_src = ggml_backend_meta_get_split_state(tensor->view_src, /*assume_sync =*/ true).axis;
                GGML_ASSERT(split_dim_view_src >= 0 && split_dim_view_src < GGML_MAX_DIMS);

                // The offset can be internal to the data split, in those cases the view offset should not be scaled.
                // If however, the offset is larger than the data split then it needs to be scaled proportionally.
                منطقي split_internal_offset = t_ij->view_offs <= tensor->view_src->nb[split_dim_view_src];
                لكل (صحيح i = 0; i < GGML_MAX_DIMS; i++) {
                    ثابت size_t dim_size = tensor->ne[i] * tensor->nb[i];
                    إذا (tensor->view_offs <= dim_size && dim_size < tensor->nb[split_dim]) {
                        split_internal_offset = true;
                        اكسر;
                    }
                }
                إذا (!split_internal_offset) {
                    t_ij->view_offs = t_ij->view_offs * ne[split_dim]/tensor->ne[split_dim];
                }
            }
        }
        // TODO: revisit once the graph allocator has been refactored, see https://github.com/ggml-org/llama.cpp/pull/25051#issuecomment-4842873396
        ggml_backend_buffer_t init_buf = simple_buf;
        إذا (t_ij->view_src != nullptr) {
            t_ij->data = (حرف *) t_ij->view_src->data + t_ij->view_offs;
            // views inherit the source slice's concrete sub-buffer (issue 22197)
            إذا (tensor->view_src != nullptr && ggml_backend_buffer_is_meta(tensor->view_src->buffer)
                    && t_ij->view_src->buffer != nullptr) {
                t_ij->buffer = t_ij->view_src->buffer;
                init_buf     = t_ij->view_src->buffer;
            }
        } وإلا إذا (simple_buf != nullptr) {
            إذا (ggml_backend_buffer_is_multi_buffer(simple_buf)) {
                GGML_ABORT("multi buffers are not supported by the meta backend");
            }
            t_ij->data = (حرف *) ggml_backend_buffer_get_base(simple_buf)
                + size_t(tensor->data) - size_t(ggml_backend_buffer_get_base(tensor->buffer));
        }

        إذا (init_buf) {
            // the backend that owns the buffer will set .extra
            ggml_backend_buffer_init_tensor(init_buf, t_ij);
        } وإلا {
            t_ij->extra = tensor->extra;
        }

        لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
            t_ij->src[i] = tensor->src[i];
            إذا (tensor->src[i] == tensor) {
                t_ij->src[i] = t_ij;
            } وإلا إذا (t_ij->src[i] != nullptr && ggml_backend_buffer_is_meta(t_ij->src[i]->buffer)) {
                t_ij->src[i] = ggml_backend_meta_buffer_simple_tensor(tensor->src[i], j);
            }
        }

        simple_tensors.push_back(t_ij);
    }

    // If one of the sources has a zero-sized slice, disable the computation:
    لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
        إذا (tensor->src[i] == nullptr || !ggml_backend_buffer_is_meta(tensor->src[i]->buffer)) {
            استمر;
        }

        ثابت ggml_backend_meta_split_state split_state_src = ggml_backend_meta_get_split_state(tensor->src[i], /*assume_sync =*/ true);
        إذا (split_state_src.axis < 0 || split_state_src.axis >= GGML_MAX_DIMS) {
            استمر;
        }
        لكل (size_t j = 0; j < n_simple_bufs; j++) {
            int64_t ne_sum = 0;
            لكل (size_t s = 0; s < split_state_src.n_segments; s++) {
                ne_sum += split_state_src.ne[s*n_simple_bufs + j] * split_state_src.nr[s];
            }
            إذا (ne_sum == 0) {
                simple_tensors[j]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
            }
        }
    }

    stc.simple_tensors[tensor] = simple_tensors;

    أرجع GGML_STATUS_SUCCESS;
}

ساكن تعداد ggml_status ggml_backend_meta_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    buf_ctx->stc_compute_index = buf_ctx->stc_compute_index_next;
    أرجع ggml_backend_meta_buffer_init_tensor_impl(buf_ctx->get_simple_tensor_container(tensor), tensor);
}

ساكن فراغ ggml_backend_meta_buffer_memset_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ثابت size_t n_bufs = ggml_backend_meta_buffer_n_bufs(buffer);
    ثابت ggml_backend_meta_split_state split_state =
            ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ خطأ);
    GGML_ASSERT(ggml_is_contiguous(tensor) || split_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

    إذا (split_state.n_segments != 1 || split_state.nr[0] != 1) {
        GGML_ASSERT(split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(split_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        std::vector<size_t> simple_offsets(n_bufs, 0);
        إذا (split_state.axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            ثابت size_t row_stride = tensor->nb[1];
            GGML_ASSERT(offset % row_stride == 0);
            GGML_ASSERT(size   % row_stride == 0);
            ثابت int64_t row_start = offset / row_stride;
            ثابت int64_t row_گنتی = size   / row_stride;
            GGML_ASSERT(row_start + row_گنتی <= tensor->ne[1]);

            ثابت int64_t blck_size = ggml_blck_size(tensor->type);
            لكل (size_t s = 0; s < split_state.n_segments; s++) {
                لكل (size_t r = 0; r < split_state.nr[s]; r++) {
                    لكل (size_t j = 0; j < n_bufs; j++) {
                        ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                        GGML_ASSERT(split_state.ne[s*n_bufs + j] % blck_size == 0);
                        ثابت size_t nbytes = split_state.ne[s*n_bufs + j]/blck_size * tensor->nb[0];
                        لكل (int64_t row = 0; row < row_گنتی; row++) {
                            ggml_backend_tensor_memset(simple_tensor, value,
                                    simple_offsets[j] + (row_start + row)*simple_tensor->nb[1], nbytes);
                        }
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            أرجع;
        }

        GGML_ASSERT(split_state.axis == GGML_BACKEND_SPLIT_AXIS_1);

        ثابت size_t row_stride = tensor->nb[2];
        GGML_ASSERT(offset % row_stride == 0);
        GGML_ASSERT(size   % row_stride == 0);
        ثابت int64_t row_start = offset / row_stride;
        ثابت int64_t row_گنتی = size   / row_stride;
        GGML_ASSERT(row_start + row_گنتی <= tensor->ne[2]);

        لكل (size_t s = 0; s < split_state.n_segments; s++) {
            لكل (size_t r = 0; r < split_state.nr[s]; r++) {
                لكل (size_t j = 0; j < n_bufs; j++) {
                    ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                    ثابت size_t nbytes = split_state.ne[s*n_bufs + j] * tensor->nb[1];
                    لكل (int64_t row = 0; row < row_گنتی; row++) {
                        ggml_backend_tensor_memset(simple_tensor, value,
                                simple_offsets[j] + (row_start + row)*simple_tensor->nb[2], nbytes);
                    }
                    simple_offsets[j] += nbytes;
                }
            }
        }
        أرجع;
    }

    بدل (split_state.axis) {
        حالة GGML_BACKEND_SPLIT_AXIS_0:
        حالة GGML_BACKEND_SPLIT_AXIS_1:
        حالة GGML_BACKEND_SPLIT_AXIS_2: {
            ثابت size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            ثابت int64_t i_start =  offset        / chunk_size_full;
            ثابت int64_t i_stop  = (offset + size) / chunk_size_full;
            لكل (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ثابت size_t chunk_size = simple_tensor->nb[split_state.axis + 1];
                إذا (chunk_size == 0) {
                    استمر;
                }
                لكل (int64_t i = i_start; i < i_stop; i++) {
                    ggml_backend_tensor_memset(simple_tensor, value, i*chunk_size, chunk_size);
                }
            }
        } اكسر;
        حالة GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
            GGML_ASSERT(value == 0);
            [[fallthrough]];
        }
        حالة GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            لكل (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ggml_backend_tensor_memset(simple_tensor, value, offset, size);
            }
        } اكسر;
        افتراضي: {
            GGML_ABORT("fatal error");
        }
    }
}

ساكن فراغ ggml_backend_meta_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, ثابت فراغ * data, size_t offset, size_t size) {
    ثابت size_t n_bufs = ggml_backend_meta_buffer_n_bufs(buffer);
    ثابت ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ خطأ);
    GGML_ASSERT(ggml_is_contiguous(tensor) || split_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

    إذا (split_state.n_segments != 1 || split_state.nr[0] != 1) {
        GGML_ASSERT(split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(split_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        size_t offset_data = 0;
        std::vector<size_t> simple_offsets(n_bufs, 0);
        إذا (split_state.axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            ثابت size_t row_stride = tensor->nb[1];
            GGML_ASSERT(offset % row_stride == 0);
            GGML_ASSERT(size   % row_stride == 0);
            ثابت int64_t row_start = offset / row_stride;
            ثابت int64_t row_گنتی = size   / row_stride;
            GGML_ASSERT(row_start + row_گنتی <= tensor->ne[1]);

            ثابت int64_t blck_size = ggml_blck_size(tensor->type);
            لكل (size_t s = 0; s < split_state.n_segments; s++) {
                لكل (size_t r = 0; r < split_state.nr[s]; r++) {
                    لكل (size_t j = 0; j < n_bufs; j++) {
                        ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                        GGML_ASSERT(split_state.ne[s*n_bufs + j] % blck_size == 0);
                        ثابت size_t nbytes = split_state.ne[s*n_bufs + j]/blck_size * tensor->nb[0];
                        ggml_backend_tensor_set_2d(simple_tensor, (ثابت حرف *) data + offset_data,
                            simple_offsets[j] + row_start * simple_tensor->nb[1], nbytes,
                            row_گنتی, simple_tensor->nb[1], tensor->nb[1]);
                        offset_data       += nbytes;
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            GGML_ASSERT(offset_data*row_گنتی == size);
            أرجع;
        }
        GGML_ASSERT(split_state.axis == GGML_BACKEND_SPLIT_AXIS_1);

        ثابت size_t row_stride = tensor->nb[2];
        GGML_ASSERT(offset % row_stride == 0);
        GGML_ASSERT(size   % row_stride == 0);
        ثابت int64_t row_start = offset / row_stride;
        ثابت int64_t row_گنتی = size   / row_stride;
        GGML_ASSERT(row_start + row_گنتی <= tensor->ne[2]);

        لكل (size_t s = 0; s < split_state.n_segments; s++) {
            لكل (size_t r = 0; r < split_state.nr[s]; r++) {
                لكل (size_t j = 0; j < n_bufs; j++) {
                    ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                    ثابت size_t nbytes = split_state.ne[s*n_bufs + j] * tensor->nb[1];
                    ggml_backend_tensor_set_2d(simple_tensor, (ثابت حرف *) data + offset_data,
                        simple_offsets[j] + row_start * simple_tensor->nb[2], nbytes,
                        row_گنتی, simple_tensor->nb[2], tensor->nb[2]);
                    offset_data       += nbytes;
                    simple_offsets[j] += nbytes;
                }
            }
        }
        GGML_ASSERT(offset_data*row_گنتی == size);
        أرجع;
    }

    بدل (split_state.axis) {
        حالة GGML_BACKEND_SPLIT_AXIS_0:
        حالة GGML_BACKEND_SPLIT_AXIS_1:
        حالة GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            ثابت size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            ثابت int64_t i_start =  offset        /chunk_size_full;
            ثابت int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            لكل (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ثابت size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                إذا (chunk_size_j == 0) {
                    استمر;
                }
                ثابت size_t simple_offset = i_start * chunk_size_j;
                ggml_backend_tensor_set_2d(simple_tensor, (ثابت حرف *) data + offset_j, simple_offset, chunk_size_j, i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } اكسر;
        حالة GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            لكل (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ggml_backend_tensor_set(simple_tensor, data, offset, size);
            }
        } اكسر;
        حالة GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
            GGML_ASSERT(tensor->type == GGML_TYPE_F32);
            GGML_ASSERT(offset % حجم(عائم) == 0);
            GGML_ASSERT(size   % حجم(عائم) == 0);
            ثابت size_t n_اقدار = size / حجم(عائم);
            size_t n_contributors = 0;
            لكل (size_t j = 0; j < n_bufs; j++) {
                n_contributors += split_state.ne[j] != 0;
            }
            ثابت منطقي has_contributor_mask = n_contributors != 0;
            إذا (!has_contributor_mask) {
                n_contributors = n_bufs;
            }
            std::vector<عائم> tmp(n_اقدار);
            لكل (size_t i = 0; i < n_اقدار; i++) {
                tmp[i] = ((ثابت عائم *) data)[i] / n_contributors;
            }
            std::vector<عائم> zero;
            إذا (has_contributor_mask) {
                zero.resize(n_اقدار, 0.0f);
            }
            لكل (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ثابت عائم * partial = has_contributor_mask && split_state.ne[j] == 0 ? zero.data() : tmp.data();
                ggml_backend_tensor_set(simple_tensor, partial, offset, size);
            }
        } اكسر;
        افتراضي: {
            GGML_ABORT("fatal error");
        }
    }
}

ساكن فراغ ggml_backend_meta_buffer_get_tensor(ggml_backend_buffer_t buffer, ثابت ggml_tensor * tensor, فراغ * data, size_t offset, size_t size) {
    ثابت size_t n_bufs = ggml_backend_meta_buffer_n_bufs(buffer);
    ثابت ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ خطأ);
    GGML_ASSERT(ggml_is_contiguous(tensor) || split_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

    إذا (split_state.n_segments != 1 || split_state.nr[0] != 1) {
        GGML_ASSERT(split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(split_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        size_t offset_data = 0;
        std::vector<size_t> simple_offsets(n_bufs, 0);
        إذا (split_state.axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            ثابت size_t row_stride = tensor->nb[1];
            GGML_ASSERT(offset % row_stride == 0);
            GGML_ASSERT(size   % row_stride == 0);
            ثابت int64_t row_start = offset / row_stride;
            ثابت int64_t row_گنتی = size   / row_stride;
            GGML_ASSERT(row_start + row_گنتی <= tensor->ne[1]);

            ثابت int64_t blck_size = ggml_blck_size(tensor->type);
            لكل (size_t s = 0; s < split_state.n_segments; s++) {
                لكل (size_t r = 0; r < split_state.nr[s]; r++) {
                    لكل (size_t j = 0; j < n_bufs; j++) {
                        ثابت ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                        GGML_ASSERT(split_state.ne[s*n_bufs + j] % blck_size == 0);
                        ثابت size_t nbytes = split_state.ne[s*n_bufs + j]/blck_size * tensor->nb[0];
                        ggml_backend_tensor_get_2d(simple_tensor, (حرف *) data + offset_data,
                            simple_offsets[j] + row_start * simple_tensor->nb[1], nbytes,
                            row_گنتی, simple_tensor->nb[1], tensor->nb[1]);
                        offset_data       += nbytes;
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            GGML_ASSERT(offset_data*row_گنتی == size);
            أرجع;
        }
        GGML_ASSERT(split_state.axis == GGML_BACKEND_SPLIT_AXIS_1);

        ثابت size_t row_stride = tensor->nb[2];
        GGML_ASSERT(offset % row_stride == 0);
        GGML_ASSERT(size   % row_stride == 0);
        ثابت int64_t row_start = offset / row_stride;
        ثابت int64_t row_گنتی = size   / row_stride;
        GGML_ASSERT(row_start + row_گنتی <= tensor->ne[2]);

        لكل (size_t s = 0; s < split_state.n_segments; s++) {
            لكل (size_t r = 0; r < split_state.nr[s]; r++) {
                لكل (size_t j = 0; j < n_bufs; j++) {
                    ثابت ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                    ثابت size_t nbytes = split_state.ne[s*n_bufs + j] * tensor->nb[1];
                    ggml_backend_tensor_get_2d(simple_tensor, (حرف *) data + offset_data,
                        simple_offsets[j] + row_start * simple_tensor->nb[2], nbytes,
                        row_گنتی, simple_tensor->nb[2], tensor->nb[2]);
                    offset_data       += nbytes;
                    simple_offsets[j] += nbytes;
                }
            }
        }
        GGML_ASSERT(offset_data*row_گنتی == size);
        أرجع;
    }

    بدل (split_state.axis) {
        حالة GGML_BACKEND_SPLIT_AXIS_0:
        حالة GGML_BACKEND_SPLIT_AXIS_1:
        حالة GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            ثابت size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            ثابت int64_t i_start =  offset        /chunk_size_full;
            ثابت int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            لكل (size_t j = 0; j < n_bufs; j++){
                ثابت ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ثابت size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                إذا (chunk_size_j == 0) {
                    استمر;
                }
                ثابت size_t simple_offset = i_start * chunk_size_j;
                ggml_backend_tensor_get_2d(simple_tensor, (حرف *) data + offset_j, simple_offset, chunk_size_j, i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } اكسر;
        حالة GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            // TODO other simple backend may be better
            ثابت ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, 0);
            ggml_backend_tensor_get(simple_tensor, data, offset, size);
        } اكسر;
        افتراضي: {
            GGML_ABORT("fatal error");
        }
    }
}

ساكن فراغ ggml_backend_meta_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ثابت size_t n_buffers = ggml_backend_meta_buffer_n_bufs(buffer);
    لكل (size_t i = 0; i < n_buffers; i++) {
        ggml_backend_buffer_clear(ggml_backend_meta_buffer_simple_buffer(buffer, i), value);
    }
}

ساكن فراغ ggml_backend_meta_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    لكل (size_t i = 0; i < buf_ctx->bufs.size(); i++) {
        ggml_backend_buffer_reset(ggml_backend_meta_buffer_simple_buffer(buffer, i));
    }
}

ساكن ثابت ggml_backend_buffer_i ggml_backend_meta_buffer_iface = {
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

منطقي ggml_backend_buffer_is_meta(ggml_backend_buffer_t buf) {
    أرجع buf != nullptr && buf->iface.free_buffer == ggml_backend_meta_buffer_iface.free_buffer;
}

فراغ ggml_backend_meta_buffer_set_usage(ggml_backend_buffer_t buffer, تعداد ggml_backend_buffer_usage usage) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    لكل (size_t i = 0; i < buf_ctx->bufs.size(); i++) {
        إذا (buf_ctx->bufs[i]) {
            ggml_backend_buffer_set_usage(buf_ctx->bufs[i].get(), usage);
        }
    }
}

ساكن ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ثابت size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);

    ثابت ggml_init_params params = {
        /*.mem_size   =*/ 1024*1024*ggml_tensor_overhead(), // FIXME
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_backend_meta_simple_tensor_container stc_static;
    ggml_backend_meta_simple_tensor_container stc_compute_0(params, n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_1(params, n_simple_bufts);

    size_t max_size = 0;
    std::vector<ggml_backend_buffer_t> bufs;
    bufs.reserve(n_simple_bufts);
    لكل (size_t i = 0; i < n_simple_bufts; i++) {
        bufs.push_back(ggml_backend_buft_alloc_buffer(ggml_backend_meta_buft_simple_buft(buft, i), size));
        GGML_ASSERT(bufs.back() != nullptr);
        max_size = std::max(max_size, ggml_backend_buffer_get_size(bufs.back()));
    }
    ggml_backend_meta_buffer_context * buf_ctx = جديد ggml_backend_meta_buffer_context(stc_static, stc_compute_0, stc_compute_1, bufs);

    أرجع ggml_backend_buffer_init(buft, ggml_backend_meta_buffer_iface, buf_ctx, max_size);
}

ساكن ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer_n(ggml_backend_buffer_type_t buft, ggml_tensor ** tensors, صحيح n_tensors) {
    ثابت size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);

    constexpr size_t compute_headroom = 16; // Maximum number of views per statically allocated tensor that can be created between evals.
    ثابت ggml_init_params params_static = {
        /*.mem_size   =*/ n_tensors * ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ثابت ggml_init_params params_compute = {
        /*.mem_size   =*/ compute_headroom * n_tensors * ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_backend_meta_simple_tensor_container stc_static   (params_static,  n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_0(params_compute, n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_1(params_compute, n_simple_bufts);

    std::vector<ggml_backend_buffer_t> bufs(n_simple_bufts, nullptr);
    ggml_backend_meta_buffer_context * meta_buf_ctx = جديد ggml_backend_meta_buffer_context(stc_static, stc_compute_0, stc_compute_1, bufs);

    ggml_backend_buffer_t meta_buf = ggml_backend_buffer_init(buft, ggml_backend_meta_buffer_iface, meta_buf_ctx, 0);
    لكل (صحيح i = 0; i < n_tensors; i++) {
        ggml_tensor * t = tensors[i];
        t->buffer = meta_buf;
        ggml_backend_meta_buffer_init_tensor_impl(meta_buf_ctx->stc_static, t);
        t->data = (فراغ *) 0x2000000000000000; // FIXME
    }
    لكل (size_t i = 0; i < n_simple_bufts; i++) {
        ggml_context * ctx = meta_buf_ctx->stc_static.ctxs[i].get();
        ggml_backend_buffer_type_t simple_buft = ggml_backend_meta_buft_simple_buft(buft, i);

        // If a ggml_context only has zero-sized tensors, ggml_backend_alloc_ctx_tensors_from_buft returns NULL.
        // For those edge cases, allocate a dummy buffer instead.
        منطقي any_nonzero_slice = خطأ;
        لكل (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            إذا (ggml_nelements(t) != 0) {
                any_nonzero_slice = true;
                اكسر;
            }
        }
        إذا (any_nonzero_slice) {
            meta_buf_ctx->bufs[i].reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx, simple_buft));
        } وإلا {
            meta_buf_ctx->bufs[i].reset(ggml_backend_buft_alloc_buffer(simple_buft, 0));
            لكل (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
                t->buffer = meta_buf_ctx->bufs[i].get();
            }
        }
        GGML_ASSERT(meta_buf_ctx->bufs[i]);
        meta_buf->size = std::max(meta_buf->size, ggml_backend_buffer_get_size(meta_buf_ctx->bufs[i].get()));
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
        ggml_cgraph * cgraph_main = nullptr;
        صحيح           offset      = 0; // Node offset vs. original graph

        std::vector<ggml_cgraph *> cgraphs_aux;
    };
    بنية backend_config {
        ggml_backend_t backend;

        std::vector<cgraph_config>           cgraphs;
        std::vector<ggml_tensor *>           nodes;
        std::vector<ggml_backend_buffer_ptr> bufs;

        backend_config(ggml_backend_t backend, ثابت size_t n_reduce_steps) : backend(backend) {
            bufs.resize(n_reduce_steps);
        }
    };
    std::string                 name;
    std::vector<backend_config> backend_configs;
    ggml_context_ptr            ctx;
    std::vector<ggml_cgraph *>  cgraphs_aux;
    std::vector<ggml_tensor *>  nodes_aux;
    size_t                      n_reduce_steps;
    صحيح                         max_nnodes    = 0;
    size_t                      max_tmp_size  = 0;
    size_t                      max_subgraphs = 0;
    size_t                      n_subgraphs   = 0;
    uint64_t                    uid           = 0;

    فراغ *                               comm_ctx       = nullptr;
    ggml_backend_comm_allreduce_tensor_t comm_allreduce = nullptr;

    ggml_backend_meta_context(ggml_backend_dev_t meta_dev, ثابت حرف * params) {
        ثابت size_t n_devs = ggml_backend_meta_dev_n_devs(meta_dev);
        n_reduce_steps = std::ceil(std::log2(n_devs));
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
            backend_configs.emplace_back(simple_backends.back(), n_reduce_steps);
        }
        name += ")";

        إذا (n_devs > 1) {
            ggml_backend_comm_init_t comm_init = (ggml_backend_comm_init_t) ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(ggml_backend_get_device(simple_backends[0])), "ggml_backend_comm_init");
            إذا (comm_init != nullptr) {
                comm_ctx = comm_init(simple_backends.data(), simple_backends.size());
            }
        }
        إذا (comm_ctx != nullptr) {
            comm_allreduce = (ggml_backend_comm_allreduce_tensor_t)
                ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(
                    ggml_backend_get_device(simple_backends[0])), "ggml_backend_comm_allreduce_tensor");
            GGML_ASSERT(comm_allreduce != nullptr);
        }
    }

    ~ggml_backend_meta_context() {
        إذا (comm_ctx != nullptr) {
            ggml_backend_comm_free_t comm_free = (ggml_backend_comm_free_t) ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_configs[0].backend)), "ggml_backend_comm_free");
            GGML_ASSERT(comm_free != nullptr);
            comm_free(comm_ctx);
        }
        لكل (auto & bc : backend_configs) {
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
    delete backend_ctx;
    delete backend;
}

ساكن فراغ ggml_backend_meta_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, ثابت فراغ * data, size_t offset, size_t size) {
    ثابت size_t n_backends = ggml_backend_meta_n_backends(backend);
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    ثابت ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ خطأ);
    GGML_ASSERT(split_state.n_segments == 1);
    GGML_ASSERT(split_state.nr[0]      == 1);

    بدل (split_state.axis) {
        حالة GGML_BACKEND_SPLIT_AXIS_0:
        حالة GGML_BACKEND_SPLIT_AXIS_1:
        حالة GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            ثابت size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            ثابت int64_t i_start =  offset        /chunk_size_full;
            ثابت int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            لكل (size_t j = 0; j < n_backends; j++){
                ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ثابت size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                إذا (chunk_size_j == 0) {
                    استمر;
                }
                ggml_backend_tensor_set_2d_async(simple_backend, simple_tensor, (ثابت حرف *) data + offset_j, offset, chunk_size_j,
                    i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } اكسر;
        حالة GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            لكل (size_t j = 0; j < n_backends; j++) {
                ggml_backend_tensor_set_async(
                    ggml_backend_meta_simple_backend(backend, j), ggml_backend_meta_buffer_simple_tensor(tensor, j), data, offset, size);
            }
        } اكسر;
        افتراضي: {
            GGML_ABORT("fatal error");
        }
    }
}

ساكن فراغ ggml_backend_meta_get_tensor_async(ggml_backend_t backend, ثابت ggml_tensor * tensor, فراغ * data, size_t offset, size_t size) {
    ثابت size_t n_backends = ggml_backend_meta_n_backends(backend);
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    ثابت ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ خطأ);
    GGML_ASSERT(split_state.n_segments == 1);
    GGML_ASSERT(split_state.nr[0]      == 1);

    بدل (split_state.axis) {
        حالة GGML_BACKEND_SPLIT_AXIS_0:
        حالة GGML_BACKEND_SPLIT_AXIS_1:
        حالة GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            ثابت size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            ثابت int64_t i_start =  offset        /chunk_size_full;
            ثابت int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            لكل (size_t j = 0; j < n_backends; j++){
                ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
                ثابت ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ثابت size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                إذا (chunk_size_j == 0) {
                    استمر;
                }
                ggml_backend_tensor_get_2d_async(simple_backend, simple_tensor, (حرف *) data + offset_j, offset, chunk_size_j,
                    i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } اكسر;
        حالة GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            // TODO other simple backend may be better
            ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, 0);
            ثابت ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, 0);
            ggml_backend_tensor_get_async(simple_backend, simple_tensor, data, offset, size);
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

ساكن تعداد ggml_status ggml_backend_meta_graph_compute(ggml_backend_t backend, بنية ggml_cgraph * cgraph) {
    GGML_ASSERT(cgraph->grads == nullptr);
    ثابت size_t n_backends = ggml_backend_meta_n_backends(backend);
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;

    // If the previous cgraph had a defined UID it can be used to skip rebuilding the subgraphs per simple backend.
    ثابت منطقي needs_rebuild = (cgraph->uid == 0) || (cgraph->uid != backend_ctx->uid);

    منطقي max_nnodes_raised = خطأ;
    إذا (cgraph->n_nodes > backend_ctx->max_nnodes) {
        لكل (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            bcj.nodes.resize(cgraph->n_nodes);
            bcj.cgraphs.resize(cgraph->n_nodes);
        }
        backend_ctx->max_nnodes = cgraph->n_nodes;
        max_nnodes_raised = true;
        أكد(needs_rebuild);
    }

    إذا (needs_rebuild) {
        std::set<ggml_backend_buffer_t> used_buffers;
        لكل (صحيح i = 0; i < cgraph->n_leafs; i++) {
            إذا (ggml_backend_buffer_is_meta(cgraph->leafs[i]->buffer)) {
                used_buffers.emplace(cgraph->leafs[i]->buffer);
            }
        }
        لكل (صحيح i = 0; i < cgraph->n_nodes; i++) {
            إذا (ggml_backend_buffer_is_meta(cgraph->nodes[i]->buffer)) {
                used_buffers.emplace(cgraph->nodes[i]->buffer);
            }
        }
        لكل (ggml_backend_buffer_t buf : used_buffers) {
            ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buf->context;
            buf_ctx->stc_compute_index_next = buf_ctx->stc_compute_index ^ 1;
            ggml_backend_meta_simple_tensor_container & stc = buf_ctx->stc_compute[buf_ctx->stc_compute_index_next];
            لكل (ggml_context_ptr & ctx : stc.ctxs) {
                ggml_reset(ctx.get());
            }
            stc.simple_tensors.clear();
        }
        size_t n_subgraphs  = 0;
        size_t max_tmp_size = 0;

        لكل (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];

            لكل (صحيح i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * گرہ = cgraph->nodes[i];
                إذا (!ggml_backend_buffer_is_meta(گرہ->buffer)) {
                    // [TAG_META_HOST_VIEWS]
                    GGML_ASSERT(ggml_backend_meta_is_host_view(گرہ));

                    // keep the node as is, mapping it to a simple tensor is not possible
                    bcj.nodes[i] = گرہ;
                    استمر;
                }
                bcj.nodes[i] = ggml_backend_meta_buffer_simple_tensor(گرہ, j);
                GGML_ASSERT(bcj.nodes[i]);
            }
        }

        {
            // For MoE models it may make sense to delay the AllReduce in order to reduce I/O:
            auto get_i_delayed_branch = [&](ثابت صحيح i) -> صحيح {
                صحيح id = i; // i_delayed
                صحيح idr = i; // i_delayed return, last safe return value

                ggml_tensor * گرہ = cgraph->nodes[id];
                int32_t n_used = ggml_گرہ_get_use_گنتی(cgraph, id);

                // Skip MIRRORED nodes that don't consume node
                auto skip_unrelated = [&]() {
                    طالما (id + 1 < cgraph->n_nodes) {
                        ggml_tensor * next = cgraph->nodes[id+1];
                        إذا (ggml_backend_meta_get_split_state(next, خطأ).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                            اكسر;
                        }
                        منطقي safe = true;
                        لكل (صحيح s = 0; s < GGML_MAX_SRC; s++) {
                            إذا (next->src[s] == nullptr) {
                                استمر;
                            }
                            إذا (next->src[s] == گرہ) {
                                safe = خطأ;
                                اكسر;
                            }
                            إذا (ggml_backend_meta_get_split_state(next->src[s], خطأ).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                                safe = خطأ;
                                اكسر;
                            }
                        }
                        إذا (!safe) {
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
                    ggml_tensor * next = cgraph->nodes[id+1];
                    إذا (next->op == GGML_OP_ADD_ID && next->src[0] == گرہ &&
                            ggml_backend_meta_get_split_state(next->src[1], خطأ).axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL &&
                            ggml_backend_meta_get_split_state(next->src[2], خطأ).axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                        گرہ = next;
                        id++;
                        idr = id;
                        n_used = ggml_گرہ_get_use_گنتی(cgraph, id);
                    }
                }
                // Chain of MULs with MIRRORED src[1]
                طالما (true) {
                    skip_unrelated();
                    إذا (id + 1 >= cgraph->n_nodes) {
                        أرجع idr;
                    }
                    ggml_tensor * next = cgraph->nodes[id+1];
                    إذا (next->op == GGML_OP_MUL && next->src[0] == گرہ &&
                            ggml_backend_meta_get_split_state(next->src[1], خطأ).axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                        گرہ = next;
                        id++;
                        idr = id;
                        n_used = ggml_گرہ_get_use_گنتی(cgraph, id);
                    } وإلا {
                        اكسر;
                    }
                }

                إذا (n_used != گرہ->ne[1] || id + 2*n_used-1 >= cgraph->n_nodes) {
                    أرجع idr;
                }
                لكل (int32_t k = 0; k < n_used; k++) {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    إذا (next->op != GGML_OP_VIEW || next->view_src != گرہ || next->view_offs != k*گرہ->nb[1] ||
                            next->ne[0] != گرہ->ne[0] || next->ne[1] != گرہ->ne[2] || next->nb[1] != گرہ->nb[2] ||
                            ggml_گرہ_get_use_گنتی(cgraph, id+1) != 1) {
                        أرجع idr;
                    }
                    id++;
                }
                {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    إذا (next->op != GGML_OP_ADD || next->src[0] != cgraph->nodes[id - (n_used-1)] ||
                            next->src[1] != cgraph->nodes[id - (n_used-2)] || ggml_گرہ_get_use_گنتی(cgraph, id+1) != 1) {
                        أرجع idr;
                    }
                    id++;
                }
                لكل (int32_t k = 0; k < n_used - 2; k++) {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    إذا (next->op != GGML_OP_ADD || next->src[0] != cgraph->nodes[id] ||
                            next->src[1] != cgraph->nodes[id - (n_used-2)] || ggml_گرہ_get_use_گنتی(cgraph, id+1) != 1) {
                        أرجع idr;
                    }
                    id++;
                }
                idr = id;
                أرجع idr;
            };

            // AllReduce(a) + AllReduce(b) == AllReduce(a + b) for independent partial branches.
            auto get_i_delayed = [&](ثابت صحيح i) -> صحيح {
                ثابت صحيح i_delayed = get_i_delayed_branch(i);
                ggml_tensor * گرہ = cgraph->nodes[i_delayed];

                إذا (ggml_گرہ_get_use_گنتی(cgraph, i_delayed) != 1) {
                    أرجع i_delayed;
                }

                لكل (صحيح id = i_delayed + 1; id < cgraph->n_nodes; id++) {
                    ggml_tensor * next = cgraph->nodes[id];
                    إذا (next->view_src == گرہ) {
                        أرجع i_delayed;
                    }
                    لكل (صحيح s = 0; s < GGML_MAX_SRC; s++) {
                        إذا (next->src[s] == گرہ) {
                            أرجع i_delayed;
                        }
                    }

                    إذا (next->view_src != nullptr && next->view_src->op == GGML_OP_NONE && ggml_backend_buffer_is_host(next->view_src->buffer)) {
                        استمر;
                    }
                    إذا (ggml_backend_meta_get_split_state(next, خطأ).axis != GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                        استمر;
                    }

                    ثابت صحيح i_other = id;
                    ثابت صحيح i_other_delayed = get_i_delayed_branch(i_other);
                    ggml_tensor * other = cgraph->nodes[i_other_delayed];
                    إذا (ggml_گرہ_get_use_گنتی(cgraph, i_other_delayed) != 1 || i_other_delayed + 1 >= cgraph->n_nodes) {
                        أرجع i_delayed;
                    }

                    ggml_tensor * sum = cgraph->nodes[i_other_delayed + 1];
                    إذا (sum->op != GGML_OP_ADD ||
                            !ggml_are_same_shape(گرہ, other) || گرہ->type != other->type || sum->type != گرہ->type ||
                            !((sum->src[0] == گرہ && sum->src[1] == other) ||
                              (sum->src[0] == other && sum->src[1] == گرہ)) ||
                            ggml_backend_meta_get_split_state(sum, خطأ).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                        أرجع i_delayed;
                    }

                    لكل (size_t j = 0; j < n_backends; j++) {
                        auto & bcj = backend_ctx->backend_configs[j];
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

            صحيح i_start = 0;
            لكل (صحيح i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * گرہ = cgraph->nodes[i];
                إذا (گرہ->view_src != nullptr && گرہ->view_src->op == GGML_OP_NONE && ggml_backend_buffer_is_host(گرہ->view_src->buffer)) {
                    استمر;
                }
                ثابت ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(گرہ, /*assume_sync =*/ خطأ);
                إذا (split_state.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                    max_tmp_size = std::max(max_tmp_size, ggml_nbytes(گرہ));
                }
                ثابت منطقي new_subgraph = i + 1 == cgraph->n_nodes || split_state.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL;
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
                        auto & bcj = backend_ctx->backend_configs[j];
                        إذا ((bcj.nodes[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                            لكل (صحيح ii = i + 1; ii <= i_delayed; ii++) {
                                bcj.nodes[ii]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
                            }
                        }
                    }
                }

                i = i_delayed;

                لكل (size_t j = 0; j < n_backends; j++) {
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

        إذا (max_tmp_size > backend_ctx->max_tmp_size) {
            لكل (size_t j = 0; j < n_backends; j++) {
                auto & bcj = backend_ctx->backend_configs[j];
                لكل (size_t i = 0; i < backend_ctx->n_reduce_steps; i++) {
                    bcj.bufs[i].reset(ggml_backend_alloc_buffer(bcj.backend, max_tmp_size));
                }
            }
            backend_ctx->max_tmp_size = max_tmp_size;
        }

        إذا (max_nnodes_raised || n_subgraphs > backend_ctx->max_subgraphs) {
            backend_ctx->max_subgraphs = std::max(backend_ctx->max_subgraphs, n_subgraphs);
            ثابت size_t n_nodes_per_device = 3 * backend_ctx->n_reduce_steps; // tmp + ADD (+zeroing) graph per step and device
            ثابت size_t n_cgraphs_per_device = 2 * backend_ctx->n_reduce_steps; // ADD ( + zeroing) graph per step and device
            ثابت size_t mem_per_device_graphs_main = backend_ctx->max_subgraphs*ggml_graph_overhead_custom(backend_ctx->max_nnodes, cgraph->grads);
            ثابت size_t mem_per_device_graphs_aux = n_cgraphs_per_device*backend_ctx->max_subgraphs*ggml_graph_overhead_custom(1, cgraph->grads);
            ثابت size_t mem_per_device_nodes_aux = n_nodes_per_device*backend_ctx->max_subgraphs*ggml_tensor_overhead();
            ثابت ggml_init_params params = {
                /*.mem_size   =*/ n_backends * (mem_per_device_graphs_main + mem_per_device_graphs_aux + mem_per_device_nodes_aux),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            backend_ctx->ctx.reset(ggml_init(params));
            لكل (size_t j = 0; j < n_backends; j++) {
                auto & bcj = backend_ctx->backend_configs[j];
                لكل (size_t i = 0; i < backend_ctx->max_subgraphs; i++) {
                    bcj.cgraphs[i].cgraph_main = ggml_new_graph_custom(backend_ctx->ctx.get(), backend_ctx->max_nnodes, /*grads =*/ خطأ);
                }
            }
            backend_ctx->cgraphs_aux.resize(n_backends*n_cgraphs_per_device*backend_ctx->max_subgraphs);
            لكل (size_t k = 0; k < backend_ctx->cgraphs_aux.size(); k++) {
                backend_ctx->cgraphs_aux[k] = ggml_new_graph_custom(backend_ctx->ctx.get(), 1, cgraph->grads);
            }
            backend_ctx->nodes_aux.resize(n_backends*n_nodes_per_device*backend_ctx->max_subgraphs);
            لكل (size_t k = 0; k < backend_ctx->nodes_aux.size(); k++) {
                backend_ctx->nodes_aux[k] = ggml_new_tensor_1d(backend_ctx->ctx.get(), GGML_TYPE_F32, 1);
            }
        }

        لكل (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            لكل (size_t i_graph = 0; i_graph < n_subgraphs; i_graph++) {
                ggml_cgraph * cgraph_ij = bcj.cgraphs[i_graph].cgraph_main;
                ثابت size_t i_گرہ_start = bcj.cgraphs[i_graph].offset;
                ثابت size_t i_گرہ_stop = i_graph + 1 < n_subgraphs ? bcj.cgraphs[i_graph + 1].offset : cgraph->n_nodes;
                cgraph_ij->n_nodes = i_گرہ_stop - i_گرہ_start;
                ggml_hash_set_reset(&cgraph_ij->visited_hash_set);
                لكل (size_t i_گرہ = i_گرہ_start; i_گرہ < i_گرہ_stop; i_گرہ++) {
                    ggml_tensor * گرہ_ij = bcj.nodes[i_گرہ];
                    cgraph_ij->nodes[i_گرہ - i_گرہ_start] = گرہ_ij;
                    ثابت size_t hash_pos_orig = ggml_hash_find(&cgraph->visited_hash_set, cgraph->nodes[i_گرہ]);
                    ثابت size_t hash_pos_ij = ggml_hash_داخل_کرو(&cgraph_ij->visited_hash_set, گرہ_ij);
                    cgraph_ij->use_counts[hash_pos_ij] = cgraph->use_counts[hash_pos_orig];
                }
                cgraph_ij->uid = ggml_graph_next_uid();
            }
        }
    }

    size_t iga = 0; // i graph aux
    size_t ina = 0; // i node aux

    auto get_گرہ_aux = [&](ggml_tensor * t) -> ggml_tensor * {
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
    auto set_tmp_data = [&](ggml_tensor * tensor, ثابت size_t j, ثابت size_t i_buf) {
        auto & bcj = backend_ctx->backend_configs[j];
        ggml_backend_buffer_ptr & buf_ptr = bcj.bufs[i_buf];
        إذا (!buf_ptr || ggml_backend_buffer_get_size(buf_ptr.get()) < backend_ctx->max_tmp_size) {
            buf_ptr.reset(ggml_backend_alloc_buffer(bcj.backend, backend_ctx->max_tmp_size));
        }
        tensor->buffer = buf_ptr.get();
        tensor->data   = ggml_backend_buffer_get_base(buf_ptr.get());
    };
    // FIXME usage_counts
    auto get_cgraph_aux = [&]() -> ggml_cgraph * {
        ggml_cgraph * ret = backend_ctx->cgraphs_aux[iga++];
        أرجع ret;
    };

    // Preferentially use backend-specific allreduce_tensor_async (e.g. NCCL for CUDA), use a generic fallback if unavailable:
    auto allreduce_fallback = [&](size_t i) -> ggml_status {
        std::vector<ggml_cgraph *> step_cgraphs(n_backends, nullptr);

        // Zero out nodes that were disabled due to having a zero-sized slice:
        لكل (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            ggml_tensor * گرہ = bcj.cgraphs[i].cgraph_main->nodes[bcj.cgraphs[i].cgraph_main->n_nodes - 1];
            إذا (گرہ->flags & GGML_TENSOR_FLAG_COMPUTE) {
                استمر;
            }
            إذا (ggml_nelements(گرہ) > 0) {
                ggml_tensor * گرہ_zero = get_گرہ_aux(گرہ);
                گرہ_zero->op = GGML_OP_FILL;
                گرہ_zero->src[0] = گرہ; // only used for the shape, the data is not read
                ggml_set_op_params_f32(گرہ_zero, 0, 0.0f);
                گرہ_zero->data = گرہ->data;
                گرہ_zero->buffer = گرہ->buffer;
                گرہ_zero->flags |= GGML_TENSOR_FLAG_COMPUTE;

                step_cgraphs[j] = get_cgraph_aux();
                step_cgraphs[j]->nodes[0] = گرہ_zero;
                step_cgraphs[j]->n_nodes = 1;
                ثابت ggml_status status = ggml_backend_graph_compute_async(bcj.backend, step_cgraphs[j]);
                إذا (status != GGML_STATUS_SUCCESS) {
                    أرجع status;
                }
            }
        }
        std::fill(step_cgraphs.begin(), step_cgraphs.end(), nullptr);

        auto ڈالو_data = [&](ثابت size_t j_src, ثابت size_t j_dst, ثابت size_t i_buf) {
            أكد(step_cgraphs[j_dst] == nullptr);
            auto & bcj_src = backend_ctx->backend_configs[j_src];
            auto & bcj_dst = backend_ctx->backend_configs[j_dst];

            ggml_tensor * گرہ_src = bcj_src.cgraphs[i].cgraph_main->nodes[bcj_src.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_tensor * گرہ_dst = bcj_dst.cgraphs[i].cgraph_main->nodes[bcj_dst.cgraphs[i].cgraph_main->n_nodes - 1];
            GGML_ASSERT(ggml_is_contiguous(گرہ_src));
            GGML_ASSERT(ggml_is_contiguous(گرہ_dst));

            ggml_tensor * گرہ_tmp = get_گرہ_aux(گرہ_dst);
            set_tmp_data(گرہ_tmp, j_dst, i_buf);

            ggml_backend_tensor_copy_async(bcj_src.backend, bcj_dst.backend, گرہ_src, گرہ_tmp);

            ggml_tensor * گرہ_red = get_گرہ_aux(گرہ_dst);
            گرہ_red->view_src = گرہ_dst->view_src == nullptr ? گرہ_dst : گرہ_dst->view_src;
            گرہ_red->view_offs = گرہ_dst->view_offs;
            گرہ_red->op = GGML_OP_ADD;
            گرہ_red->src[0] = گرہ_dst;
            گرہ_red->src[1] = گرہ_tmp;
            گرہ_red->flags |= GGML_TENSOR_FLAG_COMPUTE;
            ggml_backend_view_init(گرہ_red);

            ggml_cgraph * cgraph_aux = get_cgraph_aux();
            cgraph_aux->nodes[0] = گرہ_red;
            cgraph_aux->n_nodes = 1;
            step_cgraphs[j_dst] = cgraph_aux;
        };

        size_t offset_j = n_backends/2;
        طالما ((offset_j & (offset_j - 1)) != 0) {
            offset_j--;
        }
        ثابت size_t offset_j_max = offset_j;
        size_t i_buf = 0;

        // If n_backends is not a power of 2, fold in the excess prior to butterfly reduction:
        لكل (size_t j_src = 2*offset_j_max; j_src < n_backends; j_src++) {
            ثابت size_t j_dst = j_src - 2*offset_j_max;
            ڈالو_data(j_src, j_dst, i_buf);
            ثابت ggml_status status = ggml_backend_graph_compute_async(backend_ctx->backend_configs[j_dst].backend, step_cgraphs[j_dst]);
            إذا (status != GGML_STATUS_SUCCESS) {
                أرجع status;
            }
            i_buf = 1;
        }

        // Butterfly reduction:
        لكل (; offset_j >= 1; offset_j /= 2) {
            std::fill(step_cgraphs.begin(), step_cgraphs.end(), nullptr);

            لكل (size_t j = 0; j < 2*offset_j_max; j++) {
                ثابت size_t j_other = j ^ offset_j;
                إذا (j_other >= n_backends) {
                    استمر;
                }
                ڈالو_data(j, j_other, i_buf);
            }

            لكل (size_t j = 0; j < 2*offset_j_max; j++) {
                إذا (step_cgraphs[j] == nullptr) {
                    استمر;
                }
                auto & bcj = backend_ctx->backend_configs[j];
                ثابت ggml_status status = ggml_backend_graph_compute_async(bcj.backend, step_cgraphs[j]);
                إذا (status != GGML_STATUS_SUCCESS) {
                    أرجع status;
                }
            }
            i_buf++;
        }
        أكد(i_buf == backend_ctx->n_reduce_steps);

        // If n_backends is not a power of 2, copy back the reduced tensors to the excess:
        لكل (size_t j = 2*offset_j_max; j < n_backends; j++) {
            auto & bcj_src = backend_ctx->backend_configs[j - 2*offset_j_max];
            auto & bcj_dst = backend_ctx->backend_configs[j];

            ggml_tensor * گرہ_src = bcj_src.cgraphs[i].cgraph_main->nodes[bcj_src.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_tensor * گرہ_dst = bcj_dst.cgraphs[i].cgraph_main->nodes[bcj_dst.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_backend_tensor_copy_async(bcj_src.backend, bcj_dst.backend, گرہ_src, گرہ_dst);
        }

        أرجع GGML_STATUS_SUCCESS;
    };


    لكل (size_t i = 0; i < backend_ctx->n_subgraphs; i++) {
        لكل (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            ثابت ggml_status status = ggml_backend_graph_compute_async(bcj.backend, bcj.cgraphs[i].cgraph_main);
            إذا (status != GGML_STATUS_SUCCESS) {
                أرجع status;
            }
        }

        إذا (n_backends > 1 && i < backend_ctx->n_subgraphs - 1) {
            منطقي backend_allreduce_success = خطأ;
            إذا (backend_ctx->comm_ctx) {
                std::vector<ggml_tensor *> nodes;
                nodes.reserve(n_backends);
                لكل (size_t j = 0; j < n_backends; j++) {
                    auto & bcj = backend_ctx->backend_configs[j];
                    ggml_cgraph * cgraph_ij = bcj.cgraphs[i].cgraph_main;
                    nodes.push_back(cgraph_ij->nodes[cgraph_ij->n_nodes-1]);
                }
                backend_allreduce_success = backend_ctx->comm_allreduce(backend_ctx->comm_ctx, nodes.data());
            }

            إذا (!backend_allreduce_success) {
                ثابت ggml_status status = allreduce_fallback(i);
                إذا (status != GGML_STATUS_SUCCESS) {
                    أرجع status;
                }
            }
        }
    }
    أرجع GGML_STATUS_SUCCESS;
}

ساكن ثابت ggml_backend_i ggml_backend_meta_i = {
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

منطقي ggml_backend_is_meta(ggml_backend_t backend) {
    أرجع backend != nullptr && backend->iface.get_name == ggml_backend_meta_i.get_name;
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

ggml_backend_t ggml_backend_meta_simple_backend(ggml_backend_t meta_backend, size_t index) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    ثابت ggml_backend_meta_context * backend_ctx = (ثابت ggml_backend_meta_context *) meta_backend->context;
    أرجع backend_ctx->backend_configs[index].backend;
}
