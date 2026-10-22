#ਅਨ੍ਤਰ੍ਭਾਵ "log.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <cinttypes>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdio>

std::string rpc_log_format_ਮਾਪ(size_t size) {
    ਸਥਿਰ ਅਚਲ ਅੱਖਰ * units[] = { "B", "KiB", "MiB", "GiB", "TiB", "PiB" };

    ਦੁੱਗਣਾ value = (ਦੁੱਗਣਾ) size;
    size_t unit  = 0;
    ਜਦੋਂ (value >= 1024.0 && unit + 1 < ਆਕਾਰ(units) / ਆਕਾਰ(units[0])) {
        value /= 1024.0;
        unit++;
    }

    ਅੱਖਰ buf[32];
    ਜੇ (unit == 0) {
        snprintf(buf, ਆਕਾਰ(buf), "%zu B", size);
    } ਨਹੀਂਤਾਂ {
        snprintf(buf, ਆਕਾਰ(buf), "%.2f %s", value, units[unit]);
    }
    ਵਾਪਸ std::string(buf);
}

std::string rpc_log_format_tensor(ਅਚਲ ggml_tensor * tensor) {
    ਅੱਖਰ shape[64];
    snprintf(shape, ਆਕਾਰ(shape), "[%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]", tensor->ne[0], tensor->ne[1],
             tensor->ne[2], tensor->ne[3]);

    ਅੱਖਰ buf[GGML_MAX_NAME + 192];
    snprintf(buf, ਆਕਾਰ(buf), "%s [%s, %s, %s]", ggml_get_name(tensor), ggml_op_name(tensor->op),
             ggml_type_name(tensor->type), shape);
    ਵਾਪਸ std::string(buf);
}
