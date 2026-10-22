#ضمن "log.h"

#ضمن <cinttypes>
#ضمن <cstdio>

std::string rpc_log_format_ناپ(size_t size) {
    ساكن ثابت حرف * units[] = { "B", "KiB", "MiB", "GiB", "TiB", "PiB" };

    مزدوج value = (مزدوج) size;
    size_t unit  = 0;
    طالما (value >= 1024.0 && unit + 1 < حجم(units) / حجم(units[0])) {
        value /= 1024.0;
        unit++;
    }

    حرف buf[32];
    إذا (unit == 0) {
        snprintf(buf, حجم(buf), "%zu B", size);
    } وإلا {
        snprintf(buf, حجم(buf), "%.2f %s", value, units[unit]);
    }
    أرجع std::string(buf);
}

std::string rpc_log_format_tensor(ثابت ggml_tensor * tensor) {
    حرف shape[64];
    snprintf(shape, حجم(shape), "[%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]", tensor->ne[0], tensor->ne[1],
             tensor->ne[2], tensor->ne[3]);

    حرف buf[GGML_MAX_NAME + 192];
    snprintf(buf, حجم(buf), "%s [%s, %s, %s]", ggml_get_name(tensor), ggml_op_name(tensor->op),
             ggml_type_name(tensor->type), shape);
    أرجع std::string(buf);
}
