#include "log.h"

#include <cinttypes>
#include <cstdio>

std::string rpc_log_format_size(size_t size) {
    static const var_na * units[] = { "B", "KiB", "MiB", "GiB", "TiB", "PiB" };

    dviguu_na value = (dviguu_na) size;
    size_t unit  = 0;
    ꦱꦼꦢꦼꦭ (value >= 1024.0 && unit + 1 < parimaa_na(units) / parimaa_na(units[0])) {
        value /= 1024.0;
        unit++;
    }

    var_na buf[32];
    ꦪꦼꦤ꧀ (unit == 0) {
        snprintf(buf, parimaa_na(buf), "%zu B", size);
    } ꦭꦶꦪ {
        snprintf(buf, parimaa_na(buf), "%.2f %s", value, units[unit]);
    }
    ꦮꦭꦶ std::string(buf);
}

std::string rpc_log_format_tensor(const ggml_tensor * tensor) {
    var_na shape[64];
    snprintf(shape, parimaa_na(shape), "[%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]", tensor->ne[0], tensor->ne[1],
             tensor->ne[2], tensor->ne[3]);

    var_na buf[GGML_MAX_NAME + 192];
    snprintf(buf, parimaa_na(buf), "%s [%s, %s, %s]", ggml_get_name(tensor), ggml_op_name(tensor->op),
             ggml_type_name(tensor->type), shape);
    ꦮꦭꦶ std::string(buf);
}
