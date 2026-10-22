#અન્તર્ભાવ "log.h"

#અન્તર્ભાવ <cinttypes>
#અન્તર્ભાવ <cstdio>

std::string rpc_log_format_size(size_t size) {
    સ્થિર અચલ અક્ષર * units[] = { "B", "KiB", "MiB", "GiB", "TiB", "PiB" };

    દ્વિગુણ value = (દ્વિગુણ) size;
    size_t unit  = 0;
    જ્યારે (value >= 1024.0 && unit + 1 < કદ(units) / કદ(units[0])) {
        value /= 1024.0;
        unit++;
    }

    અક્ષર buf[32];
    જો (unit == 0) {
        snprintf(buf, કદ(buf), "%zu B", size);
    } નહીંતર {
        snprintf(buf, કદ(buf), "%.2f %s", value, units[unit]);
    }
    પાછા std::string(buf);
}

std::string rpc_log_format_tensor(અચલ ggml_tensor * tensor) {
    અક્ષર shape[64];
    snprintf(shape, કદ(shape), "[%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]", tensor->ne[0], tensor->ne[1],
             tensor->ne[2], tensor->ne[3]);

    અક્ષર buf[GGML_MAX_NAME + 192];
    snprintf(buf, કદ(buf), "%s [%s, %s, %s]", ggml_get_name(tensor), ggml_op_name(tensor->op),
             ggml_type_name(tensor->type), shape);
    પાછા std::string(buf);
}
