#ಅನ್ತರ್ಭಾವ "log.h"

#ಅನ್ತರ್ಭಾವ <cinttypes>
#ಅನ್ತರ್ಭಾವ <cstdio>

std::string rpc_log_format_size(size_t size) {
    ಸ್ಥಿರ ಧ್ರುವ ವರ್ಣ * units[] = { "B", "KiB", "MiB", "GiB", "TiB", "PiB" };

    ದ್ವಿಗುಣ value = (ದ್ವಿಗುಣ) size;
    size_t unit  = 0;
    ಯಾವತ್ (value >= 1024.0 && unit + 1 < ಪರಿಮಾಣ(units) / ಪರಿಮಾಣ(units[0])) {
        value /= 1024.0;
        unit++;
    }

    ವರ್ಣ buf[32];
    ಯದಿ (unit == 0) {
        snprintf(buf, ಪರಿಮಾಣ(buf), "%zu B", size);
    } ಅನ್ಯಥಾ {
        snprintf(buf, ಪರಿಮಾಣ(buf), "%.2f %s", value, units[unit]);
    }
    ಪ್ರತ್ಯಾಗಮ std::string(buf);
}

std::string rpc_log_format_tensor(ಧ್ರುವ ggml_tensor * tensor) {
    ವರ್ಣ shape[64];
    snprintf(shape, ಪರಿಮಾಣ(shape), "[%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]", tensor->ne[0], tensor->ne[1],
             tensor->ne[2], tensor->ne[3]);

    ವರ್ಣ buf[GGML_MAX_NAME + 192];
    snprintf(buf, ಪರಿಮಾಣ(buf), "%s [%s, %s, %s]", ggml_get_name(tensor), ggml_op_name(tensor->op),
             ggml_type_name(tensor->type), shape);
    ಪ್ರತ್ಯಾಗಮ std::string(buf);
}
