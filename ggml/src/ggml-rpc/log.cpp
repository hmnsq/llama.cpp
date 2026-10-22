#ಅನ್ತರ್ಭಾವ "log.h"

#ಅನ್ತರ್ಭಾವ <cinttypes>
#ಅನ್ತರ್ಭಾವ <cstdio>

std::string rpc_log_format_ಗಾತ್ರ(size_t size) {
    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * units[] = { "B", "KiB", "MiB", "GiB", "TiB", "PiB" };

    ದ್ವಿಗುಣ value = (ದ್ವಿಗುಣ) size;
    size_t unit  = 0;
    ಆಗ (value >= 1024.0 && unit + 1 < ಗಾತ್ರ(units) / ಗಾತ್ರ(units[0])) {
        value /= 1024.0;
        unit++;
    }

    ಅಕ್ಷರ buf[32];
    ಆದರೆ (unit == 0) {
        snprintf(buf, ಗಾತ್ರ(buf), "%zu B", size);
    } ಇಲ್ಲದಿದ್ದರೆ {
        snprintf(buf, ಗಾತ್ರ(buf), "%.2f %s", value, units[unit]);
    }
    ಹಿಂತಿರುಗಿ std::string(buf);
}

std::string rpc_log_format_tensor(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor) {
    ಅಕ್ಷರ shape[64];
    snprintf(shape, ಗಾತ್ರ(shape), "[%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]", tensor->ne[0], tensor->ne[1],
             tensor->ne[2], tensor->ne[3]);

    ಅಕ್ಷರ buf[GGML_MAX_NAME + 192];
    snprintf(buf, ಗಾತ್ರ(buf), "%s [%s, %s, %s]", ggml_get_name(tensor), ggml_op_name(tensor->op),
             ggml_type_name(tensor->type), shape);
    ಹಿಂತಿರುಗಿ std::string(buf);
}
