#അന്തര്ഭാവ "log.h"

#അന്തര്ഭാവ <cinttypes>
#അന്തര്ഭാവ <cstdio>

std::string rpc_log_format_size(size_t size) {
    സ്ഥിതം സ്ഥിരം അക്ഷരം * units[] = { "B", "KiB", "MiB", "GiB", "TiB", "PiB" };

    ഇരട്ട value = (ഇരട്ട) size;
    size_t unit  = 0;
    ആകുമ്പോൾ (value >= 1024.0 && unit + 1 < വലുപ്പം(units) / വലുപ്പം(units[0])) {
        value /= 1024.0;
        unit++;
    }

    അക്ഷരം buf[32];
    എങ്കിൽ (unit == 0) {
        snprintf(buf, വലുപ്പം(buf), "%zu B", size);
    } അല്ലെങ്കിൽ {
        snprintf(buf, വലുപ്പം(buf), "%.2f %s", value, units[unit]);
    }
    തിരികെ std::string(buf);
}

std::string rpc_log_format_tensor(സ്ഥിരം ggml_tensor * tensor) {
    അക്ഷരം shape[64];
    snprintf(shape, വലുപ്പം(shape), "[%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]", tensor->ne[0], tensor->ne[1],
             tensor->ne[2], tensor->ne[3]);

    അക്ഷരം buf[GGML_MAX_NAME + 192];
    snprintf(buf, വലുപ്പം(buf), "%s [%s, %s, %s]", ggml_get_name(tensor), ggml_op_name(tensor->op),
             ggml_type_name(tensor->type), shape);
    തിരികെ std::string(buf);
}
