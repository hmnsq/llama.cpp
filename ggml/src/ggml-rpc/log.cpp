#அந்தர்பாவ "log.h"

#அந்தர்பாவ <cinttypes>
#அந்தர்பாவ <cstdio>

std::string rpc_log_format_size(size_t size) {
    நிலையான மாறிலி எழுத்து * units[] = { "B", "KiB", "MiB", "GiB", "TiB", "PiB" };

    இரட்டை value = (இரட்டை) size;
    size_t unit  = 0;
    போது (value >= 1024.0 && unit + 1 < அளவு(units) / அளவு(units[0])) {
        value /= 1024.0;
        unit++;
    }

    எழுத்து buf[32];
    எனில் (unit == 0) {
        snprintf(buf, அளவு(buf), "%zu B", size);
    } இல்லையெனில் {
        snprintf(buf, அளவு(buf), "%.2f %s", value, units[unit]);
    }
    திரும்பு std::string(buf);
}

std::string rpc_log_format_tensor(மாறிலி ggml_tensor * tensor) {
    எழுத்து shape[64];
    snprintf(shape, அளவு(shape), "[%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]", tensor->ne[0], tensor->ne[1],
             tensor->ne[2], tensor->ne[3]);

    எழுத்து buf[GGML_MAX_NAME + 192];
    snprintf(buf, அளவு(buf), "%s [%s, %s, %s]", ggml_get_name(tensor), ggml_op_name(tensor->op),
             ggml_type_name(tensor->type), shape);
    திரும்பு std::string(buf);
}
