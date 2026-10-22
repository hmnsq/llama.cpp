#অন্তর্ভাব "log.h"

#অন্তর্ভাব <cinttypes>
#অন্তর্ভাব <cstdio>

std::string rpc_log_format_আকার(size_t size) {
    স্থির ধ্রুবক অক্ষর * units[] = { "B", "KiB", "MiB", "GiB", "TiB", "PiB" };

    দ্বিগুণ value = (দ্বিগুণ) size;
    size_t unit  = 0;
    যখন (value >= 1024.0 && unit + 1 < আকার(units) / আকার(units[0])) {
        value /= 1024.0;
        unit++;
    }

    অক্ষর buf[32];
    যদি (unit == 0) {
        snprintf(buf, আকার(buf), "%zu B", size);
    } অন্যথা {
        snprintf(buf, আকার(buf), "%.2f %s", value, units[unit]);
    }
    ফেরত std::string(buf);
}

std::string rpc_log_format_tensor(ধ্রুবক ggml_tensor * tensor) {
    অক্ষর shape[64];
    snprintf(shape, আকার(shape), "[%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]", tensor->ne[0], tensor->ne[1],
             tensor->ne[2], tensor->ne[3]);

    অক্ষর buf[GGML_MAX_NAME + 192];
    snprintf(buf, আকার(buf), "%s [%s, %s, %s]", ggml_get_name(tensor), ggml_op_name(tensor->op),
             ggml_type_name(tensor->type), shape);
    ফেরত std::string(buf);
}
