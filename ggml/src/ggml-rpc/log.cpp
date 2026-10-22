#شامل "log.h"

#شامل <cinttypes>
#شامل <cstdio>

std::string rpc_log_format_size(size_t size) {
    ایستا ثابت نویسه * units[] = { "B", "KiB", "MiB", "GiB", "TiB", "PiB" };

    دوگانه value = (دوگانه) size;
    size_t unit  = 0;
    وقتی (value >= 1024.0 && unit + 1 < اندازه(units) / اندازه(units[0])) {
        value /= 1024.0;
        unit++;
    }

    نویسه buf[32];
    اگر (unit == 0) {
        snprintf(buf, اندازه(buf), "%zu B", size);
    } وگرنه {
        snprintf(buf, اندازه(buf), "%.2f %s", value, units[unit]);
    }
    بازگشت std::string(buf);
}

std::string rpc_log_format_tensor(ثابت ggml_tensor * tensor) {
    نویسه shape[64];
    snprintf(shape, اندازه(shape), "[%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]", tensor->ne[0], tensor->ne[1],
             tensor->ne[2], tensor->ne[3]);

    نویسه buf[GGML_MAX_NAME + 192];
    snprintf(buf, اندازه(buf), "%s [%s, %s, %s]", ggml_get_name(tensor), ggml_op_name(tensor->op),
             ggml_type_name(tensor->type), shape);
    بازگشت std::string(buf);
}
