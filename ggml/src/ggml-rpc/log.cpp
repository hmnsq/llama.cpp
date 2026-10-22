#অন্তর্ভাৱ "log.h"

#অন্তর্ভাৱ <cinttypes>
#অন্তর্ভাৱ <cstdio>

std::string rpc_log_format_size(size_t size) {
    স্থির ধ্রুৱ ৱর্ণ * units[] = { "B", "KiB", "MiB", "GiB", "TiB", "PiB" };

    দ্ৱিগুণ value = (দ্ৱিগুণ) size;
    size_t unit  = 0;
    যেতিয়ালৈ (value >= 1024.0 && unit + 1 < পরিমাণ(units) / পরিমাণ(units[0])) {
        value /= 1024.0;
        unit++;
    }

    ৱর্ণ buf[32];
    যদি (unit == 0) {
        snprintf(buf, পরিমাণ(buf), "%zu B", size);
    } অন্যথা {
        snprintf(buf, পরিমাণ(buf), "%.2f %s", value, units[unit]);
    }
    ঘূৰাই std::string(buf);
}

std::string rpc_log_format_tensor(ধ্রুৱ ggml_tensor * tensor) {
    ৱর্ণ shape[64];
    snprintf(shape, পরিমাণ(shape), "[%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]", tensor->ne[0], tensor->ne[1],
             tensor->ne[2], tensor->ne[3]);

    ৱর্ণ buf[GGML_MAX_NAME + 192];
    snprintf(buf, পরিমাণ(buf), "%s [%s, %s, %s]", ggml_get_name(tensor), ggml_op_name(tensor->op),
             ggml_type_name(tensor->type), shape);
    ঘূৰাই std::string(buf);
}
