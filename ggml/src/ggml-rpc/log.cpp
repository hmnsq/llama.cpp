#ଅନ୍ତର୍ଭାଵ "log.h"

#ଅନ୍ତର୍ଭାଵ <cinttypes>
#ଅନ୍ତର୍ଭାଵ <cstdio>

std::string rpc_log_format_size(size_t size) {
    ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * units[] = { "B", "KiB", "MiB", "GiB", "TiB", "PiB" };

    ଦ୍ୱିଗୁଣ value = (ଦ୍ୱିଗୁଣ) size;
    size_t unit  = 0;
    ଯେତେବେଳେ (value >= 1024.0 && unit + 1 < ଆକାର(units) / ଆକାର(units[0])) {
        value /= 1024.0;
        unit++;
    }

    ଅକ୍ଷର buf[32];
    ଯଦି (unit == 0) {
        snprintf(buf, ଆକାର(buf), "%zu B", size);
    } ଅନ୍ୟଥା {
        snprintf(buf, ଆକାର(buf), "%.2f %s", value, units[unit]);
    }
    ଫେରନ୍ତୁ std::string(buf);
}

std::string rpc_log_format_tensor(ଅଚଳ ggml_tensor * tensor) {
    ଅକ୍ଷର shape[64];
    snprintf(shape, ଆକାର(shape), "[%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]", tensor->ne[0], tensor->ne[1],
             tensor->ne[2], tensor->ne[3]);

    ଅକ୍ଷର buf[GGML_MAX_NAME + 192];
    snprintf(buf, ଆକାର(buf), "%s [%s, %s, %s]", ggml_get_name(tensor), ggml_op_name(tensor->op),
             ggml_type_name(tensor->type), shape);
    ଫେରନ୍ତୁ std::string(buf);
}
