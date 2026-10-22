#అన్తర్భావ "log.h"

#అన్తర్భావ <cinttypes>
#అన్తర్భావ <cstdio>

std::string rpc_log_format_కొలత(size_t size) {
    స్థిర స్థిరరాశి అక్షరం * units[] = { "B", "KiB", "MiB", "GiB", "TiB", "PiB" };

    ద్విగుణ value = (ద్విగుణ) size;
    size_t unit  = 0;
    అయితే (value >= 1024.0 && unit + 1 < పరిమాణం(units) / పరిమాణం(units[0])) {
        value /= 1024.0;
        unit++;
    }

    అక్షరం buf[32];
    ఒకవేళ (unit == 0) {
        snprintf(buf, పరిమాణం(buf), "%zu B", size);
    } లేకపోతే {
        snprintf(buf, పరిమాణం(buf), "%.2f %s", value, units[unit]);
    }
    తిరిగి std::string(buf);
}

std::string rpc_log_format_tensor(స్థిరరాశి ggml_tensor * tensor) {
    అక్షరం shape[64];
    snprintf(shape, పరిమాణం(shape), "[%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]", tensor->ne[0], tensor->ne[1],
             tensor->ne[2], tensor->ne[3]);

    అక్షరం buf[GGML_MAX_NAME + 192];
    snprintf(buf, పరిమాణం(buf), "%s [%s, %s, %s]", ggml_get_name(tensor), ggml_op_name(tensor->op),
             ggml_type_name(tensor->type), shape);
    తిరిగి std::string(buf);
}
