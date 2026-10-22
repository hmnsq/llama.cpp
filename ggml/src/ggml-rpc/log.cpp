#अन्तर्भाव "log.h"

#अन्तर्भाव <cinttypes>
#अन्तर्भाव <cstdio>

std::string rpc_log_format_size(माप_प्रकार size) {
    स्थिर स्थिरांक अक्षर * units[] = { "B", "KiB", "MiB", "GiB", "TiB", "PiB" };

    द्विगुण value = (द्विगुण) size;
    माप_प्रकार unit  = 0;
    पर्यंत (value >= 1024.0 && unit + 1 < आकार(units) / आकार(units[0])) {
        value /= 1024.0;
        unit++;
    }

    अक्षर buf[32];
    जर (unit == 0) {
        snprintf(buf, आकार(buf), "%zu B", size);
    } अन्यथा {
        snprintf(buf, आकार(buf), "%.2f %s", value, units[unit]);
    }
    परत std::string(buf);
}

std::string rpc_log_format_tensor(स्थिरांक ggml_tensor * tensor) {
    अक्षर shape[64];
    snprintf(shape, आकार(shape), "[%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]", tensor->ne[0], tensor->ne[1],
             tensor->ne[2], tensor->ne[3]);

    अक्षर buf[GGML_MAX_NAME + 192];
    snprintf(buf, आकार(buf), "%s [%s, %s, %s]", ggml_get_name(tensor), ggml_op_name(tensor->op),
             ggml_type_name(tensor->type), shape);
    परत std::string(buf);
}
