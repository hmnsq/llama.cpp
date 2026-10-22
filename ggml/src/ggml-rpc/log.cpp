#समावेश "log.h"

#समावेश <cinttypes>
#समावेश <cstdio>

std::string rpc_log_format_आकार(माप_प्रकार size) {
    अटल स्थिर अक्षर * units[] = { "B", "KiB", "MiB", "GiB", "TiB", "PiB" };

    द्विगुन value = (द्विगुन) size;
    माप_प्रकार unit  = 0;
    जबतक (value >= 1024.0 && unit + 1 < माप(units) / माप(units[0])) {
        value /= 1024.0;
        unit++;
    }

    अक्षर buf[32];
    अगर (unit == 0) {
        snprintf(buf, माप(buf), "%zu B", size);
    } अन्यथा {
        snprintf(buf, माप(buf), "%.2f %s", value, units[unit]);
    }
    वापस std::string(buf);
}

std::string rpc_log_format_tensor(स्थिर ggml_tensor * tensor) {
    अक्षर shape[64];
    snprintf(shape, माप(shape), "[%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]", tensor->ne[0], tensor->ne[1],
             tensor->ne[2], tensor->ne[3]);

    अक्षर buf[GGML_MAX_NAME + 192];
    snprintf(buf, माप(buf), "%s [%s, %s, %s]", ggml_get_name(tensor), ggml_op_name(tensor->op),
             ggml_type_name(tensor->type), shape);
    वापस std::string(buf);
}
