#अन्तर्भाव "log.h"

#अन्तर्भाव <cinttypes>
#अन्तर्भाव <cstdio>

std::string rpc_log_format_size(माप_प्रकार size) {
    स्थिर ध्रुव वर्ण * units[] = { "B", "KiB", "MiB", "GiB", "TiB", "PiB" };

    द्विगुण value = (द्विगुण) size;
    माप_प्रकार unit  = 0;
    यावत् (value >= 1024.0 && unit + 1 < परिमाण(units) / परिमाण(units[0])) {
        value /= 1024.0;
        unit++;
    }

    वर्ण buf[32];
    यदि (unit == 0) {
        snprintf(buf, परिमाण(buf), "%zu B", size);
    } अन्यथा {
        snprintf(buf, परिमाण(buf), "%.2f %s", value, units[unit]);
    }
    प्रत्यागम std::string(buf);
}

std::string rpc_log_format_tensor(ध्रुव ggml_tensor * tensor) {
    वर्ण shape[64];
    snprintf(shape, परिमाण(shape), "[%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]", tensor->ne[0], tensor->ne[1],
             tensor->ne[2], tensor->ne[3]);

    वर्ण buf[GGML_MAX_NAME + 192];
    snprintf(buf, परिमाण(buf), "%s [%s, %s, %s]", ggml_get_name(tensor), ggml_op_name(tensor->op),
             ggml_type_name(tensor->type), shape);
    प्रत्यागम std::string(buf);
}
