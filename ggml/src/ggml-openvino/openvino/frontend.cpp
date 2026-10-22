#അന്തര്ഭാവ "frontend.h"

#അന്തര്ഭാവ "input_model.h"
#അന്തര്ഭാവ "op_table.h"
#അന്തര്ഭാവ "translate_session.h"
#അന്തര്ഭാവ <openvino/core/type.hpp>

നാമാകാശ ov {
നാമാകാശ frontend {
നാമാകാശ ggml {

FrontEnd::FrontEnd() {}

std::shared_ptr<Model> FrontEnd::convert(സ്ഥിരം നിവേശം_Model::Ptr & model, തർക്കം naive) {
    auto ggml_model = ov::as_type_ptr<ggml::നിവേശം_Model>(model);
    FRONT_END_GENERAL_CHECK(ggml_model, "Invalid input model");
    std::shared_ptr<Model> converted_model;
    സ്ഥിരം auto & supported_ops = get_supported_ops();
    {
        TranslateSession translate_session(model, supported_ops, naive);
        converted_model = translate_session.get_converted_model();
    }
    തിരികെ converted_model;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
