#अन्तर्भाव "frontend.h"

#अन्तर्भाव "input_model.h"
#अन्तर्भाव "op_table.h"
#अन्तर्भाव "translate_session.h"
#अन्तर्भाव <openvino/core/type.hpp>

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {

FrontEnd::FrontEnd() {}

std::shared_ptr<Model> FrontEnd::convert(ध्रुव आदान_Model::Ptr & model, तार्किक naive) {
    auto ggml_model = ov::as_type_ptr<ggml::आदान_Model>(model);
    FRONT_END_GENERAL_CHECK(ggml_model, "Invalid input model");
    std::shared_ptr<Model> converted_model;
    ध्रुव auto & supported_ops = get_supported_ops();
    {
        TranslateSession translate_session(model, supported_ops, naive);
        converted_model = translate_session.get_converted_model();
    }
    प्रत्यागम converted_model;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
