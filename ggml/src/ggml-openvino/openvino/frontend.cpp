#ಅನ್ತರ್ಭಾವ "frontend.h"

#ಅನ್ತರ್ಭಾವ "input_model.h"
#ಅನ್ತರ್ಭಾವ "op_table.h"
#ಅನ್ತರ್ಭಾವ "translate_session.h"
#ಅನ್ತರ್ಭಾವ <openvino/core/type.hpp>

ನಾಮಾಕಾಶ ov {
ನಾಮಾಕಾಶ frontend {
ನಾಮಾಕಾಶ ggml {

FrontEnd::FrontEnd() {}

std::shared_ptr<Model> FrontEnd::convert(ಧ್ರುವ ಒಳಹರಿವು_Model::Ptr & model, ತಾರ್ಕಿಕ naive) {
    auto ggml_model = ov::as_type_ptr<ggml::ಒಳಹರಿವು_Model>(model);
    FRONT_END_GENERAL_CHECK(ggml_model, "Invalid input model");
    std::shared_ptr<Model> converted_model;
    ಧ್ರುವ auto & supported_ops = get_supported_ops();
    {
        TranslateSession translate_session(model, supported_ops, naive);
        converted_model = translate_session.get_converted_model();
    }
    ಪ್ರತ್ಯಾಗಮ converted_model;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
