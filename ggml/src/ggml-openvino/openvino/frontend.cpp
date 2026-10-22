#ଅନ୍ତର୍ଭାଵ "frontend.h"

#ଅନ୍ତର୍ଭାଵ "input_model.h"
#ଅନ୍ତର୍ଭାଵ "op_table.h"
#ଅନ୍ତର୍ଭାଵ "translate_session.h"
#ଅନ୍ତର୍ଭାଵ <openvino/core/type.hpp>

ନାମାକାଶ ov {
ନାମାକାଶ frontend {
ନାମାକାଶ ggml {

FrontEnd::FrontEnd() {}

std::shared_ptr<Model> FrontEnd::convert(ଅଚଳ ନିବେଶ_Model::Ptr & model, ତାର୍କିକ naive) {
    auto ggml_model = ov::as_type_ptr<ggml::ନିବେଶ_Model>(model);
    FRONT_END_GENERAL_CHECK(ggml_model, "Invalid input model");
    std::shared_ptr<Model> converted_model;
    ଅଚଳ auto & supported_ops = get_supported_ops();
    {
        TranslateSession translate_session(model, supported_ops, naive);
        converted_model = translate_session.get_converted_model();
    }
    ଫେରନ୍ତୁ converted_model;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
