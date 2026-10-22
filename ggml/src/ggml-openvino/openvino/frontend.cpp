#અન્તર્ભાવ "frontend.h"

#અન્તર્ભાવ "input_model.h"
#અન્તર્ભાવ "op_table.h"
#અન્તર્ભાવ "translate_session.h"
#અન્તર્ભાવ <openvino/core/type.hpp>

નામાકાશ ov {
નામાકાશ frontend {
નામાકાશ ggml {

FrontEnd::FrontEnd() {}

std::shared_ptr<Model> FrontEnd::convert(અચલ આગત_Model::Ptr & model, તાર્કિક naive) {
    auto ggml_model = ov::as_type_ptr<ggml::આગત_Model>(model);
    FRONT_END_GENERAL_CHECK(ggml_model, "Invalid input model");
    std::shared_ptr<Model> converted_model;
    અચલ auto & supported_ops = get_supported_ops();
    {
        TranslateSession translate_session(model, supported_ops, naive);
        converted_model = translate_session.get_converted_model();
    }
    પાછા converted_model;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
