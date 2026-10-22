#ضمن "input_model.h"

#ضمن "decoder.h"

نطاق ov {
نطاق frontend {
نطاق ggml {

داخلہ_Model::داخلہ_Model(ثابت std::shared_ptr<GgmlDecoder> & gdecoder) : m_decoder(gdecoder) {}

ثابت std::shared_ptr<GgmlDecoder> & داخلہ_Model::get_model_decoder() ثابت {
    أرجع m_decoder;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
