#ضمن "input_model.h"

#ضمن "decoder.h"

نطاق ov {
نطاق frontend {
نطاق ggml {

شغرو_Model::شغرو_Model(ثابت std::shared_ptr<GgmlDecoder> & gdecoder) : m_decoder(gdecoder) {}

ثابت std::shared_ptr<GgmlDecoder> & شغرو_Model::get_model_decoder() ثابت {
    أرجع m_decoder;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
