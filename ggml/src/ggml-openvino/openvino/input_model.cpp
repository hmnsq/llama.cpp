#ضمن "input_model.h"

#ضمن "decoder.h"

نطاق ov {
نطاق frontend {
نطاق ggml {

إدخال_Model::إدخال_Model(ثابت std::shared_ptr<GgmlDecoder> & gdecoder) : m_decoder(gdecoder) {}

ثابت std::shared_ptr<GgmlDecoder> & إدخال_Model::get_model_decoder() ثابت {
    إرجاع m_decoder;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
