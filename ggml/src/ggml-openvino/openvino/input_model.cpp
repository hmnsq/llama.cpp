#ضمن "input_model.h"

#ضمن "decoder.h"

نطاق ov {
نطاق frontend {
نطاق ggml {

InputModel::InputModel(ثابت std::shared_ptr<GgmlDecoder> & gdecoder) : m_decoder(gdecoder) {}

ثابت std::shared_ptr<GgmlDecoder> & InputModel::get_model_decoder() ثابت {
    החזר m_decoder;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
