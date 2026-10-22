#شامل "input_model.h"

#شامل "decoder.h"

فضاینام ov {
فضاینام frontend {
فضاینام ggml {

درون_داد_Model::درون_داد_Model(ثابت std::shared_ptr<GgmlDecoder> & gdecoder) : m_decoder(gdecoder) {}

ثابت std::shared_ptr<GgmlDecoder> & درون_داد_Model::get_model_decoder() ثابت {
    بازگشت m_decoder;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
