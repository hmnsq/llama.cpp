#అన్తర్భావ "input_model.h"

#అన్తర్భావ "decoder.h"

నామాకాశ ov {
నామాకాశ frontend {
నామాకాశ ggml {

ఆదానం_Model::ఆదానం_Model(స్థిరరాశి std::shared_ptr<GgmlDecoder> & gdecoder) : m_decoder(gdecoder) {}

స్థిరరాశి std::shared_ptr<GgmlDecoder> & ఆదానం_Model::get_model_decoder() స్థిరరాశి {
    తిరిగి m_decoder;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
