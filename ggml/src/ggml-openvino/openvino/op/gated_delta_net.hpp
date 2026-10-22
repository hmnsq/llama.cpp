#సూచన once

#అన్తర్భావ "openvino/op/op.hpp"

నామాకాశ ov::op::internal {
/// \note GatedDeltaNet op class is under development and subject to change
///
/// \brief Operator performing Gated Delta Net computation
/// \ingroup ov_ops_cpp_api
తరగతి OPENVINO_API GatedDeltaNet : బహిరంగ ov::op::Op {
బహిరంగ:
    OPENVINO_OP("GatedDeltaNet")

    GatedDeltaNet() = అప్రమేయ;
    /// \brief Constructs a GatedDeltaNet operation.
    ///
    /// \param query Query tensor input.
    /// \param key Key tensor input.
    /// \param value Value tensor input.
    /// \param recurrent_state Initial recurrent state tensor.
    /// \param gate Gate tensor controlling state decay/update.
    /// \param beta Beta tensor scaling the delta update.
    /// \param fuse_qk_l2norm Enables fusing q/k L2-normalization into this op.
    /// \param q_l2_norm_eps Epsilon used for query L2-normalization when fusion is enabled.
    /// \param k_l2_norm_eps Epsilon used for key L2-normalization when fusion is enabled.
    GatedDeltaNet(స్థిరరాశి Output<Node>& query,
                  స్థిరరాశి Output<Node>& కీలకం,
                  స్థిరరాశి Output<Node>& value,
                  స్థిరరాశి Output<Node>& recurrent_state,
                  స్థిరరాశి Output<Node>& gate,
                  స్థిరరాశి Output<Node>& beta,
                  స్థిరరాశి తార్కిక fuse_qk_l2norm = అబద్ధం,
                  స్థిరరాశి దశాంశం q_l2_norm_eps = 1e-6F,
                  స్థిరరాశి దశాంశం k_l2_norm_eps = 1e-6F);

    /// \brief Constructs a GatedDeltaNet operation from input vector.
    ///
    /// \param args Input tensor vector in order: query, key, value, recurrent_state, gate, beta.
    /// \param fuse_qk_l2norm Enables fusing q/k L2-normalization into this op.
    /// \param q_l2_norm_eps Epsilon used for query L2-normalization when fusion is enabled.
    /// \param k_l2_norm_eps Epsilon used for key L2-normalization when fusion is enabled.
    GatedDeltaNet(స్థిరరాశి ov::Output_సదిశ& వాదనలు_2,
                  స్థిరరాశి తార్కిక fuse_qk_l2norm = అబద్ధం,
                  స్థిరరాశి దశాంశం q_l2_norm_eps = 1e-6F,
                  స్థిరరాశి దశాంశం k_l2_norm_eps = 1e-6F);
    శూన్య validate_and_infer_types() override;
    తార్కిక సందర్శించు_లక్షణాలు(AttributeVisitor& visitor) override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(స్థిరరాశి ov::Output_సదిశ& new_వాదనలు) స్థిరరాశి override;
    తార్కిక get_fuse_qk_l2norm() స్థిరరాశి {
        తిరిగి m_fuse_qk_l2norm;
    }
    దశాంశం get_q_l2_norm_eps() స్థిరరాశి {
        తిరిగి m_q_l2_norm_eps;
    }
    దశాంశం get_k_l2_norm_eps() స్థిరరాశి {
        తిరిగి m_k_l2_norm_eps;
    }

అంతరంగ:
    తార్కిక m_fuse_qk_l2norm = అబద్ధం;
    దశాంశం m_q_l2_norm_eps = 1e-6F;
    దశాంశం m_k_l2_norm_eps = 1e-6F;
};

}  // namespace ov::op::internal
