#هدايت once

#ضمن "openvino/op/op.hpp"

نطاق ov::op::internal {
/// \note GatedDeltaNet op class is under development and subject to change
///
/// \brief Operator performing Gated Delta Net computation
/// \ingroup ov_ops_cpp_api
صنف OPENVINO_API GatedDeltaNet : عام ov::op::Op {
عام:
    OPENVINO_OP("GatedDeltaNet")

    GatedDeltaNet() = افتراضي;
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
    GatedDeltaNet(ثابت Output<Node>& query,
                  ثابت Output<Node>& ڪنجي,
                  ثابت Output<Node>& value,
                  ثابت Output<Node>& recurrent_state,
                  ثابت Output<Node>& gate,
                  ثابت Output<Node>& beta,
                  ثابت منطقي fuse_qk_l2norm = خطأ,
                  ثابت عائم q_l2_norm_eps = 1e-6F,
                  ثابت عائم k_l2_norm_eps = 1e-6F);

    /// \brief Constructs a GatedDeltaNet operation from input vector.
    ///
    /// \param args Input tensor vector in order: query, key, value, recurrent_state, gate, beta.
    /// \param fuse_qk_l2norm Enables fusing q/k L2-normalization into this op.
    /// \param q_l2_norm_eps Epsilon used for query L2-normalization when fusion is enabled.
    /// \param k_l2_norm_eps Epsilon used for key L2-normalization when fusion is enabled.
    GatedDeltaNet(ثابت ov::Output_ويڪٽر& دليلون_2,
                  ثابت منطقي fuse_qk_l2norm = خطأ,
                  ثابت عائم q_l2_norm_eps = 1e-6F,
                  ثابت عائم k_l2_norm_eps = 1e-6F);
    فراغ validate_and_infer_types() override;
    منطقي گهمو_خاصيتون(AttributeVisitor& visitor) override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(ثابت ov::Output_ويڪٽر& new_دليلون) ثابت override;
    منطقي get_fuse_qk_l2norm() ثابت {
        أرجع m_fuse_qk_l2norm;
    }
    عائم get_q_l2_norm_eps() ثابت {
        أرجع m_q_l2_norm_eps;
    }
    عائم get_k_l2_norm_eps() ثابت {
        أرجع m_k_l2_norm_eps;
    }

خاص:
    منطقي m_fuse_qk_l2norm = خطأ;
    عائم m_q_l2_norm_eps = 1e-6F;
    عائم m_k_l2_norm_eps = 1e-6F;
};

}  // namespace ov::op::internal
