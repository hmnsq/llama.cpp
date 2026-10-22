#pragma once

#அந்தர்பாவ "openvino/op/op.hpp"

நாமாகாஶ ov::op::internal {
/// \note GatedDeltaNet op class is under development and subject to change
///
/// \brief Operator performing Gated Delta Net computation
/// \ingroup ov_ops_cpp_api
வகுப்பு OPENVINO_API GatedDeltaNet : பொது ov::op::Op {
பொது:
    OPENVINO_OP("GatedDeltaNet")

    GatedDeltaNet() = இயல்பு;
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
    GatedDeltaNet(மாறிலி Output<Node>& query,
                  மாறிலி Output<Node>& சாவி,
                  மாறிலி Output<Node>& value,
                  மாறிலி Output<Node>& recurrent_state,
                  மாறிலி Output<Node>& gate,
                  மாறிலி Output<Node>& beta,
                  மாறிலி தருக்க fuse_qk_l2norm = பொய்,
                  மாறிலி தசம q_l2_norm_eps = 1e-6F,
                  மாறிலி தசம k_l2_norm_eps = 1e-6F);

    /// \brief Constructs a GatedDeltaNet operation from input vector.
    ///
    /// \param args Input tensor vector in order: query, key, value, recurrent_state, gate, beta.
    /// \param fuse_qk_l2norm Enables fusing q/k L2-normalization into this op.
    /// \param q_l2_norm_eps Epsilon used for query L2-normalization when fusion is enabled.
    /// \param k_l2_norm_eps Epsilon used for key L2-normalization when fusion is enabled.
    GatedDeltaNet(மாறிலி ov::OutputVector& args,
                  மாறிலி தருக்க fuse_qk_l2norm = பொய்,
                  மாறிலி தசம q_l2_norm_eps = 1e-6F,
                  மாறிலி தசம k_l2_norm_eps = 1e-6F);
    வெறுமை validate_and_infer_types() override;
    தருக்க visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(மாறிலி ov::OutputVector& new_args) மாறிலி override;
    தருக்க get_fuse_qk_l2norm() மாறிலி {
        திரும்பு m_fuse_qk_l2norm;
    }
    தசம get_q_l2_norm_eps() மாறிலி {
        திரும்பு m_q_l2_norm_eps;
    }
    தசம get_k_l2_norm_eps() மாறிலி {
        திரும்பு m_k_l2_norm_eps;
    }

தனிப்பட்ட:
    தருக்க m_fuse_qk_l2norm = பொய்;
    தசம m_q_l2_norm_eps = 1e-6F;
    தசம m_k_l2_norm_eps = 1e-6F;
};

}  // namespace ov::op::internal
