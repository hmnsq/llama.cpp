#आशय once

#समावेश "openvino/op/op.hpp"

namespace ov::op::internal {
/// \note GatedDeltaNet op class is under development and subject to change
///
/// \brief Operator performing Gated Delta Net computation
/// \ingroup ov_ops_cpp_api
श्रेणी OPENVINO_API GatedDeltaNet : खुला ov::op::Op {
खुला:
    OPENVINO_OP("GatedDeltaNet")

    GatedDeltaNet() = शेष;
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
    GatedDeltaNet(स्थिर Output<Node>& query,
                  स्थिर Output<Node>& कुंजी,
                  स्थिर Output<Node>& value,
                  स्थिर Output<Node>& recurrent_state,
                  स्थिर Output<Node>& gate,
                  स्थिर Output<Node>& beta,
                  स्थिर bool fuse_qk_l2norm = false,
                  स्थिर भग्न q_l2_norm_eps = 1e-6F,
                  स्थिर भग्न k_l2_norm_eps = 1e-6F);

    /// \brief Constructs a GatedDeltaNet operation from input vector.
    ///
    /// \param args Input tensor vector in order: query, key, value, recurrent_state, gate, beta.
    /// \param fuse_qk_l2norm Enables fusing q/k L2-normalization into this op.
    /// \param q_l2_norm_eps Epsilon used for query L2-normalization when fusion is enabled.
    /// \param k_l2_norm_eps Epsilon used for key L2-normalization when fusion is enabled.
    GatedDeltaNet(स्थिर ov::OutputVector& args,
                  स्थिर bool fuse_qk_l2norm = false,
                  स्थिर भग्न q_l2_norm_eps = 1e-6F,
                  स्थिर भग्न k_l2_norm_eps = 1e-6F);
    व्योम validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(स्थिर ov::OutputVector& new_args) स्थिर override;
    bool get_fuse_qk_l2norm() स्थिर {
        वापस m_fuse_qk_l2norm;
    }
    भग्न get_q_l2_norm_eps() स्थिर {
        वापस m_q_l2_norm_eps;
    }
    भग्न get_k_l2_norm_eps() स्थिर {
        वापस m_k_l2_norm_eps;
    }

निजी:
    bool m_fuse_qk_l2norm = false;
    भग्न m_q_l2_norm_eps = 1e-6F;
    भग्न m_k_l2_norm_eps = 1e-6F;
};

}  // namespace ov::op::internal
