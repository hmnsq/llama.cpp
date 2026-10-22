#pragma once

#অন্তর্ভাব "openvino/op/op.hpp"

নামাকাশ ov::op::internal {
/// \note GatedDeltaNet op class is under development and subject to change
///
/// \brief Operator performing Gated Delta Net computation
/// \ingroup ov_ops_cpp_api
শ্রেণী OPENVINO_API GatedDeltaNet : সর্বজনীন ov::op::Op {
সর্বজনীন:
    OPENVINO_OP("GatedDeltaNet")

    GatedDeltaNet() = পূর্বনির্ধারিত;
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
    GatedDeltaNet(ধ্রুবক Output<Node>& query,
                  ধ্রুবক Output<Node>& চাবি,
                  ধ্রুবক Output<Node>& value,
                  ধ্রুবক Output<Node>& recurrent_state,
                  ধ্রুবক Output<Node>& gate,
                  ধ্রুবক Output<Node>& beta,
                  ধ্রুবক যৌক্তিক fuse_qk_l2norm = মিথ্যা,
                  ধ্রুবক দশমিক q_l2_norm_eps = 1e-6F,
                  ধ্রুবক দশমিক k_l2_norm_eps = 1e-6F);

    /// \brief Constructs a GatedDeltaNet operation from input vector.
    ///
    /// \param args Input tensor vector in order: query, key, value, recurrent_state, gate, beta.
    /// \param fuse_qk_l2norm Enables fusing q/k L2-normalization into this op.
    /// \param q_l2_norm_eps Epsilon used for query L2-normalization when fusion is enabled.
    /// \param k_l2_norm_eps Epsilon used for key L2-normalization when fusion is enabled.
    GatedDeltaNet(ধ্রুবক ov::OutputVector& args,
                  ধ্রুবক যৌক্তিক fuse_qk_l2norm = মিথ্যা,
                  ধ্রুবক দশমিক q_l2_norm_eps = 1e-6F,
                  ধ্রুবক দশমিক k_l2_norm_eps = 1e-6F);
    শূন্য validate_and_infer_types() override;
    যৌক্তিক visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(ধ্রুবক ov::OutputVector& new_args) ধ্রুবক override;
    যৌক্তিক get_fuse_qk_l2norm() ধ্রুবক {
        ফেরত m_fuse_qk_l2norm;
    }
    দশমিক get_q_l2_norm_eps() ধ্রুবক {
        ফেরত m_q_l2_norm_eps;
    }
    দশমিক get_k_l2_norm_eps() ধ্রুবক {
        ফেরত m_k_l2_norm_eps;
    }

ব্যক্তিগত:
    যৌক্তিক m_fuse_qk_l2norm = মিথ্যা;
    দশমিক m_q_l2_norm_eps = 1e-6F;
    দশমিক m_k_l2_norm_eps = 1e-6F;
};

}  // namespace ov::op::internal
