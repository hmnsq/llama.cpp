#समावेश "ggml-vulkan-common.h"

अटल vk_device_architecture get_device_architecture(स्थिर vk::PhysicalDevice& device) {
    vk::PhysicalDeviceProperties props = device.getProperties();

    अगर (props.vendorID == VK_VENDOR_ID_AMD) {
        स्थिर std::vector<vk::ExtensionProperties> ext_props = device.enumerateDeviceExtensionProperties();

        bool amd_shader_core_properties = false;
        bool integer_dot_product = false;
        bool subgroup_आकार_control = false;
        bool shader_float8 = false;

        क्रम (स्थिर स्वतः& properties : ext_props) {
            अगर (म_भेद("VK_AMD_shader_core_properties", properties.extensionName) == 0) {
                amd_shader_core_properties = true;
            } अन्यथा अगर (म_भेद("VK_KHR_shader_integer_dot_product", properties.extensionName) == 0) {
                integer_dot_product = true;
            } अन्यथा अगर (म_भेद("VK_EXT_subgroup_size_control", properties.extensionName) == 0) {
                subgroup_आकार_control = true;
            } अन्यथा अगर (म_भेद("VK_EXT_shader_float8", properties.extensionName) == 0) {
                shader_float8 = true;
            }
        }

        अगर (!amd_shader_core_properties || !integer_dot_product || !subgroup_आकार_control) {
            वापस vk_device_architecture::OTHER;
        }

        vk::PhysicalDeviceProperties2 props2;
        vk::PhysicalDeviceShaderCorePropertiesAMD shader_core_props_amd;
        vk::PhysicalDeviceShaderIntegerDotProductPropertiesKHR integer_dot_props;
        vk::Physical_Device_Subgroup_आकार_Control_Properties_EXT subgroup_आकार_control_props;

        props2.p_अग्र = &shader_core_props_amd;
        shader_core_props_amd.p_अग्र = &integer_dot_props;
        integer_dot_props.p_अग्र = &subgroup_आकार_control_props;

        device.getProperties2(&props2);

        अगर (subgroup_आकार_control_props.max_Subgroup_आकार == 64 && subgroup_आकार_control_props.min_Subgroup_आकार == 64) {
            वापस vk_device_architecture::AMD_GCN;
        }
        अगर (subgroup_आकार_control_props.max_Subgroup_आकार == 64 && subgroup_आकार_control_props.min_Subgroup_आकार == 32) {
            // RDNA
            अगर (shader_core_props_amd.wavefrontsPerSimd == 20) {
                वापस vk_device_architecture::AMD_RDNA1;
            }
            अगर (shader_float8) {
                वापस vk_device_architecture::AMD_RDNA4;
            }
            अगर (integer_dot_props.integerDotProduct4x8BitPackedMixedSignednessAccelerated) {
                वापस vk_device_architecture::AMD_RDNA3;
            }
            वापस vk_device_architecture::AMD_RDNA2;
        }
    } अन्यथा अगर (props.vendorID == VK_VENDOR_ID_INTEL) {
        स्थिर std::vector<vk::ExtensionProperties> ext_props = device.enumerateDeviceExtensionProperties();

        bool subgroup_आकार_control = false;
        bool integer_dot_product = false;

        क्रम (स्थिर स्वतः& properties : ext_props) {
            अगर (म_भेद("VK_EXT_subgroup_size_control", properties.extensionName) == 0) {
                subgroup_आकार_control = true;
            } अन्यथा अगर (म_भेद("VK_KHR_shader_integer_dot_product", properties.extensionName) == 0) {
                integer_dot_product = true;
            }
        }

        अगर (!subgroup_आकार_control || !integer_dot_product) {
            वापस vk_device_architecture::OTHER;
        }

        vk::PhysicalDeviceProperties2 props2;
        vk::Physical_Device_Subgroup_आकार_Control_Properties_EXT subgroup_आकार_control_props;
        vk::PhysicalDeviceShaderIntegerDotProductPropertiesKHR integer_dot_props;

        props2.p_अग्र = &subgroup_आकार_control_props;
        subgroup_आकार_control_props.p_अग्र = &integer_dot_props;
        device.getProperties2(&props2);

        अगर (subgroup_आकार_control_props.min_Subgroup_आकार == 16) {
            // Xe2 architecture uses SIMD16 while previous Xe and Gen architecture uses SIMD8.
            // Minimum subgroup size matches the SIMD width so we distinguish architecture by checking this value.
            // https://www.intel.com/content/www/us/en/content-details/824434/2024-intel-tech-tour-xe2-and-lunar-lake-s-gpu.html
            // https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-0/intel-xe-gpu-architecture.html
            वापस vk_device_architecture::INTEL_XE2;
        } अन्यथा अगर (subgroup_आकार_control_props.min_Subgroup_आकार == 8 &&
                 integer_dot_product && integer_dot_props.integerDotProduct4x8BitPackedSignedAccelerated) {
            वापस vk_device_architecture::INTEL_XE1;
        }
    } अन्यथा अगर (props.vendorID == VK_VENDOR_ID_NVIDIA) {
        स्थिर std::vector<vk::ExtensionProperties> ext_props = device.enumerateDeviceExtensionProperties();

        bool cooperative_matrix = false;
        bool sm_builtins = false;

        // Detect "pre-turing" based on lack of coopmat support.
        क्रम (स्थिर स्वतः& properties : ext_props) {
            अगर (म_भेद("VK_KHR_cooperative_matrix", properties.extensionName) == 0) {
                cooperative_matrix = true;
            } अन्यथा अगर (म_भेद("VK_NV_shader_sm_builtins", properties.extensionName) == 0) {
                sm_builtins = true;
            }
        }

        अगर (!cooperative_matrix) {
            वापस vk_device_architecture::NVIDIA_PRE_TURING;
        }

        अगर (sm_builtins) {
            vk::PhysicalDeviceProperties2 props2;
            vk::PhysicalDeviceShaderSMBuiltinsPropertiesNV sm_props;

            props2.p_अग्र = &sm_props;

            device.getProperties2(&props2);

            // Turing has 32, following architectures have 48
            अगर (sm_props.shaderWarpsPerSM == 32) {
                वापस vk_device_architecture::NVIDIA_TURING;
            }
        }
    } अन्यथा अगर(props.vendorID == VK_VENDOR_ID_QUALCOMM){
        स्थिर std::vector<vk::ExtensionProperties> ext_props = device.enumerateDeviceExtensionProperties();

        bool cooperative_matrix = false;
        bool cooperative_matrix_conversion = false;

        क्रम (स्थिर स्वतः& properties : ext_props) {
            अगर (म_भेद("VK_KHR_cooperative_matrix", properties.extensionName) == 0) {
                cooperative_matrix = true;
            } अन्यथा अगर (म_भेद("VK_QCOM_cooperative_matrix_conversion", properties.extensionName) == 0) {
                cooperative_matrix_conversion = true;
            }
        }

        अगर (cooperative_matrix && cooperative_matrix_conversion) {
            वापस vk_device_architecture::QUALCOMM_ADRENO;
        }
    }
    वापस vk_device_architecture::OTHER;
}

bool ggml_vk_lightning_indexer_k_type_supported(ggml_type type) {
    वापस std::find(lightning_indexer_k_types.begin(), lightning_indexer_k_types.end(), type) != lightning_indexer_k_types.end();
}
व्योम ggml_vk_print_device_fault_info(स्थिर vk_device& device) {
    अगर (!device->device_fault || !device->pfn_vkGetDeviceFaultInfoEXT) {
        वापस;
    }

    VkDeviceFaultCountsEXT fault_counts {};
    fault_counts.sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT;
    Vk_परिणाम res = device->pfn_vkGetDeviceFaultInfoEXT(device->device, &fault_counts, nullptr);
    अगर (res != VK_SUCCESS) {
        GGML_LOG_ERROR("ggml_vulkan: vkGetDeviceFaultInfoEXT (counts) failed: %d\n", res);
        वापस;
    }

    std::vector<VkDeviceFaultAddressInfoEXT> address_infos(fault_counts.address_Info_गिनती);
    std::vector<VkDeviceFaultVendorInfoEXT> vendor_infos(fault_counts.vendor_Info_गिनती);

    VkDeviceFaultInfoEXT fault_info {};
    fault_info.sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT;
    fault_info.pAddressInfos = address_infos.तथ्य();
    fault_info.pVendorInfos = vendor_infos.तथ्य();

    res = device->pfn_vkGetDeviceFaultInfoEXT(device->device, &fault_counts, &fault_info);
    अगर (res != VK_SUCCESS) {
        GGML_LOG_ERROR("ggml_vulkan: vkGetDeviceFaultInfoEXT (info) failed: %d\n", res);
        वापस;
    }

    अगर (fault_counts.address_Info_गिनती == 0 && fault_counts.vendor_Info_गिनती == 0 && fault_info.description[0] == '\0') {
        वापस;
    }

    अगर (fault_info.description[0] != '\0') {
        GGML_LOG_ERROR("ggml_vulkan: device fault on %s: %s\n", device->name.c_str(), fault_info.description);
    }

    क्रम (uint32_t i = 0; i < fault_counts.address_Info_गिनती; i++) {
        स्थिर स्वतः& info = address_infos[i];
        GGML_LOG_CONT("  address fault %u: type=%d address=0x%llx precision=0x%llx\n",
                i, (पूर्णांक)info.addressType,
                (अचिन्हित दीर्घ दीर्घ)info.reportedAddress,
                (अचिन्हित दीर्घ दीर्घ)info.addressPrecision);
    }
    क्रम (uint32_t i = 0; i < fault_counts.vendor_Info_गिनती; i++) {
        स्थिर स्वतः& info = vendor_infos[i];
        GGML_LOG_CONT("  vendor fault %u: %s (code=0x%llx data=0x%llx)\n",
                i, info.description,
                (अचिन्हित दीर्घ दीर्घ)info.vendorFaultCode,
                (अचिन्हित दीर्घ दीर्घ)info.vendor_Fault_तथ्य);
    }
}
uint64_t ggml_vk_get_गाँठ_flops(स्थिर ggml_tensor * गाँठ) {
    अगर (गाँठ->op == GGML_OP_MUL_MAT || गाँठ->op == GGML_OP_MUL_MAT_ID) {
        स्थिर uint64_t m     = गाँठ->ne[0];
        स्थिर uint64_t n     = गाँठ->ne[1];
        स्थिर uint64_t k     = गाँठ->src[1]->ne[0];
        स्थिर uint64_t batch = गाँठ->ne[2] * गाँठ->ne[3];
        वापस m * n * (k + (k - 1)) * batch;
    }
    अगर (गाँठ->op == GGML_OP_CONV_2D || गाँठ->op == GGML_OP_CONV_TRANSPOSE_2D) {
        स्थिर ggml_tensor * knl = गाँठ->src[0];
        स्थिर uint64_t Cout  = गाँठ->ne[2];
        स्थिर uint64_t आकार_K = गाँठ->src[1]->ne[2] * knl->ne[0] * knl->ne[1];
        स्थिर uint64_t आकार_N_2 = गाँठ->ne[3] * गाँठ->ne[0] * गाँठ->ne[1];
        वापस Cout * आकार_N_2 * (आकार_K + (आकार_K - 1));
    }
    अगर (गाँठ->op == GGML_OP_CONV_3D) {
        स्थिर ggml_tensor * knl = गाँठ->src[0];
        स्थिर uint64_t OC     = ggml_get_op_params_i32(गाँठ, 11);
        स्थिर uint64_t IC     = ggml_get_op_params_i32(गाँठ, 9);
        स्थिर uint64_t आकार_K = IC * knl->ne[0] * knl->ne[1] * knl->ne[2];
        स्थिर uint64_t आकार_N_2 = गाँठ->ne[3] / OC * गाँठ->ne[0] * गाँठ->ne[1] * गाँठ->ne[2];
        वापस OC * आकार_N_2 * (आकार_K + (आकार_K - 1));
    }
    अगर (गाँठ->op == GGML_OP_FLASH_ATTN_EXT) {
        स्थिर ggml_tensor * q = गाँठ->src[0];
        स्थिर ggml_tensor * k = गाँठ->src[1];
        स्थिर ggml_tensor * v = गाँठ->src[2];
        वापस 2ull * q->ne[1] * q->ne[2] * (k->ne[0] + v->ne[0]) * k->ne[1] * q->ne[3];
    }
    वापस 0;
}
व्योम ggml_vk_print_गाँठ_list(स्थिर ggml_cgraph * cgraph, पूर्णांक start, पूर्णांक end) {
    uint64_t योग_flops = 0;
    पूर्णांक n_ops = 0;
    क्रम (पूर्णांक j = start; j <= end && j < cgraph->n_nodes; j++) {
        uint64_t flops = ggml_vk_get_गाँठ_flops(cgraph->nodes[j]);
        योग_flops += flops;
        n_ops++;
        अगर (flops > 0) {
            GGML_LOG_CONT("  node %d: %s (%s) [%.2f GFLOP]\n",
                    j, cgraph->nodes[j]->name, ggml_op_name(cgraph->nodes[j]->op),
                    flops / 1e9);
        } अन्यथा {
            GGML_LOG_CONT("  node %d: %s (%s)\n",
                    j, cgraph->nodes[j]->name, ggml_op_name(cgraph->nodes[j]->op));
        }
    }
    GGML_LOG_CONT("  total: %d ops, %.2f GFLOP\n", n_ops, योग_flops / 1e9);
}
व्योम ggml_vk_print_device_lost_info(स्थिर vk_device& device) {
    ggml_vk_print_device_fault_info(device);
    अगर (device->serialize_submissions && device->diag_cgraph != nullptr && device->diag_पूर्व_start >= 0) {
        GGML_LOG_ERROR("ggml_vulkan: device lost on %s, likely caused by previous submission (nodes %d to %d):\n",
                device->name.c_str(), device->diag_पूर्व_start, device->diag_पूर्व_end);
        ggml_vk_print_गाँठ_list(device->diag_cgraph, device->diag_पूर्व_start, device->diag_पूर्व_end);
    } अन्यथा {
        GGML_LOG_ERROR("ggml_vulkan: device lost on %s\n", device->name.c_str());
    }
}
व्योम * स्थिर vk_ptr_base = (व्योम *)(uintptr_t) 0x1000;  // NOLINT

uint64_t vk_tensor_offset(स्थिर ggml_tensor * tensor) {
    अगर (tensor->view_src) {
        वापस (uint8_t *) tensor->view_src->तथ्य - (uint8_t *) vk_ptr_base;
    }
    वापस (uint8_t *) tensor->तथ्य - (uint8_t *) vk_ptr_base;
}

माप_प्रकार ggml_vk_tensor_बफर_offset(स्थिर ggml_backend_vk_context * ctx, स्थिर ggml_tensor * t) {
    // vk_tensor_offset() is relative to vk_ptr_base, but mapped host tensors need an offset relative to their Vulkan buffer.
    अगर (ctx->device->uma) {
        vk_बफर buf = nullptr;
        माप_प्रकार off = 0;
        ggml_vk_host_get(ctx->device, t->तथ्य, buf, off);
        अगर (buf) {
            वापस off;
        }
    }
    वापस (माप_प्रकार)(vk_tensor_offset(t) + t->view_offs);
}
माप_प्रकार ggml_vk_descriptor_offset(माप_प्रकार tensor_offset, माप_प्रकार alignment, माप_प्रकार type_आकार) {
    // Move the descriptor back until its distance to the tensor is divisible by the tensor type size.
    माप_प्रकार descriptor_offset = tensor_offset & ~(alignment - 1);
    जबतक ((tensor_offset - descriptor_offset) % type_आकार != 0) {
        GGML_ASSERT(descriptor_offset >= alignment);
        descriptor_offset -= alignment;
    }

    वापस descriptor_offset;
}
uint32_t get_misalign_bytes(स्थिर ggml_backend_vk_context * ctx, स्थिर ggml_tensor * t) {
    स्थिर माप_प्रकार tensor_offset = ggml_vk_tensor_बफर_offset(ctx, t);
    स्थिर माप_प्रकार descriptor_offset = ggml_vk_descriptor_offset(
        tensor_offset, ctx->device->properties.limits.min_Storage_बफर_Offset_Alignment, ggml_type_आकार(t->type));
    GGML_ASSERT(tensor_offset - descriptor_offset <= UINT32_MAX);
    वापस tensor_offset - descriptor_offset;
}

uint32_t ggml_vk_concat_unit_आकार(ggml_type type) {
    स्थिर uint32_t type_आकार = ggml_type_आकार(type);

    अगर (!ggml_is_quantized(type)) {
        वापस type_आकार;
    }

    // Use the widest existing concat shader that evenly divides a quant block.
    अगर (type_आकार % 8 == 0) {
        वापस 8;
    }
    अगर (type_आकार % 4 == 0) {
        वापस 4;
    }
    अगर (type_आकार % 2 == 0) {
        वापस 2;
    }
    वापस 1;
}
bool ggml_vk_concat_supported(स्थिर ggml_tensor * src0, स्थिर ggml_tensor * src1, स्थिर ggml_tensor * dst) {
    अगर (src0->type != src1->type || src0->type != dst->type) {
        वापस false;
    }

    अगर (!ggml_is_quantized(src0->type)) {
        स्थिर माप_प्रकार type_आकार = ggml_type_आकार(src0->type);
        वापस type_आकार == 1 || type_आकार == 2 || type_आकार == 4 || type_आकार == 8;
    }

    // Quantized tensor rows are block-aligned when created.
    वापस ggml_is_contiguous_rows(src0) && ggml_is_contiguous_rows(src1) && ggml_is_contiguous_rows(dst);
}
अटल bool vk_instance_initialized = false;

vk_instance_t vk_instance;

अटल Vk_Device_आकार ggml_vk_get_max_बफर_range(स्थिर ggml_backend_vk_context * ctx, स्थिर vk_बफर &buf, स्थिर Vk_Device_आकार offset) {
    स्थिर Vk_Device_आकार range = std::min(Vk_Device_आकार{buf->size - offset},
                                        Vk_Device_आकार{ctx->device->properties.limits.max_Storage_बफर_Range});
    वापस range;
}

व्योम ggml_vk_wait_for_fence(ggml_backend_vk_context * ctx) {
    // Use waitForFences while most of the graph executes. Hopefully the CPU can sleep
    // during this wait.
    अगर (ctx->almost_ready_fence_pending) {
        VK_CHECK(ctx->device->device.waitForFences({ ctx->almost_ready_fence }, true, UINT64_MAX), "almost_ready_fence", ctx->device);
        ctx->device->device.resetFences({ ctx->almost_ready_fence });
        ctx->almost_ready_fence_pending = false;
    }

    // Spin (w/pause) waiting for the graph to finish executing.
    vk::परिणाम_2 परिणाम;
    क्रम (;;) {
        प्रयास {
            परिणाम = ctx->device->device.getFenceStatus(ctx->fence);
        } पकड़ो (vk::DeviceLostError &) {
            ggml_vk_print_device_lost_info(ctx->device);
            GGML_LOG_ERROR("ggml_vulkan: getFenceStatus at %s:%d\n", __खाता__, __LINE__);
            फेंको;
        }
        अगर (परिणाम == vk::परिणाम_2::eSuccess) {
            अवरोध;
        }
        अगर (परिणाम != vk::परिणाम_2::eNotReady) {
            GGML_LOG_ERROR("ggml_vulkan: error %s at %s:%d\n", to_string(परिणाम).c_str(), __खाता__, __LINE__);
            फेंको vk::SystemError(vk::make_error_code(परिणाम), "ggml_vulkan: getFenceStatus");
        }
        क्रम (uint32_t i = 0; i < 100; ++i) {
            YIELD();
            YIELD();
            YIELD();
            YIELD();
            YIELD();
            YIELD();
            YIELD();
            YIELD();
            YIELD();
            YIELD();
        }
    }
    ctx->device->device.resetFences({ ctx->fence });
}

अटल bool ggml_vk_strip_decode_vector(स्थिर uint32_t * code, माप_प्रकार word_गिनती, std::vector<uint32_t> & out) {
    अटल स्थिर अक्षर kDecodeVectorExt[] = "SPV_NV_cooperative_matrix_decode_vector";

    अगर (word_गिनती < 5) {
        वापस false;
    }

    bool uses_decode_vector = false;
    क्रम (माप_प्रकार pos = 5; pos < word_गिनती; ) {
        uint32_t word = code[pos];
        uint32_t wc   = word >> spv::Word_गिनती_Shift;
        uint32_t op   = word & spv::OpCodeMask;
        GGML_ASSERT(wc > 0 && pos + wc <= word_गिनती);
        अगर (op == spv::OpExtension && wc >= 2) {
            स्थिर अक्षर * s = reinterpret_cast<स्थिर अक्षर *>(&code[pos + 1]);
            अगर (म_भेद(s, kDecodeVectorExt) == 0) {
                uses_decode_vector = true;
                अवरोध;
            }
        }
        pos += wc;
    }

    अगर (!uses_decode_vector) {
        वापस false;
    }

    VK_LOG_DEBUG("ggml_vk_strip_decode_vector: stripping SPV_NV_cooperative_matrix_decode_vector");

    // Bulk-copy unchanged runs and only break the run when an instruction needs to
    // be dropped or patched. Use reserve + insert/push_back so the destination buffer
    // is touched exactly once (no zero-initialization pass from resize()).
    out.सफा();
    out.reserve(word_गिनती);

    माप_प्रकार run_start = 0;
    स्वतः flush_run = [&](माप_प्रकार up_to) {
        अगर (up_to > run_start) {
            out.insert(out.end(), code + run_start, code + up_to);
        }
    };

    क्रम (माप_प्रकार pos = 5; pos < word_गिनती; ) {
        uint32_t word = code[pos];
        uint32_t wc   = word >> spv::Word_गिनती_Shift;
        uint32_t op   = word & spv::OpCodeMask;
        GGML_ASSERT(wc > 0 && pos + wc <= word_गिनती);

        अगर (op == spv::OpExtension && wc >= 2) {
            स्थिर अक्षर * s = reinterpret_cast<स्थिर अक्षर *>(&code[pos + 1]);
            अगर (म_भेद(s, kDecodeVectorExt) == 0) {
                flush_run(pos);
                pos += wc;
                run_start = pos;
                जारी;
            }
        }

        अगर (op == spv::OpCapability && wc == 2 && code[pos + 1] == kSpvCapabilityCooperativeMatrixDecodeVectorNV) {
            flush_run(pos);
            pos += wc;
            run_start = pos;
            जारी;
        }

        अगर (op == kSpvOpCooperativeMatrixLoadTensorNV) {
            // [opcode/wc][ResultType][Result][Pointer][Object][TensorLayout][MemOperand mask][mem extras...][TA mask][ta extras...]
            GGML_ASSERT(wc >= 8);

            uint32_t mem_mask = code[pos + 6];
            माप_प्रकार   cur      = pos + 7;
            // Each of these MemoryAccess bits (when set) carries one trailing operand.
            cur += (mem_mask & 0x2)     ? 1 : 0; // Aligned
            cur += (mem_mask & 0x8)     ? 1 : 0; // MakePointerAvailable
            cur += (mem_mask & 0x10)    ? 1 : 0; // MakePointerVisible
            cur += (mem_mask & 0x10000) ? 1 : 0; // AliasScopeINTELMask
            cur += (mem_mask & 0x20000) ? 1 : 0; // NoAliasINTELMask
            GGML_ASSERT(cur < pos + wc);

            uint32_t ta_mask = code[cur];
            अगर ((ta_mask & kSpvTensorAddressingDecodeVectorFuncBit) == 0) {
                pos += wc;
                जारी; // leave instruction inside the current unchanged run
            }

            flush_run(pos);

            // Append unchanged prefix of the instruction (header through the mem-extras).
            माप_प्रकार inst_start = out.size();
            माप_प्रकार pre_n      = cur - pos;
            out.insert(out.end(), code + pos, code + pos + pre_n);

            // Emit TA mask with the DecodeVectorFunc bit cleared.
            out.push_back(ta_mask & ~kSpvTensorAddressingDecodeVectorFuncBit);

            // TA extras: TensorView (0x1) and DecodeFunc (0x2) are kept verbatim;
            // DecodeVectorFunc (0x4) is dropped along with its trailing id operand.
            माप_प्रकार keep_ta_extras = ((ta_mask & 0x1) ? 1 : 0) + ((ta_mask & 0x2) ? 1 : 0);
            अगर (keep_ta_extras) {
                out.insert(out.end(), code + cur + 1, code + cur + 1 + keep_ta_extras);
            }

            GGML_ASSERT(wc == pre_n + 1 + keep_ta_extras + 1);

            // Patch the instruction header with the new (one-shorter) word count.
            uint32_t new_wc = wc - 1;
            out[inst_start] = (new_wc << spv::Word_गिनती_Shift) | op;

            pos += wc;
            run_start = pos;
            जारी;
        }

        pos += wc;
    }

    flush_run(word_गिनती);
    वापस true;
}

अटल bool ggml_vk_roll_bk_loop(स्थिर uint32_t * code, माप_प्रकार word_गिनती, std::vector<uint32_t> & out) {
    अगर (word_गिनती < 5) {
        वापस false;
    }

    काष्ठा vk_spv_loop {
        माप_प्रकार   header;
        माप_प्रकार   end;
        uint32_t control;
    };

    std::vector<vk_spv_loop> loops;

    // Collect a list of all loops in the module.
    क्रम (माप_प्रकार pos = 5; pos < word_गिनती; ) {
        स्थिर uint32_t wc = code[pos] >> spv::Word_गिनती_Shift;
        स्थिर uint32_t op = code[pos] & spv::OpCodeMask;
        अगर (wc == 0 || pos + wc > word_गिनती) {
            वापस false;
        }

        अगर (op == spv::OpLoopMerge && wc >= 4) { loops.push_back({ pos, 0, code[pos + 3] }); }

        अगर (op == spv::OpLabel && wc >= 2) {
            क्रम (स्वतः & l : loops) {
                अगर (l.end == 0 && code[l.header + 1] == code[pos + 1]) { l.end = pos; }
            }
        }

        pos += wc;
    }

    स्वतः encloses = [](स्थिर vk_spv_loop & a, स्थिर vk_spv_loop & b) {
        वापस a.header < b.header && b.header < a.end;
    };

    // Find the BK loop.
    स्थिर vk_spv_loop * bk = nullptr;
    क्रम (स्थिर स्वतः & h : loops) {
        अगर (h.control != spv::LoopControlUnrollMask) {
            जारी;
        }
        स्थिर vk_spv_loop * parent = nullptr;
        bool has_child = false;
        क्रम (स्थिर स्वतः & g : loops) {
            अगर (encloses(g, h) && (!parent || g.header > parent->header)) {
                parent = &g;
            }
            अगर (encloses(h, g)) {
                has_child = true;
            }
        }
        // BK loop should be the last loop nested inside the loop with no hint
        // and have at least one child loop.
        अगर (parent &&
            parent->control == spv::LoopControlMaskNone &&
            has_child &&
            (!bk || h.header > bk->header)) {
            bk = &h;
        }
    }
    अगर (!bk) {
        वापस false;
    }

    // set DontUnroll instead of Unroll
    out.assign(code, code + word_गिनती);
    out[bk->header + 3] = spv::LoopControlDontUnrollMask;
    वापस true;
}

अटल व्योम ggml_vk_create_pipeline_func(vk_device& device, vk_pipeline& pipeline, माप_प्रकार spv_आकार, स्थिर व्योम* spv_तथ्य, स्थिर std::string entrypoint,
                                         uint32_t parameter_गिनती, std::array<uint32_t, 3> wg_denoms, std::vector<uint32_t> specialization_constants,
                                         bool disable_robustness, bool require_full_subgroups, uint32_t required_subgroup_आकार) {
    VK_LOG_DEBUG("ggml_vk_create_pipeline(" << device->name << ", " << pipeline->name << ", " << entrypoint << ", " << parameter_गिनती <<
                 ", (" << wg_denoms[0] << "," << wg_denoms[1] << "," << wg_denoms[2] << "), specialization_constants, " <<
                 disable_robustness << ", " << require_full_subgroups << ", " << required_subgroup_आकार << ")");
    GGML_ASSERT(parameter_गिनती > 0);
    GGML_ASSERT(parameter_गिनती <= MAX_PARAMETER_गिनती);
    GGML_ASSERT(wg_denoms[0] > 0 && wg_denoms[1] > 0 && wg_denoms[2] > 0); // NOLINT

    vk::ShaderModuleCreateInfo shader_module_create_info({}, spv_आकार, reinterpret_cast<स्थिर uint32_t *>(spv_तथ्य));

    // Patch SPIR-V to enable supported FP16 float controls, avoiding the need
    // for separate shader variants.
    std::vector<uint32_t> spirv;
    अगर (device->float_controls_rte_fp16 || device->float_controls_denorm_preserve_fp16) {
        स्थिर uint32_t* spv_शब्दगण = reinterpret_cast<स्थिर uint32_t *>(spv_तथ्य);
        माप_प्रकार word_गिनती = spv_आकार / माप(uint32_t);
        spirv.assign(spv_शब्दगण, spv_शब्दगण + word_गिनती);

        // Find insertion points respecting SPIR-V layout order:
        //   Header(5) -> OpCapability -> OpExtension -> ... -> OpEntryPoint -> OpExecutionMode -> ...
        माप_प्रकार pos = 5; // skip header
        माप_प्रकार cap_जोड़ो_pos = pos;
        माप_प्रकार ext_जोड़ो_pos = pos;
        माप_प्रकार exec_जोड़ो_pos = pos;
        uint32_t entry_point_id = 0;

        जबतक (pos < spirv.size()) {
            uint32_t opcode = spirv[pos] & spv::OpCodeMask;
            uint32_t len    = spirv[pos] >> spv::Word_गिनती_Shift;
            अगर (len == 0) अवरोध;

            अगर (opcode == spv::OpCapability) {
                cap_जोड़ो_pos = pos + len;
                ext_जोड़ो_pos = pos + len;
            } अन्यथा अगर (opcode == spv::OpExtension) {
                ext_जोड़ो_pos = pos + len;
            } अन्यथा अगर (opcode == spv::OpEntryPoint) {
                entry_point_id = spirv[pos + 2];
                exec_जोड़ो_pos = pos + len;
            } अन्यथा अगर (opcode == spv::OpExecutionMode || opcode == spv::OpExecutionModeId) {
                exec_जोड़ो_pos = pos + len;
            } अन्यथा अगर (entry_point_id != 0) {
                अवरोध;
            }

            pos += len;
        }

        // Insert from latest position first so earlier indices stay valid.

        अगर (device->float_controls_rte_fp16) {
            // OpExecutionMode %entrypoint RoundingModeRTE 16
            uint32_t exec_mode[] = { (4u << spv::Word_गिनती_Shift) | spv::OpExecutionMode, entry_point_id, spv::ExecutionModeRoundingModeRTE, 16 };
            spirv.insert(spirv.begin() + exec_जोड़ो_pos, std::begin(exec_mode), std::end(exec_mode));
        }

        अगर (device->float_controls_denorm_preserve_fp16) {
            // OpExecutionMode %entrypoint DenormPreserve 16
            uint32_t exec_mode[] = { (4u << spv::Word_गिनती_Shift) | spv::OpExecutionMode, entry_point_id, spv::ExecutionModeDenormPreserve, 16 };
            spirv.insert(spirv.begin() + exec_जोड़ो_pos, std::begin(exec_mode), std::end(exec_mode));
        }

        // OpExtension "SPV_KHR_float_controls"
        स्थिर अक्षर ext_str[] = "SPV_KHR_float_controls";
        माप_प्रकार ext_str_शब्दगण = CEIL_DIV(माप(ext_str), माप(uint32_t));
        std::vector<uint32_t> extension(1 + ext_str_शब्दगण, 0);
        extension[0] = (uint32_t)((1 + ext_str_शब्दगण) << spv::Word_गिनती_Shift) | spv::OpExtension;
        स_नकल(&extension[1], ext_str, माप(ext_str));
        spirv.insert(spirv.begin() + ext_जोड़ो_pos, extension.begin(), extension.end());

        अगर (device->float_controls_rte_fp16) {
            // OpCapability RoundingModeRTE
            uint32_t capability[] = { (2u << spv::Word_गिनती_Shift) | spv::OpCapability, spv::CapabilityRoundingModeRTE };
            spirv.insert(spirv.begin() + cap_जोड़ो_pos, std::begin(capability), std::end(capability));
        }

        अगर (device->float_controls_denorm_preserve_fp16) {
            // OpCapability DenormPreserve
            uint32_t capability[] = { (2u << spv::Word_गिनती_Shift) | spv::OpCapability, spv::CapabilityDenormPreserve };
            spirv.insert(spirv.begin() + cap_जोड़ो_pos, std::begin(capability), std::end(capability));
        }

        shader_module_create_info = vk::ShaderModuleCreateInfo({}, spirv.size() * माप(uint32_t), spirv.तथ्य());
    }

#अगर defined(GGML_VULKAN_COOPMAT2_DECODE_VECTOR_GLSLC_SUPPORT)
    अगर (device->coopmat2 && !device->coopmat2_decode_vector) {
        स्थिर uint32_t * src   = spirv.empty() ? reinterpret_cast<स्थिर uint32_t *>(spv_तथ्य) : spirv.तथ्य();
        माप_प्रकार           src_n = spirv.empty() ? spv_आकार / माप(uint32_t) : spirv.size();
        std::vector<uint32_t> stripped;
        अगर (ggml_vk_strip_decode_vector(src, src_n, stripped)) {
            spirv = std::move(stripped);
            shader_module_create_info = vk::ShaderModuleCreateInfo({}, spirv.size() * माप(uint32_t), spirv.तथ्य());
        }
    }
#पूर्ण_अगर

#अगर VK_HEADER_VERSION >= 287
    // Roll the mul_mm BK loop on Asahi Linux. Skip bf16 and the mul_mmq pipelines.
    अगर (device->driver_id == vk::DriverId::eMesaHoneykrisp &&
        pipeline->name.rfind("matmul", 0) == 0 &&
        pipeline->name.find("bf16") == std::string::npos &&
        pipeline->name.find("q8_1") == std::string::npos) {
        स्थिर uint32_t * src   = spirv.empty() ? reinterpret_cast<स्थिर uint32_t *>(spv_तथ्य) : spirv.तथ्य();
        माप_प्रकार           src_n = spirv.empty() ? spv_आकार / माप(uint32_t) : spirv.size();
        std::vector<uint32_t> rolled;
        अगर (ggml_vk_roll_bk_loop(src, src_n, rolled)) {
            spirv = std::move(rolled);
            shader_module_create_info = vk::ShaderModuleCreateInfo({}, spirv.size() * माप(uint32_t), spirv.तथ्य());
        }
    }
#पूर्ण_अगर

    प्रयास {
        pipeline->shader_module = device->device.createShaderModule(shader_module_create_info);
    } पकड़ो (स्थिर vk::SystemError& e) {
        std::अ_त्रुटि << "ggml_vulkan: shader module creation failed for " << pipeline->name << ": " << e.what() << std::प_पूर्ण;
        फेंको;
    }

    vk::धकेलो_Constant_Range pcr(
        vk::ShaderStageFlagBits::eCompute,
        0,
        pipeline->धकेलो_constant_आकार
    );

    vk::PipelineLayoutCreateInfo pipeline_layout_create_info(vk::PipelineLayoutCreateFlags(), device->dsl, pcr);
    प्रयास {
        pipeline->layout = device->device.createPipelineLayout(pipeline_layout_create_info);
    } पकड़ो (स्थिर vk::SystemError& e) {
        std::अ_त्रुटि << "ggml_vulkan: pipeline layout creation failed for " << pipeline->name << ": " << e.what() << std::प_पूर्ण;
        फेंको;
    }

    std::vector<vk::SpecializationMapEntry> specialization_entries(specialization_constants.size());

    क्रम (माप_प्रकार i = 0; i < specialization_constants.size(); i++) {
        specialization_entries[i].constantID = i;
        specialization_entries[i].offset = i * माप(uint32_t);
        specialization_entries[i].size = माप(uint32_t);
    }

    vk::SpecializationInfo specialization_info(
        specialization_entries.size(),
        specialization_entries.तथ्य(),
        specialization_constants.size() * माप(uint32_t),
        specialization_constants.तथ्य()
    );

    vk::PipelineShaderStageCreateFlags pipeline_shader_stage_create_flags{};

    अगर (device->subgroup_require_full_support && require_full_subgroups) {
        pipeline_shader_stage_create_flags |= vk::PipelineShaderStageCreateFlagBits::eRequireFullSubgroupsEXT;
    }

    vk::PipelineShaderStageCreateInfo pipeline_shader_create_info(
            pipeline_shader_stage_create_flags,
            vk::ShaderStageFlagBits::eCompute,
            pipeline->shader_module,
            entrypoint.c_str(),
            &specialization_info);

    vk::Pipeline_Shader_Stage_Required_Subgroup_आकार_Create_Info_EXT pipeline_shader_stage_required_subgroup_आकार_create_info;
    pipeline_shader_stage_required_subgroup_आकार_create_info.required_Subgroup_आकार = required_subgroup_आकार;
    अगर (device->subgroup_आकार_control && required_subgroup_आकार > 0) {
        GGML_ASSERT(device->subgroup_min_आकार <= required_subgroup_आकार && required_subgroup_आकार <= device->subgroup_max_आकार);
        pipeline_shader_create_info.setPNext(&pipeline_shader_stage_required_subgroup_आकार_create_info);
    }

    vk::ComputePipelineCreateInfo compute_pipeline_create_info(
        device->pipeline_executable_properties_support ?
            vk::PipelineCreateFlagBits::eCaptureStatisticsKHR :
            vk::PipelineCreateFlags{},
        pipeline_shader_create_info,
        pipeline->layout);

    vk::PipelineRobustnessCreateInfoEXT rci;

    अगर (device->pipeline_robustness && disable_robustness) {
        rci.storageBuffers = vk::Pipeline_Robustness_बफर_Behavior_EXT::eDisabled;
        rci.uniformBuffers = vk::Pipeline_Robustness_बफर_Behavior_EXT::eDisabled;
        compute_pipeline_create_info.setPNext(&rci);
    }

#अगर defined(VK_EXT_shader_64bit_indexing)
    vk::PipelineCreateFlags2CreateInfo pipelineFlags2CreateInfo;
    अगर (pipeline->is_64b_indexing)
    {
        pipelineFlags2CreateInfo.flags = vk::PipelineCreateFlagBits2::e64BitIndexingEXT;
        अगर (device->pipeline_executable_properties_support) {
            pipelineFlags2CreateInfo.flags |= vk::PipelineCreateFlagBits2::eCaptureStatisticsKHR;
        }
        pipelineFlags2CreateInfo.setPNext(compute_pipeline_create_info.p_अग्र);
        compute_pipeline_create_info.setPNext(&pipelineFlags2CreateInfo);
    }
#पूर्ण_अगर

    प्रयास {
        pipeline->pipeline = device->device.createComputePipeline(VK_NULL_HANDLE, compute_pipeline_create_info).value;
    } पकड़ो (स्थिर vk::SystemError& e) {
        std::अ_त्रुटि << "ggml_vulkan: compute pipeline creation failed for " << pipeline->name << ": " << e.what() << std::प_पूर्ण;
        फेंको;
    }

    अगर (vk_instance.debug_utils_support) {
        vk::DebugUtilsObjectNameInfoEXT duoni;
        duoni.objectType = vk::ObjectType::ePipeline;
        duoni.pObjectName = pipeline->name.c_str();
        duoni.objectHandle = /*reinterpret_cast*/(uint64_t)(static_cast<VkPipeline>(pipeline->pipeline));
        vk_instance.pfn_vkSetDebugUtilsObjectNameEXT(device->device, &static_cast<VkDebugUtilsObjectNameInfoEXT &>(duoni));
    }

    अगर (device->pipeline_executable_properties_support) {
        vk::PipelineExecutableInfoKHR executableInfo;
        executableInfo.pipeline = pipeline->pipeline;

        स्वतः statistics = device->device.getPipelineExecutableStatisticsKHR(executableInfo);

        bool print_stats = !vk_pipeline_stats_filter.empty() &&
                           pipeline->name.find(vk_pipeline_stats_filter) != std::string::npos;
        अगर (print_stats) {
            std::अ_त्रुटि << "ggml_vulkan: pipeline stats for " << pipeline->name << ":" << std::प_पूर्ण;
        }

        क्रम (स्वतः & s : statistics) {
            अगर (print_stats) {
                std::अ_त्रुटि << "ggml_vulkan:   " << s.name.तथ्य() << ": ";
                चयन (s.format) {
                    हाल vk::PipelineExecutableStatisticFormatKHR::eBool32:
                        std::अ_त्रुटि << (s.value.b32 ? "true" : "false");
                        अवरोध;
                    हाल vk::PipelineExecutableStatisticFormatKHR::eInt64:
                        std::अ_त्रुटि << s.value.i64;
                        अवरोध;
                    हाल vk::PipelineExecutableStatisticFormatKHR::eUint64:
                        std::अ_त्रुटि << s.value.u64;
                        अवरोध;
                    हाल vk::PipelineExecutableStatisticFormatKHR::eFloat64:
                        std::अ_त्रुटि << s.value.f64;
                        अवरोध;
                }
                std::अ_त्रुटि << std::प_पूर्ण;
            }
            // "Register Count" is reported by NVIDIA drivers.
            अगर (म_भेद(s.name, "Register Count") == 0) {
                VK_LOG_DEBUG(pipeline->name << " " << s.name << ": " << s.value.u64 << " registers");
                pipeline->register_गिनती = (uint32_t)s.value.u64;
            }
        }
    }

    {
        std::lock_guard<std::mutex> guard(device->compile_mutex);
        device->all_pipelines.push_back(pipeline);
        pipeline->compiled = true;
        pipeline->compile_pending = false;
    }
    device->compile_cv.notify_all();
}

व्योम ggml_vk_destroy_pipeline(vk::Device& device, vk_pipeline& pipeline) {
    VK_LOG_DEBUG("ggml_pipeline_destroy_pipeline(" << pipeline->name << ")");
    device.destroyPipelineLayout(pipeline->layout);

    device.destroyShaderModule(pipeline->shader_module);

    device.destroyPipeline(pipeline->pipeline);
}

व्योम ggml_pipeline_request_descriptor_sets(ggml_backend_vk_context *ctx, vk_pipeline& pipeline, uint32_t n) {
    VK_LOG_DEBUG("ggml_pipeline_request_descriptor_sets(" << pipeline->name << ", " << n << ")");
    ctx->pipeline_descriptor_set_requirements += n;
    अगर (!pipeline->compiled) {
        ggml_vk_load_shaders(ctx->device, pipeline);
    }
    ggml_pipeline_allocate_descriptor_sets(ctx);
}

व्योम ggml_pipeline_allocate_descriptor_sets(ggml_backend_vk_context * ctx) {

    अगर (ctx->descriptor_sets.size() >= ctx->pipeline_descriptor_set_requirements) {
        // Enough descriptors are available
        वापस;
    }

    vk_device& device = ctx->device;

    // Grow by 50% to avoid frequent allocations
    uint32_t needed = std::max(3 * ctx->descriptor_sets.size() / 2, माप_प्रकार{ctx->pipeline_descriptor_set_requirements});
    uint32_t to_alloc = needed - ctx->descriptor_sets.size();
    uint32_t pool_remaining = VK_DEVICE_DESCRIPTOR_POOL_आकार - ctx->descriptor_sets.size() % VK_DEVICE_DESCRIPTOR_POOL_आकार;
    uint32_t pool_idx = ctx->descriptor_sets.size() / VK_DEVICE_DESCRIPTOR_POOL_आकार;

    जबतक (to_alloc > 0) {
        स्थिर uint32_t alloc_गिनती = std::min(pool_remaining, to_alloc);
        to_alloc -= alloc_गिनती;
        pool_remaining = VK_DEVICE_DESCRIPTOR_POOL_आकार;

        अगर (pool_idx >= ctx->descriptor_pools.size()) {
            vk::Descriptor_Pool_आकार descriptor_pool_आकार(vk::DescriptorType::e_Storage_बफर, MAX_PARAMETER_गिनती * VK_DEVICE_DESCRIPTOR_POOL_आकार);
            vk::DescriptorPoolCreateInfo descriptor_pool_create_info({}, VK_DEVICE_DESCRIPTOR_POOL_आकार, descriptor_pool_आकार);
            ctx->descriptor_pools.push_back(device->device.createDescriptorPool(descriptor_pool_create_info));
        }

        std::vector<vk::DescriptorSetLayout> layouts(alloc_गिनती);
        क्रम (uint32_t i = 0; i < alloc_गिनती; i++) {
            layouts[i] = device->dsl;
        }
        vk::DescriptorSetAllocateInfo descriptor_set_alloc_info(ctx->descriptor_pools[pool_idx], alloc_गिनती, layouts.तथ्य());
        std::vector<vk::DescriptorSet> sets = device->device.allocateDescriptorSets(descriptor_set_alloc_info);
        ctx->descriptor_sets.insert(ctx->descriptor_sets.end(), sets.begin(), sets.end());

        pool_idx++;
    }
    ctx->descriptor_set_bindings.resize(ctx->descriptor_sets.size());
}

अटल vk_command_बफर* ggml_vk_create_cmd_बफर(vk_device& device, vk_command_pool& p) {
    VK_LOG_DEBUG("ggml_vk_create_cmd_buffer()");
    vk::Command_बफर_Allocate_Info command_बफर_alloc_info(
        p.pool,
        vk::Command_बफर_Level::ePrimary,
        1);
    स्थिर std::vector<vk::Command_बफर> cmd_buffers = device->device.allocateCommandBuffers(command_बफर_alloc_info);
    p.cmd_buffers.push_back({ cmd_buffers.front(), 0, true });
    वापस &p.cmd_buffers[p.cmd_buffers.size()-1];
}

व्योम ggml_vk_submit(vk_context& ctx, vk::Fence fence) {
    अगर (ctx->seqs.empty()) {
        अगर (fence) {
            ctx->p->q->handle->submit({}, fence);
        }
        वापस;
    }
    VK_LOG_DEBUG("ggml_vk_submit(" << ctx << ", " << fence << ")");

    std::vector<std::vector<uint64_t>> tl_wait_vals;
    std::vector<std::vector<uint64_t>> tl_signal_vals;
    std::vector<std::vector<vk::Semaphore>> tl_wait_semaphores;
    std::vector<std::vector<vk::Semaphore>> tl_signal_semaphores;
    std::vector<vk::TimelineSemaphoreSubmitInfo> tl_submit_infos;
    std::vector<vk::SubmitInfo> submit_infos;
    पूर्णांक idx = -1;
    std::vector<std::vector<vk::PipelineStageFlags>> stage_flags;

    माप_प्रकार reserve = 0;

    क्रम (स्थिर स्वतः& sequence : ctx->seqs) {
        reserve += sequence.size();
    }

    // Pre-reserve vectors to prevent reallocation, which invalidates pointers
    tl_wait_semaphores.reserve(reserve);
    tl_wait_vals.reserve(reserve);
    tl_signal_semaphores.reserve(reserve);
    tl_signal_vals.reserve(reserve);
    tl_submit_infos.reserve(reserve);
    submit_infos.reserve(reserve);
    stage_flags.reserve(reserve);

    क्रम (स्थिर स्वतः& sequence : ctx->seqs) {
        क्रम (स्थिर स्वतः& submission : sequence) {
            stage_flags.push_back({});
            idx++;
            tl_wait_vals.push_back({});
            tl_wait_semaphores.push_back({});
            tl_signal_vals.push_back({});
            tl_signal_semaphores.push_back({});
            क्रम (माप_प्रकार i = 0; i < submission.wait_semaphores.size(); i++) {
                stage_flags[idx].push_back(ctx->p->q->stage_flags);
                tl_wait_vals[idx].push_back(submission.wait_semaphores[i].value);
                tl_wait_semaphores[idx].push_back(submission.wait_semaphores[i].s);
            }
            क्रम (माप_प्रकार i = 0; i < submission.signal_semaphores.size(); i++) {
                tl_signal_vals[idx].push_back(submission.signal_semaphores[i].value);
                tl_signal_semaphores[idx].push_back(submission.signal_semaphores[i].s);
            }
            tl_submit_infos.push_back({
                (uint32_t) submission.wait_semaphores.size(),
                tl_wait_vals[idx].तथ्य(),
                (uint32_t) submission.signal_semaphores.size(),
                tl_signal_vals[idx].तथ्य(),
            });
            tl_submit_infos[idx].sType = vk::StructureType::eTimelineSemaphoreSubmitInfo;
            tl_submit_infos[idx].p_अग्र = nullptr;
            vk::SubmitInfo si{
                (uint32_t) submission.wait_semaphores.size(),
                tl_wait_semaphores[idx].तथ्य(),
                stage_flags[idx].तथ्य(),
                1,
                &submission.बफर->buf,
                (uint32_t) submission.signal_semaphores.size(),
                tl_signal_semaphores[idx].तथ्य(),
            };
            si.setPNext(&tl_submit_infos[idx]);
            submit_infos.push_back(si);
        }
    }

    ctx->p->q->handle->submit(submit_infos, fence);

    ctx->seqs.सफा();
}

uint32_t ggml_vk_find_queue_family_सूचकांक(std::vector<vk::QueueFamilyProperties>& queue_family_props, स्थिर vk::QueueFlags& required, स्थिर vk::QueueFlags& avoid, int32_t compute_सूचकांक, uint32_t min_अंक_queues) {
    VK_LOG_DEBUG("ggml_vk_find_queue_family_index()");
    स्थिर uint32_t qfsize = queue_family_props.size();

    // Try with avoid preferences first
    क्रम (uint32_t i = 0; i < qfsize; i++) {
        अगर (queue_family_props[i].queue_गिनती >= min_अंक_queues && (compute_सूचकांक < 0 || i != (uint32_t) compute_सूचकांक) && queue_family_props[i].queueFlags & required && !(queue_family_props[i].queueFlags & avoid)) {
            वापस i;
        }
    }

    // Fall back to only required
    क्रम (माप_प्रकार i = 0; i < qfsize; i++) {
        अगर (queue_family_props[i].queue_गिनती >= min_अंक_queues && (compute_सूचकांक < 0 || i != (uint32_t) compute_सूचकांक) && queue_family_props[i].queueFlags & required) {
            वापस i;
        }
    }

    // Fall back to reusing compute queue
    क्रम (माप_प्रकार i = 0; i < qfsize; i++) {
        अगर (queue_family_props[i].queue_गिनती >= min_अंक_queues && queue_family_props[i].queueFlags & required) {
            वापस i;
        }
    }

    // Fall back to ignoring min_num_queries
    क्रम (माप_प्रकार i = 0; i < qfsize; i++) {
        अगर (queue_family_props[i].queueFlags & required) {
            वापस i;
        }
    }

    // All commands that are allowed on a queue that supports transfer operations are also allowed on a queue that supports either graphics or compute operations.
    // Thus, if the capabilities of a queue family include VK_QUEUE_GRAPHICS_BIT or VK_QUEUE_COMPUTE_BIT, then reporting the VK_QUEUE_TRANSFER_BIT capability separately for that queue family is optional.
    अगर (compute_सूचकांक >= 0) {
        वापस compute_सूचकांक;
    }

    std::अ_त्रुटि << "ggml_vulkan: No suitable queue family index found." << std::प_पूर्ण;

    क्रम(स्वतः &q_family : queue_family_props) {
        std::अ_त्रुटि << "Queue number: "  + std::to_string(q_family.queue_गिनती) << " flags: " + to_string(q_family.queueFlags) << std::प_पूर्ण;
    }
    पात();
}

std::unique_ptr<vk_queue> ggml_vk_create_queue(vk_device& device, uint32_t queue_family_सूचकांक, uint32_t queue_सूचकांक, vk::PipelineStageFlags&& stage_flags, bool transfer_only) {
    VK_LOG_DEBUG("ggml_vk_create_queue()");
    std::lock_guard<std::recursive_mutex> guard(device->mutex);

    स्वतः q = std::make_unique<vk_queue>();
    q->queue_family_सूचकांक = queue_family_सूचकांक;
    q->transfer_only = transfer_only;

    std::shared_ptr<vk_queue_handle> h;
    vk::DeviceQueueInfo2 queue_info2{};
    queue_info2.queue_Family_सूचकांक = queue_family_सूचकांक;
    queue_info2.queue_सूचकांक_2 = queue_सूचकांक;

    अगर (device->has_internally_synchronized_queues) {
        h = std::make_shared<vk_queue_handle_unsynchronized>();
        queue_info2.flags = eInternallySynchronizedKHR;
    } अन्यथा {
        h = std::make_shared<vk_queue_handle_synchronized>();
    }

    h->queue = device->device.getQueue2(queue_info2);
    h->device = device;
    // Avoid concurrent submissions on NVIDIA due to driver bug.
    अगर (device->vendor_id == VK_VENDOR_ID_NVIDIA) {
        h->device_submit_mutex = &device->queue_submit_mutex;
    }
    q->handle = h;

    q->cmd_pool.init(device, q.लो());

    q->stage_flags = stage_flags;
    वापस q;
}

std::unique_ptr<vk_queue> ggml_vk_create_aliased_queue(vk_device& device, स्थिर std::unique_ptr<vk_queue>& source) {
    std::lock_guard<std::recursive_mutex> guard(device->mutex);
    स्वतः q = std::make_unique<vk_queue>();
    q->handle = source->handle;
    q->queue_family_सूचकांक = source->queue_family_सूचकांक;
    q->stage_flags = source->stage_flags;
    q->transfer_only = source->transfer_only;
    q->cmd_pool.init(device, q.लो());
    वापस q;
}

vk_context ggml_vk_create_context(ggml_backend_vk_context * ctx, vk_command_pool& p) {
    vk_context परिणाम = std::make_shared<vk_context_struct>();
    VK_LOG_DEBUG("ggml_vk_create_context(" << परिणाम << ")");
    ctx->gc.contexts.emplace_back(परिणाम);
    परिणाम->p = &p;
    वापस परिणाम;
}

vk_context ggml_vk_create_temporary_context(vk_command_pool& p) {
    vk_context परिणाम = std::make_shared<vk_context_struct>();
    VK_LOG_DEBUG("ggml_vk_create_temporary_context(" << परिणाम << ")");
    परिणाम->p = &p;
    वापस परिणाम;
}

अटल vk_semaphore * ggml_vk_create_द्विआधारी_semaphore(ggml_backend_vk_context * ctx) {
    VK_LOG_DEBUG("ggml_vk_create_timeline_semaphore()");
    vk::SemaphoreTypeCreateInfo tci{ vk::SemaphoreType::e_द्विआधारी, 0 };
    vk::SemaphoreCreateInfo ci{};
    ci.setPNext(&tci);
    vk::Semaphore semaphore = ctx->device->device.createSemaphore(ci);
    ctx->gc.semaphores.push_back({ semaphore, 0 });
    वापस &ctx->gc.semaphores[ctx->gc.semaphores.size() - 1];
}

अटल vk_semaphore * ggml_vk_create_timeline_semaphore(ggml_backend_vk_context * ctx) {
    VK_LOG_DEBUG("ggml_vk_create_timeline_semaphore()");
    अगर (ctx->semaphore_idx >= ctx->gc.tl_semaphores.size()) {
        vk::SemaphoreTypeCreateInfo tci{ vk::SemaphoreType::eTimeline, 0 };
        vk::SemaphoreCreateInfo ci{};
        ci.setPNext(&tci);
        vk::Semaphore semaphore = ctx->device->device.createSemaphore(ci);
        ctx->gc.tl_semaphores.push_back({ semaphore, 0 });
    }
    वापस &ctx->gc.tl_semaphores[ctx->semaphore_idx++];
}

अटल vk::Event ggml_vk_create_event(ggml_backend_vk_context * ctx) {
    अगर (ctx->event_idx >= ctx->gc.events.size()) {
        ctx->gc.events.push_back(ctx->device->device.createEvent({}));
    }
    वापस ctx->gc.events[ctx->event_idx++];
}

व्योम ggml_vk_command_pool_cleanup(vk_device& device, vk_command_pool& p) {
    VK_LOG_DEBUG("ggml_vk_command_pool_cleanup()");

    // Requires command buffers to be done
    device->device.resetCommandPool(p.pool);
    // Don't clear the command buffers and mark them as not in use.
    // This allows us to reuse them
    क्रम (स्वतः& cmd_बफर : p.cmd_buffers) {
        cmd_बफर.in_use = false;
    }
}

व्योम ggml_vk_queue_command_pools_cleanup(vk_device& device) {
    VK_LOG_DEBUG("ggml_vk_queue_command_pools_cleanup()");

    // Arbitrary frequency to cleanup/reuse command buffers
    अटल constexpr uint32_t cleanup_frequency = 10;

    अगर (device->compute_queue && device->compute_queue->cmd_pool.buffers_in_use() >= cleanup_frequency) {
        ggml_vk_command_pool_cleanup(device, device->compute_queue->cmd_pool);
    }
    अगर (device->transfer_queue && device->transfer_queue->cmd_pool.buffers_in_use() >= cleanup_frequency) {
        ggml_vk_command_pool_cleanup(device, device->transfer_queue->cmd_pool);
    }
}

vk_subbuffer ggml_vk_subbuffer(स्थिर ggml_backend_vk_context* ctx, स्थिर vk_बफर& buf, माप_प्रकार offset) {
    वापस { buf, offset, ggml_vk_get_max_बफर_range(ctx, buf, offset) };
}

व्योम ggml_vk_sync_buffers(ggml_backend_vk_context* ctx, vk_context& subctx) {
    VK_LOG_DEBUG("ggml_vk_sync_buffers()");

    स्थिर bool transfer_queue = subctx->p->q->transfer_only;

    अगर (ctx) {
        ctx->prealloc_x_need_sync = ctx->prealloc_y_need_sync = ctx->prealloc_split_k_need_sync = false;
    }

    subctx->s->बफर->buf.pipelineBarrier(
        subctx->p->q->stage_flags,
        subctx->p->q->stage_flags,
        {},
        { {
          { !transfer_queue ? (vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite) : (vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite) },
          { !transfer_queue ? (vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite) : (vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite) }
        } },
        {},
        {}
    );
}

अटल व्योम ggml_vk_reset_event(vk_context& ctx, vk::Event& event) {
    VK_LOG_DEBUG("ggml_vk_set_event()");

    ctx->s->बफर->buf.resetEvent(
        event,
        ctx->p->q->stage_flags
    );
}

व्योम ggml_vk_set_event(vk_context& ctx, vk::Event& event) {
    VK_LOG_DEBUG("ggml_vk_set_event()");

    ctx->s->बफर->buf.setEvent(
        event,
        ctx->p->q->stage_flags
    );
}

व्योम ggml_vk_wait_events(vk_context& ctx, std::vector<vk::Event>&& events) {
    VK_LOG_DEBUG("ggml_vk_wait_events()");
    अगर (events.empty()) {
        वापस;
    }

    ctx->s->बफर->buf.waitEvents(
        events,
        ctx->p->q->stage_flags,
        ctx->p->q->stage_flags,
        {},
        {},
        {}
    );
}

अटल vk_fa_tuning_params get_fa_tuning_params_scalar(स्थिर vk_device& device, uint32_t hsk, uint32_t hsv, uint32_t n_rows, uint32_t n_kv, ggml_type k_type, ggml_type v_type, bool f32acc) {

    vk_fa_tuning_params परिणाम{};
    परिणाम.path = FA_SCALAR;

    अगर (device->vendor_id == VK_VENDOR_ID_INTEL) {
        // Disable subgroup use due to performance issues when enforcing subgroup sizes
        परिणाम.subgroup_आकार = 32;
        परिणाम.disable_subgroups = true;
    } अन्यथा अगर (device->vendor_id == VK_VENDOR_ID_AMD && device->architecture != AMD_GCN) {
        परिणाम.subgroup_आकार = n_rows < 4 ? 32 : device->subgroup_आकार;
    } अन्यथा {
        परिणाम.subgroup_आकार = device->subgroup_आकार;
    }

    // Row split splits the workgroup so that synchronization only has to happen within subgroups, which avoids barriers
    uint32_t row_split_max_hsk = 64;
    अगर (device->vendor_id == VK_VENDOR_ID_AMD && device->architecture != AMD_GCN && !device->uma) {
        row_split_max_hsk = n_rows <= 8 ? 64 : 128;
    }
    परिणाम.row_split = (n_rows < 4 || hsk <= row_split_max_hsk) ? 1 : 4;

    अगर (परिणाम.subgroup_आकार > 32 && (n_rows < 4 || hsk < (परिणाम.row_split == 1 ? 128 : 64))) {
        परिणाम.workgroup_आकार = परिणाम.subgroup_आकार * 2;
    } अन्यथा {
        परिणाम.workgroup_आकार = परिणाम.subgroup_आकार * 4;
    }

    स्थिर uint32_t D = hsk | hsv;

    स्थिर bool reduce_block_rows = D & 8 || n_kv < 1024 || device->vendor_id == VK_VENDOR_ID_INTEL;

    अगर (n_rows == 1) {
        परिणाम.block_rows = 1;
        परिणाम.block_cols = 64;
    } अन्यथा {
        // row_split 1 means higher register use per row, so block size has to be adjusted
        अगर (परिणाम.row_split == 1) {
            परिणाम.block_rows = n_rows == 2 ? 2 : ((n_rows <= 4 || reduce_block_rows) ? 4 : 8);
        } अन्यथा {
            परिणाम.block_rows = n_rows <= 4 ? 4 : ((n_rows <= 8 || reduce_block_rows) ? 8 : 16);
        }

        परिणाम.block_cols = (D & 8) ? 64 : 32;
    }

    स्थिर uint32_t D_lsb = D ^ (D & (D-1));  // extract lowest set bit

    परिणाम.d_split = std::min(std::min(परिणाम.subgroup_आकार, 8u), D_lsb / 4);

    परिणाम.shmem_staging = (device->vendor_id == VK_VENDOR_ID_NVIDIA && hsk < 256 && hsv < 256) ? 1 : 0;

    अगर (!reduce_block_rows && !ggml_vk_flash_attn_scalar_shmem_support(device, परिणाम, hsk, hsv, f32acc, k_type, v_type)) {
        परिणाम.block_rows /= 2;
    }

    // On AMD RDNA, for small head sizes and big batch size the shader uses few registers, so too many subgroups get scheduled
    // at once and end up thrashing the cache. Fix this by setting a large (unused) shmem buffer that reduces occupancy.
    // This targets an occupancy of 4 subgroups per SIMD.
    अगर (device->vendor_id == VK_VENDOR_ID_AMD && device->properties.limits.max_Compute_Shared_Memory_आकार == 65536) {
        अगर (device->architecture != AMD_GCN && n_rows >= 64 && hsk <= 128) {
            // 30kb target for hsk > 64, 26kb for <= 64 due to smaller workgroup size
            // Values are guessed, tested on RDNA2
            परिणाम.limit_occupancy_shmem = (hsk <= 64 ? 26 : 30) * 1024 / 4 / 4;
        } अन्यथा अगर (device->architecture == AMD_GCN && n_rows <= 8 && hsk >= 256) {
            // Same thing for GCN, with an occupancy target of 2 subgroups per SIMD.
            // Here low-batch FA with large head size is affected.
            // n_rows < 4 switch because workgroup size switches from 128 to 256 there.
            परिणाम.limit_occupancy_shmem = (n_rows < 4 ? 14 : 26) * 1024 / 4 / 4;
        }
    }

    वापस परिणाम;
}

अटल vk_fa_tuning_params get_fa_tuning_params_coopmat1(स्थिर vk_device& device, uint32_t hsk, uint32_t hsv, uint32_t n_rows, uint32_t n_kv, ggml_type k_type, ggml_type v_type, bool f32acc) {
    GGML_UNUSED(n_rows);
    GGML_UNUSED(n_kv);
    GGML_UNUSED(k_type);
    GGML_UNUSED(v_type);
    GGML_UNUSED(f32acc);

    vk_fa_tuning_params परिणाम{};
    परिणाम.path = FA_COOPMAT1;

    स्थिर uint32_t D = hsk | hsv;

    स्थिर uint32_t coopmat_block_rows = 16;
    स्थिर uint32_t coopmat_block_cols = 16;

    स्थिर uint32_t अंक_subgroups = 4;

    परिणाम.block_rows = coopmat_block_rows;
    परिणाम.block_cols = coopmat_block_cols * अंक_subgroups;
    परिणाम.row_split = अंक_subgroups;
    परिणाम.subgroup_आकार = device->subgroup_आकार;
    परिणाम.workgroup_आकार = अंक_subgroups * परिणाम.subgroup_आकार;

    स्थिर uint32_t D_lsb = D ^ (D & (D-1));  // extract lowest set bit
    परिणाम.d_split = std::min(std::min(परिणाम.subgroup_आकार, 8u), D_lsb / 4);

    परिणाम.shmem_staging = (device->vendor_id == VK_VENDOR_ID_NVIDIA && hsk < 256 && hsv < 256) ? 1 : 0;

    वापस परिणाम;
}

अटल vk_fa_tuning_params get_fa_tuning_params_coopmat2(स्थिर vk_device& device, uint32_t hsk, uint32_t hsv, uint32_t n_rows, uint32_t n_kv, ggml_type k_type, ggml_type v_type, bool f32acc) {
    GGML_UNUSED(n_kv);
    GGML_UNUSED(f32acc);

    vk_fa_tuning_params परिणाम{};
    परिणाम.path = FA_COOPMAT2;

    स्थिर uint32_t D = hsk | hsv;

    स्थिर bool small_rows = n_rows < 32;

    अगर (small_rows) {
        परिणाम.block_rows = 32;
        परिणाम.block_cols = 32;
    } अन्यथा अगर (ggml_is_quantized(k_type) || ggml_is_quantized(v_type) || hsk >= 256 || hsv >= 256) {
        परिणाम.block_rows = (hsk >= 512 || hsv >= 512) ? 32 : 64;
        परिणाम.block_cols = 32;
    } अन्यथा {
        परिणाम.block_rows = 64;
        परिणाम.block_cols = 64;
    }

    परिणाम.subgroup_आकार = device->subgroup_आकार;
    परिणाम.workgroup_आकार = (small_rows && (D % 32) == 0) ? 256 : 128;

    वापस परिणाम;
}

vk_fa_tuning_params get_fa_tuning_params(स्थिर vk_device& device, uint32_t hsk, uint32_t hsv, uint32_t n_rows, uint32_t n_kv, ggml_type k_type, ggml_type v_type, bool f32acc) {
    FaCodePath path = device->coopmat2 ? FA_COOPMAT2 :
                      device->coopmat1_fa_support ? FA_COOPMAT1 : FA_SCALAR;

    अगर (path == FA_COOPMAT2 && k_type == GGML_TYPE_BF16 && !device->coopmat2_bf16_support) {
        path = FA_COOPMAT1;
    }
    अगर (path == FA_COOPMAT1 && k_type == GGML_TYPE_BF16 && !device->coopmat_bf16_support) {
        path = FA_SCALAR;
    }

    अगर (path == FA_COOPMAT1 && device->architecture == vk_device_architecture::NVIDIA_TURING) {
        // Nvidia compiler bug, see https://github.com/ggml-org/llama.cpp/pull/19075#issuecomment-3820716090
        path = FA_SCALAR;
    }

    अगर (path == FA_COOPMAT1) {
        bool shape_ok = (f32acc && device->coopmat_support_16x16x16_f32acc) ||
                        (!f32acc && device->coopmat_support_16x16x16_f16acc);
        स्थिर vk_fa_tuning_params params = get_fa_tuning_params_coopmat1(device, hsk, hsv, n_rows, n_kv, k_type, v_type, f32acc);
        bool shmem_ok = ggml_vk_flash_attn_coopmat_shmem_support(device, params, hsk, hsv, f32acc, k_type, v_type);

        अगर (!shape_ok || !shmem_ok) {
            path = FA_SCALAR;
        }
    }

    // scalar is faster than coopmat when N==1
    अगर (n_rows == 1 && (path == FA_COOPMAT1 || path == FA_COOPMAT2)) {
        path = FA_SCALAR;
    }

    चयन (path) {
    हाल FA_SCALAR:
        वापस get_fa_tuning_params_scalar(device, hsk, hsv, n_rows, n_kv, k_type, v_type, f32acc);
    हाल FA_COOPMAT1:
        वापस get_fa_tuning_params_coopmat1(device, hsk, hsv, n_rows, n_kv, k_type, v_type, f32acc);
    हाल FA_COOPMAT2:
        वापस get_fa_tuning_params_coopmat2(device, hsk, hsv, n_rows, n_kv, k_type, v_type, f32acc);
    शेष:
        फेंको std::runtime_error("unsupported FaCodePath");
    }
}

vk_fa_pipeline_state get_fa_pipeline_state(स्थिर vk_device& device, स्थिर vk_fa_tuning_params& params, uint32_t hsk, uint32_t hsv, bool aligned, bool f32acc,
                                                  bool use_mask, bool use_mask_opt, bool use_logit_softcap, bool use_sparse, ggml_type k_type, ggml_type v_type) {
    स्थिर bool old_amd_windows = device->vendor_id == VK_VENDOR_ID_AMD && device->driver_id == vk::DriverId::eAmdProprietary &&
                                 (device->architecture == AMD_GCN || device->architecture == AMD_RDNA1 || device->architecture == AMD_RDNA2);

    uint32_t flags = (use_mask_opt      ? 1 : 0) |
                     (use_mask          ? 2 : 0) |
                     (use_logit_softcap ? 4 : 0) |
                     (old_amd_windows   ? 8 : 0) |
                     (use_sparse        ? 16 : 0);

    स्थिर uint32_t subgroup_आकार = params.disable_subgroups ? 0 : params.subgroup_आकार;

    वापस vk_fa_pipeline_state{hsk, hsv, params.block_rows, params.block_cols, params.d_split, params.row_split, params.shmem_staging, params.path, params.workgroup_आकार, subgroup_आकार, aligned, f32acc, flags, params.limit_occupancy_shmem, k_type, v_type};
}

अटल uint32_t fa_block_bytes(ggml_type t) {
    अगर (t == GGML_TYPE_F32) {
        वापस 16u;
    }
    वापस (uint32_t) ggml_type_आकार(t);
}

अटल std::vector<uint32_t> get_fa_spec_constants(स्थिर vk_fa_pipeline_state& state) {
    वापस {
        /* 0 WorkGroupSize   */ state.workgroup_आकार,
        /* 1 Br              */ state.Br,
        /* 2 Bc              */ state.Bc,
        /* 3 HSK             */ state.HSK,
        /* 4 HSV             */ state.HSV,
        /* 5 Clamp           */ static_cast<uint32_t>(!state.aligned),
        /* 6 D_split         */ state.D_split,
        /* 7 row_split       */ state.row_split,
        /* 8 SubGroupSize    */ state.subgroup_आकार,
        /* 9 SHMEM_STAGING   */ state.shmem_staging ? 1u : 0u,
        /*10 Flags           */ state.flags,
        /*11 LIMIT_OCCUPANCY_SHMEM */ state.limit_occupancy_shmem,
        /*12 FaTypeK         */ static_cast<uint32_t>(state.k_type),
        /*13 FaTypeV         */ static_cast<uint32_t>(state.v_type),
        /*14 FaBlockBytesK   */ fa_block_bytes(state.k_type),
        /*15 FaBlockBytesV   */ fa_block_bytes(state.v_type),
    };
}

अटल bool ggml_vk_matmul_shmem_support(स्थिर vk_device& device, स्थिर std::vector<uint32_t>& warptile, bool mul_mat_id, ggml_type src0_type) {

    uint32_t lut_आकार = 0;
    चयन (src0_type) {
    हाल GGML_TYPE_IQ1_S:
    हाल GGML_TYPE_IQ1_M:
        // Regular matmul uses the compact uint16_t IQ1 grid; the expanded
        // uint32_t grid is only enabled for the q8_1/int-dot vector path.
        lut_आकार = 2*2048;
        अवरोध;
    हाल GGML_TYPE_IQ2_XXS:
        lut_आकार = 8*256;
        अवरोध;
    हाल GGML_TYPE_IQ2_XS:
        lut_आकार = 8*512;
        अवरोध;
    हाल GGML_TYPE_IQ2_S:
        lut_आकार = 8*1024;
        अवरोध;
    हाल GGML_TYPE_IQ3_XXS:
        lut_आकार = 4*256;
        अवरोध;
    हाल GGML_TYPE_IQ3_S:
        lut_आकार = 4*512;
        अवरोध;
    हाल GGML_TYPE_IQ4_NL:
    हाल GGML_TYPE_IQ4_XS:
    हाल GGML_TYPE_MXFP4:
        lut_आकार = 4*16;
        अवरोध;
    हाल GGML_TYPE_NVFP4:
        // Same kvalues budget as MXFP4 plus ue4m3_fp32_lut[128] (types.glsl, DATA_A_NVFP4).
        lut_आकार = 4*16 + 128u * (uint32_t)माप(भग्न);
        अवरोध;
    शेष:
        अवरोध;
    }

    // Needs to be kept up to date on shader changes
    // Needs to stay aligned with ggml_vk_mul_mm_spec.
    स्थिर bool intel_shmem_stride_pad_zero = device->vendor_id == VK_VENDOR_ID_INTEL && device->coopmat_support &&
                                              device->driver_id == vk::DriverId::eIntelProprietaryWindows;
    स्थिर uint32_t bank_conflict_offset = intel_shmem_stride_pad_zero ? 0 : (device->coopmat_support ? 8 : 1);
    स्थिर uint32_t type_आकार = device->fp16 ? माप(ggml_fp16_t) : माप(भग्न);
    स्थिर uint32_t warps = warptile[0] / warptile[10];

    स्थिर uint32_t load_bufs = (warptile[1] + warptile[2]) * (warptile[3] + bank_conflict_offset) * type_आकार;
    स्थिर uint32_t mmid_row_ids = mul_mat_id ? (warptile[2] * 2 * माप(uint16_t)) : 0;
    स्थिर uint32_t coopmat_stage = device->coopmat_support ? warptile[7] * warptile[8] / warps * माप(भग्न) : 0;

    स्थिर uint32_t योग_आकार = load_bufs + mmid_row_ids + coopmat_stage + lut_आकार;
    स्थिर bool supported = योग_आकार <= device->properties.limits.max_Compute_Shared_Memory_आकार;

    VK_LOG_DEBUG("ggml_vk_matmul_shmem_support(warptile=(" << warptile[0] << "," << warptile[1] << "," << warptile[2] << "), "
                 "mul_mat_id=" << mul_mat_id << ", src0_type=" << ggml_type_name(src0_type) << ", supported=" << supported);

    वापस supported;
}

अटल bool ggml_vk_matmul_int_shmem_support(स्थिर vk_device& device, स्थिर std::vector<uint32_t>& warptile, bool mul_mat_id, ggml_type src0_type) {

    // FLOAT_TYPE in the shader is float16_t with fp16 support, otherwise float.
    स्थिर uint32_t fp_आकार   = device->fp16 ? 2u : 4u;
    स्थिर uint32_t fp_align  = fp_आकार;
    स्थिर uint32_t fp2_आकार  = 2u * fp_आकार;
    स्थिर uint32_t fp2_align = device->fp16 ? 4u : 8u;

    काष्ठा member { uint32_t size, align; };
    स्वतः std430_आकार = [](std::initializer_list<member> members) {
        uint32_t off = 0, struct_align = 1;
        क्रम (स्थिर स्वतः &m : members) {
            off = (off + m.align - 1) & ~(m.align - 1);
            off += m.size;
            struct_align = std::max(struct_align, m.align);
        }
        वापस (off + struct_align - 1) & ~(struct_align - 1);
    };

    uint32_t block_a_आकार = 0;
    चयन (src0_type) {
        हाल GGML_TYPE_Q2_0:    block_a_आकार = std430_आकार({{32, 4}, {fp_आकार,  fp_align}});                  अवरोध; // qs[8] + dm
        हाल GGML_TYPE_Q4_0:    block_a_आकार = std430_आकार({{16, 4}, {fp_आकार,  fp_align}});                  अवरोध; // qs[16/4] + dm
        हाल GGML_TYPE_Q4_1:    block_a_आकार = std430_आकार({{16, 4}, {fp2_आकार, fp2_align}});                 अवरोध; // qs[16/4] + dm(vec2)
        हाल GGML_TYPE_Q5_0:    block_a_आकार = std430_आकार({{16, 4}, {4, 4}, {fp_आकार,  fp_align}});          अवरोध; // qs[16/4] + qh + dm
        हाल GGML_TYPE_Q5_1:    block_a_आकार = std430_आकार({{16, 4}, {4, 4}, {fp2_आकार, fp2_align}});         अवरोध; // qs[16/4] + qh + dm(vec2)
        हाल GGML_TYPE_Q8_0:    block_a_आकार = std430_आकार({{32, 4}, {fp_आकार,  fp_align}});                  अवरोध; // qs[8] + dm
        हाल GGML_TYPE_IQ4_XS:  block_a_आकार = std430_आकार({{32, 4}, {fp_आकार,  fp_align}});                  अवरोध; // qs[8] + d
        हाल GGML_TYPE_MXFP4:   block_a_आकार = std430_आकार({{32, 4}, {fp_आकार,  fp_align}});                  अवरोध; // qs[8] + d
        हाल GGML_TYPE_IQ4_NL:  block_a_आकार = std430_आकार({{32, 4}, {fp_आकार,  fp_align}});                  अवरोध; // qs[8] + d
        हाल GGML_TYPE_NVFP4:   block_a_आकार = std430_आकार({{32, 4}, {fp2_आकार, fp2_align}});                 अवरोध; // qs[8] + d_scales(vec2)
        हाल GGML_TYPE_Q2_K:    block_a_आकार = std430_आकार({{ 8, 4}, {2, 2}, {fp2_आकार, fp2_align}});         अवरोध; // qs[2] + scales(u8vec2) + dm(vec2)
        हाल GGML_TYPE_Q3_K:    block_a_आकार = std430_आकार({{16, 4}, {fp2_आकार, fp2_align}});                 अवरोध; // qs[4] + d_scales(vec2)
        हाल GGML_TYPE_Q4_K:    block_a_आकार = std430_आकार({{16, 4}, {fp2_आकार, fp2_align}});                 अवरोध; // qs[4] + dm(vec2)
        हाल GGML_TYPE_Q5_K:    block_a_आकार = std430_आकार({{32, 4}, {fp2_आकार, fp2_align}});                 अवरोध; // qs[8] + dm(vec2)
        हाल GGML_TYPE_Q6_K:    block_a_आकार = std430_आकार({{32, 4}, {fp2_आकार, fp2_align}});                 अवरोध; // qs[8] + d_scales(vec2)
        हाल GGML_TYPE_IQ3_S:   block_a_आकार = std430_आकार({{32, 4}, {fp_आकार,  fp_align}});                  अवरोध; // qs[8] + d
        शेष:
            वापस false;
    }

    // IQ3_S also copies its 512-entry grid into shared memory (types.glsl, init_iq_shmem)
    स्थिर uint32_t lut_आकार = (src0_type == GGML_TYPE_IQ3_S) ? 4*512 : 0;

    // block_b_cache: { int32_t qs[8]; FLOAT_TYPEV2 ds; }
    स्थिर uint32_t block_b_आकार = std430_आकार({{32, 4}, {fp2_आकार, fp2_align}});

    स्थिर uint32_t BM = warptile[1];
    स्थिर uint32_t BN = warptile[2];
    // mul_mmq.comp: BK_STEP=1 for MUL_MAT_ID, 4 otherwise.
    स्थिर uint32_t BK_STEP = mul_mat_id ? 1u : 4u;

    स्थिर uint32_t buf_a_आकार = BM * BK_STEP * block_a_आकार;
    स्थिर uint32_t buf_b_आकार = BN * BK_STEP * block_b_आकार;
    स्थिर uint32_t mmid_row_ids = mul_mat_id ? (BN * 2u * (uint32_t)माप(uint16_t)) : 0u;

    स्थिर uint32_t योग_आकार = buf_a_आकार + buf_b_आकार + mmid_row_ids + lut_आकार;
    स्थिर bool supported = योग_आकार <= device->properties.limits.max_Compute_Shared_Memory_आकार;

    VK_LOG_DEBUG("ggml_vk_matmul_int_shmem_support(warptile=(" << warptile[0] << "," << warptile[1] << "," << warptile[2] << "), "
                 "mul_mat_id=" << mul_mat_id << ", src0_type=" << ggml_type_name(src0_type) << ", total=" << योग_आकार << ", supported=" << supported);

    वापस supported;
}

अटल bool ggml_vk_matmul_cm1_int_shmem_support(स्थिर vk_device& device, स्थिर std::vector<uint32_t>& warptile, bool mul_mat_id, ggml_type src0_type) {

    bool kscales2 = false;    // two scale sets per block
    bool has_dm   = false;    // d+m as vec2 + b-side sum
    bool has_kvalues = false;
    चयन (src0_type) {
        हाल GGML_TYPE_Q4_0: हाल GGML_TYPE_Q5_0: हाल GGML_TYPE_Q8_0:
            अवरोध;
        हाल GGML_TYPE_Q4_1: हाल GGML_TYPE_Q5_1:
        हाल GGML_TYPE_Q4_K: हाल GGML_TYPE_Q5_K:
            has_dm = true;                          अवरोध;
        हाल GGML_TYPE_IQ4_NL: हाल GGML_TYPE_IQ4_XS: हाल GGML_TYPE_MXFP4:
            has_kvalues = true;                     अवरोध;
        हाल GGML_TYPE_Q3_K: हाल GGML_TYPE_Q6_K:
            kscales2 = true;                        अवरोध;
        हाल GGML_TYPE_NVFP4:
            kscales2 = true; has_kvalues = true;    अवरोध;
        शेष:
            वापस false;
    }

    स्थिर uint32_t BM         = warptile[1];
    स्थिर uint32_t BN         = warptile[2];

    स्थिर uint32_t BK      = 32;
    स्थिर uint32_t BK_STEP = mul_mat_id ? 2u : 4u;
    स्थिर uint32_t QPITCH  = BK_STEP * (BK / 4u) + 4u;
    स्थिर uint32_t KSCALES = kscales2 ? 2u : 1u;

    uint32_t योग = 0;
    योग += BM * QPITCH * (uint32_t)माप(uint32_t);   // buf_a_qs
    योग += BN * QPITCH * (uint32_t)माप(uint32_t);   // buf_b_qs
    योग += has_dm ? (BM * BK_STEP * 2u * (uint32_t)माप(भग्न))   // buf_a_dm (vec2)
                    : (BM * BK_STEP * KSCALES * (uint32_t)माप(भग्न)); // buf_a_d
    योग += BN * BK_STEP * (uint32_t)माप(भग्न);     // buf_b_d
    अगर (has_dm) {
        योग += BN * BK_STEP * (uint32_t)माप(भग्न); // buf_b_s
    }
    अगर (has_kvalues) {
        योग += 16u * (uint32_t)माप(int8_t);         // cm1_kvalues[16]
    }
    अगर (src0_type == GGML_TYPE_NVFP4 && !device->ocp_fp4) {
        योग += 128u * (uint32_t)माप(भग्न);         // ue4m3_fp32_lut[128]
    }
    अगर (mul_mat_id) {
        योग += BN * 2u * (uint32_t)माप(uint16_t);   // row_ids[BN] (u16vec2)
    }

    स्थिर bool supported = योग <= device->properties.limits.max_Compute_Shared_Memory_आकार;

    VK_LOG_DEBUG("ggml_vk_matmul_cm1_int_shmem_support(warptile=(" << warptile[0] << "," << warptile[1] << "," << warptile[2] << "), "
                 "mul_mat_id=" << mul_mat_id << ", src0_type=" << ggml_type_name(src0_type) << ", total=" << योग << ", supported=" << supported);

    वापस supported;
}

अटल स्थिर std::unordered_map<std::string, uint32_t> rdna1_pipelines = {
    {"soft_max", 64}, {"im2col", 64},
    {"argmax", 64}, {"mul_mat_vec", 64},
    {"mul_mat_vec_f16", 32}, {"mul_mat_vec_f32_f16", 32}
};

अटल स्थिर std::unordered_map<std::string, uint32_t> rdna2_pipelines = {
    {"soft_max", 64}, {"im2col", 64},
};

अटल std::vector<GpuPipelineConfig> gpu_pipeline_configs = {
    {
        vk_device_architecture::AMD_RDNA1,
        {
            rdna1_pipelines,
        },
        RDNA_DEFAULT_SUBGROUP_आकार
    },
    {
        vk_device_architecture::AMD_RDNA2,
        {
            rdna2_pipelines,
        },
        RDNA_DEFAULT_SUBGROUP_आकार
    },
};

uint32_t get_subgroup_आकार(स्थिर std::string &pipeline_name, स्थिर vk_device_architecture &arch) {
    क्रम (स्थिर स्वतः &config : gpu_pipeline_configs) {
        अगर (config.arch == arch) {
            स्वतः pipIt = config.pipelines.find(pipeline_name);
            अगर (pipIt != config.pipelines.end()) {
                वापस pipIt->second;
            }
            std::vector<std::pair<std::string, uint32_t>> sorted_pipelines(config.pipelines.begin(), config.pipelines.end());
            std::sort(sorted_pipelines.begin(), sorted_pipelines.end(),
                      [](स्थिर स्वतः &a, स्थिर स्वतः &b) { वापस a.first.size() > b.first.size(); });
            क्रम (स्थिर स्वतः &entry : sorted_pipelines) {
                अगर (pipeline_name.find(entry.first) != std::string::npos) {
                    वापस entry.second;
                }
            }
            वापस config.default_subgroup_आकार;
        }
    }
    वापस 0; // If no matching configuration is found
}

अटल bool ggml_vk_fa_type_needs_shmem(ggml_type type) {
    चयन (type) {
    हाल GGML_TYPE_IQ4_NL:
        वापस true;
    शेष:
        वापस false;
    }
}

अटल bool ggml_vk_fa_scalar_uses_mmq(स्थिर vk_device& device, ggml_type k_type, ggml_type v_type) {
#अगर defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
    वापस device->integer_dot_product && device->subgroup_clustered &&
           !ggml_vk_fa_type_needs_shmem(v_type) &&
           (k_type == GGML_TYPE_Q4_0 || k_type == GGML_TYPE_Q4_1 ||
            k_type == GGML_TYPE_Q5_0 || k_type == GGML_TYPE_Q5_1 ||
            k_type == GGML_TYPE_Q8_0);
#अन्यथा
    GGML_UNUSED(device);
    GGML_UNUSED(k_type);
    GGML_UNUSED(v_type);
    वापस false;
#पूर्ण_अगर
}

व्योम ggml_vk_load_shaders(vk_device& device, vk_pipeline requested) {
    VK_LOG_DEBUG("ggml_vk_load_shaders(" << device->name << ")");

    // some shaders have a minimum subgroup size
    स्थिर uint32_t subgroup_आकार_8 = std::max(device->subgroup_आकार, 8u);
    स्थिर uint32_t subgroup_आकार_16 = std::max(device->subgroup_आकार, 16u);
    स्थिर uint32_t subgroup_आकार_32 = std::max(device->subgroup_आकार, 32u);

    // clamp WARP for l_/m_ warptiles so WM <= BM (breaks on subgroupSize > 64)
    स्थिर uint32_t mm_warp_8  = std::min(subgroup_आकार_8,  64u);
    स्थिर uint32_t mm_warp_16 = std::min(subgroup_आकार_16, 64u);

    स्थिर uint32_t mul_mat_subgroup_आकार = (device->vendor_id == VK_VENDOR_ID_INTEL && device->subgroup_आकार_control) ? device->subgroup_min_आकार : device->subgroup_आकार;
    स्थिर uint32_t mul_mat_subgroup_आकार_8 = std::max(mul_mat_subgroup_आकार, 8u);
    स्थिर uint32_t mul_mat_subgroup_आकार_16 = std::max(mul_mat_subgroup_आकार, 16u);
    स्थिर uint32_t mul_mat_subgroup_आकार_32 = std::max(mul_mat_subgroup_आकार, 32u);
    स्थिर uint32_t mul_mat_mm_warp_8  = std::min(mul_mat_subgroup_आकार_8,  64u);
    स्थिर uint32_t mul_mat_mm_warp_16 = std::min(mul_mat_subgroup_आकार_16, 64u);

    स्थिर bool subgroup_min_आकार_16 = (!device->subgroup_आकार_control && device->subgroup_आकार >= 16) ||
                                      (device->subgroup_आकार_control && device->subgroup_max_आकार >= 16);

    // mulmat
    // Warptile layout (indices match mul_mm.comp constantIDs):
    //   [0..9]  : BLOCK_SIZE, BM, BN, BK, WM, WN, WMITER, TM, TN, TK
    //   [10]    : WARP / required_subgroup_size (read via WARP_SIZE_IDX)
    अटल constexpr माप_प्रकार WARP_आकार_IDX = 10;
    std::vector<uint32_t> l_warptile, m_warptile, s_warptile,
                          l_warptile_id, m_warptile_id, s_warptile_id,
                          l_warptile_mmq, m_warptile_mmq, s_warptile_mmq,
                          l_warptile_mmq_int, m_warptile_mmq_int, s_warptile_mmq_int,
                          l_warptile_mmq_cm1_int, m_warptile_mmq_cm1_int, s_warptile_mmq_cm1_int,
                          l_warptile_mmq_cm1_int_k, m_warptile_mmq_cm1_int_k, s_warptile_mmq_cm1_int_k,
                          l_warptile_mmq_int_k, m_warptile_mmq_int_k, s_warptile_mmq_int_k,
                          l_warptile_mmq_k, m_warptile_mmq_k, s_warptile_mmq_k,
                          l_warptile_mmqid, m_warptile_mmqid, s_warptile_mmqid,
                          l_warptile_mmqid_int, m_warptile_mmqid_int, s_warptile_mmqid_int,
                          l_warptile_mmqid_int_k, m_warptile_mmqid_int_k, s_warptile_mmqid_int_k;
    std::array<uint32_t, 3> l_wg_denoms, m_wg_denoms, s_wg_denoms,
                            l_mmq_wg_denoms, m_mmq_wg_denoms, s_mmq_wg_denoms,
                            l_mmq_wg_denoms_k, m_mmq_wg_denoms_k, s_mmq_wg_denoms_k,
                            l_mmq_cm1_wg_denoms_k, m_mmq_cm1_wg_denoms_k, s_mmq_cm1_wg_denoms_k,
                            l_mmqid_wg_denoms, m_mmqid_wg_denoms, s_mmqid_wg_denoms;

    uint32_t l_align, m_align, s_align;

    // RDNA3.5 preferred wave32 here
    स्थिर bool cm1_use_wave32 = device->vendor_id == VK_VENDOR_ID_AMD &&
                                device->subgroup_आकार_control &&
                                device->subgroup_min_आकार <= 32 && device->subgroup_max_आकार >= 32;
    स्थिर uint32_t cm1_sg = cm1_use_wave32 ? 32 : device->subgroup_आकार;

    vk_pipeline wait_pipeline;
    CompileTask claimed_task {};
    bool has_claimed_task = false;

    // The rest of the walk reads and writes shared device state, so hold the
    // lock until we're done deciding what to compile.
    std::unique_lock<std::mutex> compile_lock(device->compile_mutex);

    अगर (device->coopmat2) {
        // spec constants and tile sizes for non-quant matmul/matmul_id
        l_warptile = { 256, 128, 256, 64, 1 };
        m_warptile = { 256, 128, 128, 64, 0 };
        s_warptile = { 128,  64,  64, 64, 0 };
        l_wg_denoms = {128, 256, 1 };
        m_wg_denoms = {128, 128, 1 };
        s_wg_denoms = { 64,  64, 1 };

        // spec constants and tile sizes for quant matmul (non-Qi_K)
        l_warptile_mmq = { 256, 128, 256, 64, 1 };
        m_warptile_mmq = { 256, 128, 128, 64, 1 };
        s_warptile_mmq = { 256, 32,  64, 128, 0 };
        l_mmq_wg_denoms = { 128, 256, 1 };
        m_mmq_wg_denoms = { 128, 128, 1 };
        s_mmq_wg_denoms = { 32,  64,  1 };

        // spec constants and tile sizes for quant matmul (Qi_K)
        l_warptile_mmq_k = { 256, 128, 256, 64, 1 };
        m_warptile_mmq_k = { 256, 128, 128, 64, 1 };
        s_warptile_mmq_k = { 256, 32,  64, 128, 0 };
        l_mmq_wg_denoms_k = { 128, 256, 1 };
        m_mmq_wg_denoms_k = { 128, 128, 1 };
        s_mmq_wg_denoms_k = { 32,  64,  1 };

        // spec constants and tile sizes for quant matmul_id
        स्थिर uint32_t mmqid_bk = device->coopmat2_decode_vector ? 64u : 32u;
        l_warptile_mmqid = { 256, 128, 128, mmqid_bk, 1 };
        m_warptile_mmqid = { 256, 128, 64,  mmqid_bk, 0 };
        s_warptile_mmqid = { 256, 128, 64,  mmqid_bk, 0 };
        l_mmqid_wg_denoms = { 128, 128, 1 };
        m_mmqid_wg_denoms = { 128, 64, 1 };
        s_mmqid_wg_denoms = { 128, 64, 1 };

        l_align = 128;
        m_align =  64;
        s_align =  32;
    } अन्यथा {
        // Matrix cores require different warp group sizes
        स्थिर uint32_t tm_l = device->coopmat_support ? device->coopmat_m : 4;
        स्थिर uint32_t tm_m = device->coopmat_support ? device->coopmat_m : 4;
        स्थिर uint32_t tm_s = device->coopmat_support ? device->coopmat_m : 2;
        स्थिर uint32_t tn_l = device->coopmat_support ? device->coopmat_n : 4;
        स्थिर uint32_t tn_m = device->coopmat_support ? device->coopmat_n : 2;
        स्थिर uint32_t tn_s = device->coopmat_support ? device->coopmat_n : 2;
        स्थिर uint32_t tk_l = device->coopmat_support ? device->coopmat_k : 1;
        स्थिर uint32_t tk_m = device->coopmat_support ? device->coopmat_k : 1;
        स्थिर uint32_t tk_s = device->coopmat_support ? device->coopmat_k : 1;

        स्थिर uint32_t itm = device->coopmat_int_m;
        स्थिर uint32_t itn = device->coopmat_int_n;
        स्थिर uint32_t itk = device->coopmat_int_k;

        स्थिर uint32_t s_warptile_wm = device->subgroup_आकार == 8 ? 8 : 32;

        l_warptile = { 128,             128, 128, 16, mm_warp_8 * 2, 64, 2, tm_l, tn_l, tk_l, mm_warp_8 };
        m_warptile = { 128,              64,  64, 16, mm_warp_8,     32, 2, tm_m, tn_m, tk_m, mm_warp_8 };
        s_warptile = { subgroup_आकार_32, 32,  32, 16, s_warptile_wm, 32, 2, tm_s, tn_s, tk_s, subgroup_आकार_8 };

        l_warptile_mmq = { 128,             128, 128, 32, mm_warp_8 * 2, 64, 2, tm_l, tn_l, tk_l, mm_warp_8 };
        m_warptile_mmq = { 128,              64,  64, 32, mm_warp_8,     32, 2, tm_m, tn_m, tk_m, mm_warp_8 };
        s_warptile_mmq = { subgroup_आकार_32, 32,  32, 32, s_warptile_wm, 32, 2, tm_s, tn_s, tk_s, subgroup_आकार_8 };

        // Integer MMQ has a smaller shared memory profile, but heavier register use
        l_warptile_mmq_int = { 128,             128, 128, 32, mm_warp_8 * 2, 64, 2, 4, 4, 1, mm_warp_8 };
        m_warptile_mmq_int = { 128,              64,  64, 32, mm_warp_8,     32, 2, 2, 2, 1, mm_warp_8 };
        s_warptile_mmq_int = { subgroup_आकार_32, 32,  32, 32, s_warptile_wm, 32, 2, 2, 1, 1, subgroup_आकार_8 };

        स्थिर स्वतः cm1_bs = [cm1_sg](uint32_t bm, uint32_t bn) {
            वापस cm1_sg * (bm / std::min(cm1_sg, bm)) * (bn / 32);
        };

        l_warptile_mmq_cm1_int = { cm1_bs(128, 128), 128, 128, 32, std::min(cm1_sg, 128u), 32, 2, itm, itn, itk, cm1_sg, (uint32_t)device->architecture };
        m_warptile_mmq_cm1_int = { cm1_bs( 64,  64),  64,  64, 32, std::min(cm1_sg,  64u), 32, 2, itm, itn, itk, cm1_sg, (uint32_t)device->architecture };
        s_warptile_mmq_cm1_int = { cm1_bs( 32,  32),  32,  32, 32, std::min(cm1_sg,  32u), 32, 2, itm, itn, itk, cm1_sg, (uint32_t)device->architecture };

        l_warptile_mmq_cm1_int_k = { cm1_bs( 64, 128),  64, 128, 32, std::min(cm1_sg,  64u), 32, 2, itm, itn, itk, cm1_sg, (uint32_t)device->architecture };
        m_warptile_mmq_cm1_int_k = { cm1_bs( 64,  64),  64,  64, 32, std::min(cm1_sg,  64u), 32, 2, itm, itn, itk, cm1_sg, (uint32_t)device->architecture };
        s_warptile_mmq_cm1_int_k = { cm1_bs( 32,  32),  32,  32, 32, std::min(cm1_sg,  32u), 32, 2, itm, itn, itk, cm1_sg, (uint32_t)device->architecture };

        l_mmq_cm1_wg_denoms_k = { l_warptile_mmq_cm1_int_k[1], l_warptile_mmq_cm1_int_k[2], 1 };
        m_mmq_cm1_wg_denoms_k = { m_warptile_mmq_cm1_int_k[1], m_warptile_mmq_cm1_int_k[2], 1 };
        s_mmq_cm1_wg_denoms_k = { s_warptile_mmq_cm1_int_k[1], s_warptile_mmq_cm1_int_k[2], 1 };

        // K-quants use even more registers, mitigate by setting WMITER to 1
        l_warptile_mmq_int_k = { 128,               128, 128, 32, mm_warp_8 * 2, 64, 1, 4, 4, 1, mm_warp_8 };
        m_warptile_mmq_int_k = { 128,                64,  64, 32, mm_warp_8,     32, 1, 2, 2, 1, mm_warp_8 };
        s_warptile_mmq_int_k = { subgroup_आकार_32,   32,  32, 32, s_warptile_wm, 32, 1, 2, 1, 1, subgroup_आकार_8 };

        l_warptile_id = { 128,                      128, 128, 16, mul_mat_mm_warp_16 * 2, 64, 2, tm_l, tn_l, tk_l, mul_mat_mm_warp_16 };
        m_warptile_id = { 128,                       64,  64, 16, mul_mat_mm_warp_16,     32, 2, tm_m, tn_m, tk_m, mul_mat_mm_warp_16 };
        s_warptile_id = { mul_mat_subgroup_आकार_16,  32,  32, 16, s_warptile_wm,          32, 2, tm_s, tn_s, tk_s, mul_mat_subgroup_आकार_16 };

        l_warptile_mmqid = { 128,                       128, 128, 32, mul_mat_mm_warp_8 * 2, 64, 2, tm_l, tn_l, tk_l, mul_mat_mm_warp_8 };
        m_warptile_mmqid = { 128,                        64,  64, 32, mul_mat_mm_warp_8,     32, 2, tm_m, tn_m, tk_m, mul_mat_mm_warp_8 };
        s_warptile_mmqid = { mul_mat_subgroup_आकार_32,   32,  32, 32, s_warptile_wm,         32, 2, tm_s, tn_s, tk_s, mul_mat_subgroup_आकार_8 };

        l_warptile_mmqid_int = { 128,                       128, 128, 32, mul_mat_mm_warp_8 * 2, 64, 2, 4, 4, 1, mul_mat_mm_warp_8 };
        m_warptile_mmqid_int = { 128,                        64,  64, 32, mul_mat_mm_warp_8,     32, 2, 2, 2, 1, mul_mat_mm_warp_8 };
        s_warptile_mmqid_int = { mul_mat_subgroup_आकार_32,   32,  32, 32, s_warptile_wm,         32, 2, 2, 1, 1, mul_mat_subgroup_आकार_8 };

        l_warptile_mmqid_int_k = { 128,                     128, 128, 32, mul_mat_mm_warp_16 * 2, 64, 1, 4, 4, 1, mul_mat_mm_warp_16 };
        m_warptile_mmqid_int_k = { 128,                      64,  64, 32, mul_mat_mm_warp_16,     32, 1, 2, 2, 1, mul_mat_mm_warp_16 };
        s_warptile_mmqid_int_k = { mul_mat_subgroup_आकार_32, 32,  32, 32, s_warptile_wm,          32, 1, 2, 1, 1, mul_mat_subgroup_आकार_16 };

        // chip specific tuning
        अगर ((device->architecture == AMD_GCN) && (device->driver_id != vk::DriverId::eAmdProprietary)) {
            m_warptile_mmq = m_warptile_mmq_int = { 256, 64, 64, 32, 16, 16, 2, 2, 2, 1, 16 };
            m_warptile_mmqid = m_warptile_mmqid_int = { 256, 64, 64, 32, 16, 16, 2, 2, 2, 1, 16 };
        } अन्यथा अगर (device->vendor_id == VK_VENDOR_ID_AMD && device->coopmat_support && device->driver_id != vk::DriverId::eAmdProprietary) {
            // This is intentionally using tx_m values, slight performance increase
            l_warptile = { 256, 128, 128, 16, mm_warp_8, 64, 2, tm_m, tn_m, tk_m, mm_warp_8 };
            l_warptile_mmq = l_warptile_mmq_int = { 256, 128, 128, 32, mm_warp_8, 64, 2, tm_m, tn_m, tk_m, mm_warp_8 };
            l_warptile_mmq_int_k = { 256, 128, 128, 32, mm_warp_16, 64, 1, 4, 2, 1, mm_warp_16 };
        } अन्यथा अगर (device->vendor_id == VK_VENDOR_ID_QUALCOMM && device->coopmat_support) {
            m_warptile     = { 64, 64, 64, 16, 64, 64, 1, tm_l, tn_l, tk_l, 64 };
            m_warptile_mmq = { 64, 64, 64, 32, 64, 64, 1, tm_m, tn_m, tk_m, 64 };
        }

        l_mmq_wg_denoms = l_wg_denoms = {128, 128, 1 };
        m_mmq_wg_denoms = m_wg_denoms = { 64,  64, 1 };
        s_mmq_wg_denoms = s_wg_denoms = { 32,  32, 1 };
        l_align = 128;
        m_align =  64;
        s_align =  32;

        अगर (device->vendor_id == VK_VENDOR_ID_INTEL && device->coopmat_support) {
            // Xe1/Xe2/Xe3 with coopmat enabled - warptile performance tuning
            l_warptile = { 512, 128, 128, 16, mm_warp_8, 32, 2, tm_l, tn_l, tk_l, mm_warp_8 };
            अगर (device->architecture == INTEL_XE1) {
                l_warptile_mmq  = { 512, 256, 128, 32, 32, 32, 2, tm_l, tn_l, tk_l, 16 };
                l_mmq_wg_denoms = { 256, 128, 1 };
                l_align         = 32;  //set as BK
            } अन्यथा {
                l_warptile_mmq  = { 512, 128, 256, 32, 32, 32, 2, tm_l, tn_l, tk_l, 16 };
                l_mmq_wg_denoms = { 128, 256, 1 };
                l_align         = 32;  //set as BK
            }
        }

        स्थिर bool use_cm1_int = device->coopmat_int_support &&
                                 (device->architecture == AMD_RDNA3 || device->architecture == AMD_RDNA4);

        क्रम (uint32_t i = 0; i < GGML_TYPE_गिनती; ++i) {
            ggml_type t = (ggml_type)i;
            // Disable medium and large matrix multiplication if not enough shared memory is available
            // Check mmq warptiles as the largest configuration
            // Throw an error if not enough for any matrix multiplication is available
            अगर (!ggml_vk_matmul_shmem_support(device, s_warptile_mmq, false, t)) {
                std::अ_त्रुटि << "ggml_vulkan: Error: Shared memory size too small for matrix multiplication." << std::प_पूर्ण;
                फेंको std::runtime_error("Shared memory size too small for matrix multiplication.");
            } अन्यथा अगर (!ggml_vk_matmul_shmem_support(device, m_warptile_mmq, false, t)) {
                device->mul_mat_m[i] = false;
                device->mul_mat_l[i] = false;
            } अन्यथा अगर (!ggml_vk_matmul_shmem_support(device, l_warptile_mmq, false, t)) {
                device->mul_mat_l[i] = false;
            }

            // Disable mul_mat_id if not enough shared memory is available
            अगर (!ggml_vk_matmul_shmem_support(device, s_warptile_mmqid, true, t)) {
                device->mul_mat_id_s[i] = false;
                device->mul_mat_id_m[i] = false;
                device->mul_mat_id_l[i] = false;
            } अन्यथा अगर (!ggml_vk_matmul_shmem_support(device, m_warptile_mmqid, true, t)) {
                device->mul_mat_id_m[i] = false;
                device->mul_mat_id_l[i] = false;
            } अन्यथा अगर (!ggml_vk_matmul_shmem_support(device, l_warptile_mmqid, true, t)) {
                device->mul_mat_id_l[i] = false;
            }

            // The q8_1 mmq path has its own (larger) shmem layout, check it separately.
            // K-quants and IQ3_S use the _int_k warptiles, others use _int.
            // cm1 splits k-tiles on the KSCALES==2 types and shares tiles between dense/id.
            स्थिर bool is_k_quant = (t == GGML_TYPE_Q2_K || t == GGML_TYPE_Q3_K ||
                                     t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q5_K ||
                                     t == GGML_TYPE_Q6_K || t == GGML_TYPE_IQ3_S);
            स्थिर bool cm1_k_tile = (t == GGML_TYPE_Q3_K || t == GGML_TYPE_Q6_K ||
                                     t == GGML_TYPE_NVFP4);

            स्थिर स्वतः & s_int   = use_cm1_int ? (cm1_k_tile ? s_warptile_mmq_cm1_int_k : s_warptile_mmq_cm1_int)
                                               : (is_k_quant  ? s_warptile_mmq_int_k     : s_warptile_mmq_int);
            स्थिर स्वतः & m_int   = use_cm1_int ? (cm1_k_tile ? m_warptile_mmq_cm1_int_k : m_warptile_mmq_cm1_int)
                                               : (is_k_quant  ? m_warptile_mmq_int_k     : m_warptile_mmq_int);
            स्थिर स्वतः & l_int   = use_cm1_int ? (cm1_k_tile ? l_warptile_mmq_cm1_int_k : l_warptile_mmq_cm1_int)
                                               : (is_k_quant  ? l_warptile_mmq_int_k     : l_warptile_mmq_int);
            स्थिर स्वतः & s_intid = use_cm1_int ? (cm1_k_tile ? s_warptile_mmq_cm1_int_k : s_warptile_mmq_cm1_int)
                                               : (is_k_quant  ? s_warptile_mmqid_int_k   : s_warptile_mmqid_int);
            स्थिर स्वतः & m_intid = use_cm1_int ? (cm1_k_tile ? m_warptile_mmq_cm1_int_k : m_warptile_mmq_cm1_int)
                                               : (is_k_quant  ? m_warptile_mmqid_int_k   : m_warptile_mmqid_int);
            स्थिर स्वतः & l_intid = use_cm1_int ? (cm1_k_tile ? l_warptile_mmq_cm1_int_k : l_warptile_mmq_cm1_int)
                                               : (is_k_quant  ? l_warptile_mmqid_int_k   : l_warptile_mmqid_int);

            स्थिर स्वतः int_shmem_support = [&](स्थिर std::vector<uint32_t>& wt, bool id) {
                वापस use_cm1_int ? ggml_vk_matmul_cm1_int_shmem_support(device, wt, id, t)
                                   : ggml_vk_matmul_int_shmem_support(device, wt, id, t);
            };

            अगर (!int_shmem_support(s_int, false)) {
                device->mul_mat_s_int[i] = false;
                device->mul_mat_m_int[i] = false;
                device->mul_mat_l_int[i] = false;
            } अन्यथा अगर (!int_shmem_support(m_int, false)) {
                device->mul_mat_m_int[i] = false;
                device->mul_mat_l_int[i] = false;
            } अन्यथा अगर (!int_shmem_support(l_int, false)) {
                device->mul_mat_l_int[i] = false;
            }

            अगर (!int_shmem_support(s_intid, true)) {
                device->mul_mat_id_s_int[i] = false;
                device->mul_mat_id_m_int[i] = false;
                device->mul_mat_id_l_int[i] = false;
            } अन्यथा अगर (!int_shmem_support(m_intid, true)) {
                device->mul_mat_id_m_int[i] = false;
                device->mul_mat_id_l_int[i] = false;
            } अन्यथा अगर (!int_shmem_support(l_intid, true)) {
                device->mul_mat_id_l_int[i] = false;
            }
        }
    }

    स्वतः स्थिर &ggml_vk_create_pipeline = [&](vk_device& device, vk_pipeline& base_pipeline, स्थिर अक्षर *name, माप_प्रकार spv_आकार, स्थिर व्योम* spv_तथ्य, स्थिर अक्षर *entrypoint,
                                              uint32_t parameter_गिनती, uint32_t धकेलो_constant_आकार, std::array<uint32_t, 3> wg_denoms, स्थिर std::vector<uint32_t>& specialization_constants,
                                              uint32_t align, bool disable_robustness = false, bool require_full_subgroups = false, uint32_t required_subgroup_आकार = 0) {

        अगर (!require_full_subgroups && required_subgroup_आकार == 0) {
            required_subgroup_आकार = get_subgroup_आकार(name, device->architecture);
        }

        vk_pipeline *ptr = &base_pipeline;

        पूर्णांक अंक_pipelines = 1;
#अगर defined(VK_EXT_shader_64bit_indexing)
        अगर (device->shader_64b_indexing) {
            अंक_pipelines = 2;
        }
#पूर्ण_अगर
        क्रम (पूर्णांक i = 0; i < अंक_pipelines; ++i, ptr = &(*ptr)->अग्र) {
            vk_pipeline &pipeline = *ptr;
            अगर (!pipeline) {
                pipeline = std::make_shared<vk_pipeline_struct>();
            }
            अगर (!pipeline->initialized) {
                pipeline->name = name;
                pipeline->parameter_गिनती = parameter_गिनती;
                pipeline->धकेलो_constant_आकार = धकेलो_constant_आकार;
                pipeline->wg_denoms = wg_denoms;
                pipeline->align = align;
                pipeline->initialized = true;
#अगर defined(VK_EXT_shader_64bit_indexing)
                pipeline->is_64b_indexing = (i == 1);
#पूर्ण_अगर
            }

            // We only care about the pipeline this call asked for; the rest
            // (including the 64-bit indexing variant) are handled by their
            // own request_descriptor_sets / load_shaders calls.
            अगर (pipeline.लो() != requested.लो()) {
                जारी;
            }

            अगर (pipeline->compiled) {
                जारी;
            }

            wait_pipeline = pipeline;

            अगर (!pipeline->compile_pending) {
                pipeline->compile_pending = true;
                claimed_task.pipeline = pipeline;
                claimed_task.spv_आकार = spv_आकार;
                claimed_task.spv_तथ्य = spv_तथ्य;
                claimed_task.entrypoint = entrypoint;
                claimed_task.parameter_गिनती = parameter_गिनती;
                claimed_task.wg_denoms = wg_denoms;
                claimed_task.specialization_constants = specialization_constants;
                claimed_task.disable_robustness = disable_robustness;
                claimed_task.require_full_subgroups = require_full_subgroups;
                claimed_task.required_subgroup_आकार = required_subgroup_आकार;
                has_claimed_task = true;
            }
        }
    };

    स्वतः स्थिर &ggml_vk_create_pipeline2 = [&](vk_device& device, vk_pipeline& pipeline, स्थिर std::string &name, माप_प्रकार spv_आकार, स्थिर व्योम* spv_तथ्य, स्थिर अक्षर *entrypoint,
                                              uint32_t parameter_गिनती, uint32_t धकेलो_constant_आकार, std::array<uint32_t, 3> wg_denoms, स्थिर std::vector<uint32_t>& specialization_constants,
                                              uint32_t align, bool disable_robustness = false, bool require_full_subgroups = false, uint32_t required_subgroup_आकार = 0) {
        वापस ggml_vk_create_pipeline(device, pipeline, name.c_str(), spv_आकार, spv_तथ्य, entrypoint,
                                       parameter_गिनती, धकेलो_constant_आकार, wg_denoms, specialization_constants,
                                       align, disable_robustness, require_full_subgroups, required_subgroup_आकार);
    };

    // FA scalar has two SPIR-V modules (MMQ vs non-MMQ); FA cm1 has one. K/V
    // quant type is selected at runtime via the FaTypeK / FaTypeV spec constants.

    क्रम (स्वतः &fa : device->pipeline_flash_attn_f32_f16) {
        अगर (fa.first.path != FA_SCALAR) जारी;
        स्थिर uint32_t Br = fa.first.Br;
        स्थिर uint32_t Bc = fa.first.Bc;
        स्थिर bool aligned = fa.first.aligned;
        स्थिर bool f32acc = fa.first.f32acc;
        स्थिर uint32_t fa_sgs = fa.first.subgroup_आकार;
        स्थिर bool fa_ds = fa.first.subgroup_आकार == 0;

        स्थिर bool bf16_kv = fa.first.k_type == GGML_TYPE_BF16;
        स्थिर bool use_mmq = ggml_vk_fa_scalar_uses_mmq(device, fa.first.k_type, fa.first.v_type);
        स्थिर व्योम * spv_तथ्य = nullptr;
        माप_प्रकार spv_आकार = 0;
        स्थिर अक्षर *name = nullptr;
        अगर (bf16_kv) {
            spv_तथ्य = flash_attn_f32_f16_fp32_तथ्य;
            spv_आकार = flash_attn_f32_f16_fp32_len;
            name = aligned ? "flash_attn_f32_bf16_aligned" : "flash_attn_f32_bf16";
        } अन्यथा अगर (use_mmq) {
#अगर defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
            अगर (device->fp16) {
                अगर (f32acc) { spv_तथ्य = flash_attn_f32_f16_int8_तथ्य;        spv_आकार = flash_attn_f32_f16_int8_len; }
                अन्यथा        { spv_तथ्य = flash_attn_f32_f16_f16acc_int8_तथ्य; spv_आकार = flash_attn_f32_f16_f16acc_int8_len; }
            } अन्यथा {
                spv_तथ्य = flash_attn_f32_f16_fp32_int8_तथ्य;
                spv_आकार = flash_attn_f32_f16_fp32_int8_len;
            }
#पूर्ण_अगर
            name = aligned ? "flash_attn_f32_f16_aligned" : "flash_attn_f32_f16";
        } अन्यथा {
            अगर (device->fp16) {
                अगर (device->dot2_f16) {
                    अगर (f32acc) { spv_तथ्य = flash_attn_f32_f16_dot2_तथ्य;        spv_आकार = flash_attn_f32_f16_dot2_len; }
                    अन्यथा        { spv_तथ्य = flash_attn_f32_f16_dot2_f16acc_तथ्य; spv_आकार = flash_attn_f32_f16_dot2_f16acc_len; }
                } अन्यथा {
                    अगर (f32acc) { spv_तथ्य = flash_attn_f32_f16_तथ्य;        spv_आकार = flash_attn_f32_f16_len; }
                    अन्यथा        { spv_तथ्य = flash_attn_f32_f16_f16acc_तथ्य; spv_आकार = flash_attn_f32_f16_f16acc_len; }
                }
            } अन्यथा {
                spv_तथ्य = flash_attn_f32_f16_fp32_तथ्य;
                spv_आकार = flash_attn_f32_f16_fp32_len;
            }
            name = aligned ? "flash_attn_f32_f16_aligned" : "flash_attn_f32_f16";
        }
        ggml_vk_create_pipeline(device, fa.second, name, spv_आकार, spv_तथ्य, "main", 8,
                                माप(vk_flash_attn_धकेलो_constants), {Br, 1, 1},
                                get_fa_spec_constants(fa.first), aligned ? Bc : 1, true,
                                !fa_ds, !fa_ds ? fa_sgs : 0);
    }

#अगर defined(VK_KHR_cooperative_matrix) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    अगर (device->coopmat1_fa_support) {
        क्रम (स्वतः &fa : device->pipeline_flash_attn_f32_f16) {
            अगर (fa.first.path != FA_COOPMAT1) जारी;
            स्थिर uint32_t Br = fa.first.Br;
            स्थिर uint32_t Bc = fa.first.Bc;
            स्थिर bool aligned = fa.first.aligned;
            स्थिर bool f32acc = fa.first.f32acc;
            स्थिर uint32_t fa_sgs = fa.first.subgroup_आकार;
            स्थिर bool fa_ds = fa.first.subgroup_आकार == 0;

            स्थिर bool bf16_kv = fa.first.k_type == GGML_TYPE_BF16;

            स्थिर व्योम * spv_तथ्य;
            माप_प्रकार spv_आकार;
            स्थिर अक्षर *name;
            अगर (bf16_kv) {
#अगर defined(VK_KHR_shader_bfloat16) && defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
                अगर (!device->coopmat_bf16_support) जारी;
                spv_तथ्य = flash_attn_f32_f16_bf16_cm1_तथ्य;
                spv_आकार = flash_attn_f32_f16_bf16_cm1_len;
                name = aligned ? "flash_attn_f32_bf16_aligned_cm1" : "flash_attn_f32_bf16_cm1";
#अन्यथा
                जारी;
#पूर्ण_अगर
            } अन्यथा {
                अगर (f32acc) { spv_तथ्य = flash_attn_f32_f16_cm1_तथ्य;        spv_आकार = flash_attn_f32_f16_cm1_len; }
                अन्यथा        { spv_तथ्य = flash_attn_f32_f16_f16acc_cm1_तथ्य; spv_आकार = flash_attn_f32_f16_f16acc_cm1_len; }
                name = aligned ? "flash_attn_f32_f16_aligned_cm1" : "flash_attn_f32_f16_cm1";
            }
            ggml_vk_create_pipeline(device, fa.second, name, spv_आकार, spv_तथ्य, "main", 8,
                                    माप(vk_flash_attn_धकेलो_constants), {Br, 1, 1},
                                    get_fa_spec_constants(fa.first), aligned ? Bc : 1, true,
                                    !fa_ds, !fa_ds ? fa_sgs : 0);
        }
    }
#पूर्ण_अगर

#अगर defined(VK_NV_cooperative_matrix2) && defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
    अगर (device->coopmat2) {
        क्रम (स्वतः &fa : device->pipeline_flash_attn_f32_f16) {
            अगर (fa.first.path != FA_COOPMAT2) जारी;
            स्थिर uint32_t Br = fa.first.Br;
            स्थिर uint32_t Bc = fa.first.Bc;
            स्थिर bool aligned = fa.first.aligned;
            स्थिर bool f32acc = fa.first.f32acc;

            स्थिर bool bf16_kv = fa.first.k_type == GGML_TYPE_BF16;
            स्थिर व्योम * spv_तथ्य;
            माप_प्रकार spv_आकार;
            स्थिर अक्षर * name;
            अगर (bf16_kv) {
#अगर defined(VK_KHR_shader_bfloat16) && defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
                अगर (!device->coopmat2_bf16_support) जारी;
                spv_तथ्य = flash_attn_f32_f16_bf16_cm2_तथ्य;
                spv_आकार = flash_attn_f32_f16_bf16_cm2_len;
                name = aligned ? "flash_attn_f32_bf16_aligned_cm2" : "flash_attn_f32_bf16_cm2";
#अन्यथा
                जारी;
#पूर्ण_अगर
            } अन्यथा अगर (aligned) {
                अगर (f32acc) { spv_तथ्य = flash_attn_f32_f16_cm2_तथ्य;        spv_आकार = flash_attn_f32_f16_cm2_len;        name = "flash_attn_f32_f16_aligned_f32acc_cm2"; }
                अन्यथा        { spv_तथ्य = flash_attn_f32_f16_f16acc_cm2_तथ्य; spv_आकार = flash_attn_f32_f16_f16acc_cm2_len; name = "flash_attn_f32_f16_aligned_f16acc_cm2"; }
            } अन्यथा {
                अगर (f32acc) { spv_तथ्य = flash_attn_f32_f16_cm2_तथ्य;        spv_आकार = flash_attn_f32_f16_cm2_len;        name = "flash_attn_f32_f16_f32acc_cm2"; }
                अन्यथा        { spv_तथ्य = flash_attn_f32_f16_f16acc_cm2_तथ्य; spv_आकार = flash_attn_f32_f16_f16acc_cm2_len; name = "flash_attn_f32_f16_f16acc_cm2"; }
            }
            ggml_vk_create_pipeline(device, fa.second, name, spv_आकार, spv_तथ्य, "main", 8,
                                    माप(vk_flash_attn_धकेलो_constants), {Br, 1, 1},
                                    get_fa_spec_constants(fa.first), aligned ? Bc : 1, true, false, 0);
        }
    }
#पूर्ण_अगर

    स्वतः स्थिर &ggml_vk_mul_mm_spec = [&device](std::vector<uint32_t> spec, bool aligned) {
        spec.push_back(aligned ? 1u : 0u);  // constantID=11: ALIGNED
        अगर (device->vendor_id == VK_VENDOR_ID_INTEL && device->coopmat_support &&
            device->driver_id == vk::DriverId::eIntelProprietaryWindows) {
            spec.push_back(0u);  // constantID=12: SHMEM_STRIDE_PAD = 0
            spec.push_back(1u);  // constantID=13: APPLY_SLM_A_RESHAPE = true
        }
        वापस spec;
    };

    स्वतः स्थिर &ggml_vk_mul_mm_spec_quant = [&device](std::vector<uint32_t> spec, bool aligned, uint32_t type) {
        spec.push_back(aligned ? 1u : 0u);  // constantID=11: ALIGNED
        spec.push_back(type);               // constantID=12: MmTypeA
        अगर (device->vendor_id == VK_VENDOR_ID_INTEL && device->coopmat_support &&
            device->driver_id == vk::DriverId::eIntelProprietaryWindows) {
            spec.push_back(0u);  // constantID=13: SHMEM_STRIDE_PAD = 0
            spec.push_back(1u);  // constantID=14: APPLY_SLM_A_RESHAPE = true
        }
        वापस spec;
    };

    अटल स्थिर ggml_type non_lut_quant_types[] = {
        GGML_TYPE_Q1_0, GGML_TYPE_Q2_0, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0,
        GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0,
    };

#घोषणा FOR_EACH_LUT_TYPE_NONFP4(X) \
    X(GGML_TYPE_IQ1_S,   iq1_s)   \
    X(GGML_TYPE_IQ1_M,   iq1_m)   \
    X(GGML_TYPE_IQ2_XXS, iq2_xxs) \
    X(GGML_TYPE_IQ2_XS,  iq2_xs)  \
    X(GGML_TYPE_IQ2_S,   iq2_s)   \
    X(GGML_TYPE_IQ3_XXS, iq3_xxs) \
    X(GGML_TYPE_IQ3_S,   iq3_s)   \
    X(GGML_TYPE_IQ4_XS,  iq4_xs)  \
    X(GGML_TYPE_IQ4_NL,  iq4_nl)
#घोषणा FOR_EACH_LUT_FP4_TYPE(X) \
    X(GGML_TYPE_MXFP4,   mxfp4)   \
    X(GGML_TYPE_NVFP4,   nvfp4)
#घोषणा FOR_EACH_LUT_TYPE(X) \
    FOR_EACH_LUT_TYPE_NONFP4(X)  \
    FOR_EACH_LUT_FP4_TYPE(X)

    स्थिर पूर्णांक mul_mat_id_param_गिनती = 5;

    using spec_fn_t = std::function<std::vector<uint32_t>(स्थिर std::vector<uint32_t>&, bool)>;
    स्वतः स्थिर &create_mm_pipelines = [&](
        स्थिर vk_matmul_pipeline_कुंजी& कुंजी,
        स्थिर std::vector<vk_tile_config>& tile_configs,
        स्थिर std::string& shader_name, माप_प्रकार spv_len, स्थिर व्योम* spv_तथ्य,
        uint32_t धकेलो_constant_आकार, uint32_t param_गिनती,
        स्थिर spec_fn_t& spec_fn,
        bool disable_robustness = false, bool require_full_subgroups = false, uint32_t required_subgroup_आकार = 0,
        bool create_aligned = true, bool pin_subgroup_to_warp = false
    ) {
        स्वतः& vec = device->pipeline_matmul[कुंजी];
        स्थिर bool first_call = vec.empty();
        क्रम (माप_प्रकार i = 0; i < tile_configs.size(); i++) {
            स्थिर स्वतः& tc = tile_configs[i];

            // Intel coopmat1 pins the required subgroup size to each warptile's WARP element.
            स्थिर uint32_t rsgs = pin_subgroup_to_warp ? tc.warptile[WARP_आकार_IDX] : required_subgroup_आकार;
            स्थिर bool     rfs  = require_full_subgroups || pin_subgroup_to_warp;

            अगर (first_call) {
                vk_matmul_pipeline_pair pair{};
                pair.align = tc.align;
                std::string suffix = "_" + std::to_string(i);
                pair.unaligned = std::make_shared<vk_pipeline_struct>();
                अगर (create_aligned) {
                    pair.aligned = std::make_shared<vk_pipeline_struct>();
                }
                vec.push_back(pair);
            }

            ggml_vk_create_pipeline(device, vec[i].unaligned,
                vec[i].unaligned->name.empty() ? (shader_name + "_" + std::to_string(i)).c_str() : vec[i].unaligned->name.c_str(),
                spv_len, spv_तथ्य, "main", param_गिनती, धकेलो_constant_आकार,
                tc.wg_denoms, spec_fn(tc.warptile, false), 1,
                disable_robustness, rfs, rsgs);

            अगर (vec[i].aligned) {
                ggml_vk_create_pipeline(device, vec[i].aligned,
                    vec[i].aligned->name.empty() ? (shader_name + "_aligned_" + std::to_string(i)).c_str() : vec[i].aligned->name.c_str(),
                    spv_len, spv_तथ्य, "main", param_गिनती, धकेलो_constant_आकार,
                    tc.wg_denoms, spec_fn(tc.warptile, true), tc.align,
                    disable_robustness, rfs, rsgs);
            }
        }
    };

    स्वतः filter_tc = [&](स्थिर std::vector<vk_tile_config>& configs, ggml_type type, bool is_id, bool is_int = false) -> std::vector<vk_tile_config> {
        std::vector<vk_tile_config> परिणाम;
        bool enabled[3];
        अगर (is_int) {
            enabled[0] = is_id ? device->mul_mat_id_s_int[type] : device->mul_mat_s_int[type];
            enabled[1] = is_id ? device->mul_mat_id_m_int[type] : device->mul_mat_m_int[type];
            enabled[2] = is_id ? device->mul_mat_id_l_int[type] : device->mul_mat_l_int[type];
        } अन्यथा {
            enabled[0] = is_id ? device->mul_mat_id_s[type] : device->mul_mat_s[type];
            enabled[1] = is_id ? device->mul_mat_id_m[type] : device->mul_mat_m[type];
            enabled[2] = is_id ? device->mul_mat_id_l[type] : device->mul_mat_l[type];
        }
        क्रम (माप_प्रकार i = 0; i < configs.size() && i < 3; i++) {
            अगर (enabled[i]) परिणाम.push_back(configs[i]);
        }
        वापस परिणाम;
    };

    std::vector<vk_tile_config> tc_mm = {{s_warptile, s_wg_denoms, s_align}, {m_warptile, m_wg_denoms, m_align}, {l_warptile, l_wg_denoms, l_align}};
    std::vector<vk_tile_config> tc_mmq = {{s_warptile_mmq, s_mmq_wg_denoms, s_align}, {m_warptile_mmq, m_mmq_wg_denoms, m_align}, {l_warptile_mmq, l_mmq_wg_denoms, l_align}};

#अगर defined(VK_NV_cooperative_matrix2) && defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
    अगर (device->coopmat2) {
        स्वतः स्थिर &ggml_vk_mul_mm_cm2_spec = [&](std::vector<uint32_t> spec, bool aligned, uint32_t type = UINT32_MAX) {
            spec.push_back(aligned ? 1u : 0u);        // ALIGNED
            spec.push_back(device->subgroup_आकार);     // subgroup_size
            अगर (type != UINT32_MAX) {
                spec.push_back(type);                  // MmTypeA
                spec.push_back((uint32_t)ggml_type_आकार((ggml_type)type)); // MmABlockBytes
            }
            वापस spec;
        };

        std::vector<vk_tile_config> tc_mmq_k = {{s_warptile_mmq_k, s_mmq_wg_denoms_k, s_align}, {m_warptile_mmq_k, m_mmq_wg_denoms_k, m_align}, {l_warptile_mmq_k, l_mmq_wg_denoms_k, l_align}};
        std::vector<vk_tile_config> tc_mmqid = {{s_warptile_mmqid, s_mmqid_wg_denoms, s_align}, {m_warptile_mmqid, m_mmqid_wg_denoms, m_align}, {l_warptile_mmqid, l_mmqid_wg_denoms, l_align}};

        spec_fn_t cm2_spec = [&](स्थिर std::vector<uint32_t>& wt, bool a) { वापस ggml_vk_mul_mm_cm2_spec(wt, a); };

        // F16 x F16
        create_mm_pipelines({GGML_TYPE_F16, GGML_TYPE_F16, false, true},  tc_mm, "matmul_f16_f16acc", matmul_f16_f16acc_cm2_len, matmul_f16_f16acc_cm2_तथ्य, माप(vk_mat_mat_धकेलो_constants), 3, cm2_spec, true);
        create_mm_pipelines({GGML_TYPE_F16, GGML_TYPE_F16, false, false}, tc_mm, "matmul_f16",        matmul_f16_cm2_len,        matmul_f16_cm2_तथ्य,        माप(vk_mat_mat_धकेलो_constants), 3, cm2_spec, true);
#अगर defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        अगर (device->coopmat_bf16_support) {
            create_mm_pipelines({GGML_TYPE_BF16, GGML_TYPE_BF16, false, false}, tc_mm, "matmul_bf16", matmul_bf16_cm2_len, matmul_bf16_cm2_तथ्य, माप(vk_mat_mat_धकेलो_constants), 3, cm2_spec, true);
        }
#पूर्ण_अगर
        क्रम (स्थिर स्वतः type : non_lut_quant_types) {
            // regression in unified shader on Ampere
            अगर (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
                जारी;
            }
            स्वतः& tc = ((type >= GGML_TYPE_Q2_K && type <= GGML_TYPE_Q6_K) || type == GGML_TYPE_TQ1_0 || type == GGML_TYPE_TQ2_0) ? tc_mmq_k : tc_mmq;
            spec_fn_t qs = [&, type](स्थिर std::vector<uint32_t>& wt, bool a) { वापस ggml_vk_mul_mm_cm2_spec(wt, a, (uint32_t)type); };
            create_mm_pipelines({type, GGML_TYPE_F16, false, true},  tc, "matmul_quant_f16_f16acc", matmul_quant_f16_f16acc_cm2_len, matmul_quant_f16_f16acc_cm2_तथ्य, माप(vk_mat_mat_धकेलो_constants), 3, qs, true);
            create_mm_pipelines({type, GGML_TYPE_F16, false, false}, tc, "matmul_quant_f16",        matmul_quant_f16_cm2_len,        matmul_quant_f16_cm2_तथ्य,        माप(vk_mat_mat_धकेलो_constants), 3, qs, true);
        }
        create_mm_pipelines({GGML_TYPE_Q4_K, GGML_TYPE_F16, false, true},  tc_mmq_k, "matmul_q4_k_f16_f16acc", matmul_q4_k_f16_f16acc_cm2_len, matmul_q4_k_f16_f16acc_cm2_तथ्य, माप(vk_mat_mat_धकेलो_constants), 3, cm2_spec, true);
        create_mm_pipelines({GGML_TYPE_Q4_K, GGML_TYPE_F16, false, false}, tc_mmq_k, "matmul_q4_k_f16",        matmul_q4_k_f16_cm2_len,        matmul_q4_k_f16_cm2_तथ्य,        माप(vk_mat_mat_धकेलो_constants), 3, cm2_spec, true);
        create_mm_pipelines({GGML_TYPE_Q5_K, GGML_TYPE_F16, false, true},  tc_mmq_k, "matmul_q5_k_f16_f16acc", matmul_q5_k_f16_f16acc_cm2_len, matmul_q5_k_f16_f16acc_cm2_तथ्य, माप(vk_mat_mat_धकेलो_constants), 3, cm2_spec, true);
        create_mm_pipelines({GGML_TYPE_Q5_K, GGML_TYPE_F16, false, false}, tc_mmq_k, "matmul_q5_k_f16",        matmul_q5_k_f16_cm2_len,        matmul_q5_k_f16_cm2_तथ्य,        माप(vk_mat_mat_धकेलो_constants), 3, cm2_spec, true);
#घोषणा X_CM2(TYPE, tstr) \
        { स्वतः tc = filter_tc(tc_mmq, TYPE, false); \
          अगर (!tc.empty()) { \
              create_mm_pipelines({TYPE, GGML_TYPE_F16, false, true},  tc, "matmul_" #tstr "_f16_f16acc", matmul_##tstr##_f16_f16acc_cm2_len, matmul_##tstr##_f16_f16acc_cm2_data, माप(vk_mat_mat_धकेलो_constants), 3, cm2_spec, true); \
              create_mm_pipelines({TYPE, GGML_TYPE_F16, false, false}, tc, "matmul_" #tstr "_f16",        matmul_##tstr##_f16_cm2_len,        matmul_##tstr##_f16_cm2_data,        माप(vk_mat_mat_धकेलो_constants), 3, cm2_spec, true); \
          } }
        FOR_EACH_LUT_TYPE_NONFP4(X_CM2)
#अगर defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
        अगर (device->ocp_fp4) {
#घोषणा X_CM2_OCP(TYPE, tstr) \
            { स्वतः tc = filter_tc(tc_mmq, TYPE, false); \
              अगर (!tc.empty()) { \
                  create_mm_pipelines({TYPE, GGML_TYPE_F16, false, true},  tc, "matmul_" #tstr "_f16_ocp_f16acc", matmul_##tstr##_f16_ocp_f16acc_cm2_len, matmul_##tstr##_f16_ocp_f16acc_cm2_data, माप(vk_mat_mat_धकेलो_constants), 3, cm2_spec, true); \
                  create_mm_pipelines({TYPE, GGML_TYPE_F16, false, false}, tc, "matmul_" #tstr "_f16_ocp",        matmul_##tstr##_f16_ocp_cm2_len,        matmul_##tstr##_f16_ocp_cm2_data,        माप(vk_mat_mat_धकेलो_constants), 3, cm2_spec, true); \
              } }
            FOR_EACH_LUT_FP4_TYPE(X_CM2_OCP)
#अघोषित X_CM2_OCP
        } अन्यथा
#पूर्ण_अगर
        {
            FOR_EACH_LUT_FP4_TYPE(X_CM2)
        }
#अघोषित X_CM2

        GGML_ASSERT(device->subgroup_ballot);

        create_mm_pipelines({GGML_TYPE_F16, GGML_TYPE_F16, true, true},  tc_mm, "matmul_id_subgroup_f16_f16acc", matmul_id_subgroup_f16_f16acc_cm2_len, matmul_id_subgroup_f16_f16acc_cm2_तथ्य, माप(vk_mat_mat_id_धकेलो_constants), 5, cm2_spec, true);
        create_mm_pipelines({GGML_TYPE_F16, GGML_TYPE_F16, true, false}, tc_mm, "matmul_id_subgroup_f16",        matmul_id_subgroup_f16_cm2_len,        matmul_id_subgroup_f16_cm2_तथ्य,        माप(vk_mat_mat_id_धकेलो_constants), 5, cm2_spec, true);
#अगर defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        अगर (device->coopmat_bf16_support) {
            create_mm_pipelines({GGML_TYPE_BF16, GGML_TYPE_BF16, true, false}, tc_mm, "matmul_id_subgroup_bf16", matmul_id_subgroup_bf16_cm2_len, matmul_id_subgroup_bf16_cm2_तथ्य, माप(vk_mat_mat_id_धकेलो_constants), 5, cm2_spec, true);
        }
#पूर्ण_अगर
        क्रम (स्थिर स्वतः type : non_lut_quant_types) {
            अगर (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
                जारी;
            }
            spec_fn_t qs_id = [&, type](स्थिर std::vector<uint32_t>& wt, bool a) { वापस ggml_vk_mul_mm_cm2_spec(wt, a, (uint32_t)type); };
            create_mm_pipelines({type, GGML_TYPE_F16, true, true},  tc_mmqid, "matmul_id_subgroup_quant_f16_f16acc", matmul_id_subgroup_quant_f16_f16acc_cm2_len, matmul_id_subgroup_quant_f16_f16acc_cm2_तथ्य, माप(vk_mat_mat_id_धकेलो_constants), 5, qs_id, true);
            create_mm_pipelines({type, GGML_TYPE_F16, true, false}, tc_mmqid, "matmul_id_subgroup_quant_f16",        matmul_id_subgroup_quant_f16_cm2_len,        matmul_id_subgroup_quant_f16_cm2_तथ्य,        माप(vk_mat_mat_id_धकेलो_constants), 5, qs_id, true);
        }
        create_mm_pipelines({GGML_TYPE_Q4_K, GGML_TYPE_F16, true, true},  tc_mmqid, "matmul_id_subgroup_q4_k_f16_f16acc", matmul_id_subgroup_q4_k_f16_f16acc_cm2_len, matmul_id_subgroup_q4_k_f16_f16acc_cm2_तथ्य, माप(vk_mat_mat_id_धकेलो_constants), 5, cm2_spec, true);
        create_mm_pipelines({GGML_TYPE_Q4_K, GGML_TYPE_F16, true, false}, tc_mmqid, "matmul_id_subgroup_q4_k_f16",        matmul_id_subgroup_q4_k_f16_cm2_len,        matmul_id_subgroup_q4_k_f16_cm2_तथ्य,        माप(vk_mat_mat_id_धकेलो_constants), 5, cm2_spec, true);
        create_mm_pipelines({GGML_TYPE_Q5_K, GGML_TYPE_F16, true, true},  tc_mmqid, "matmul_id_subgroup_q5_k_f16_f16acc", matmul_id_subgroup_q5_k_f16_f16acc_cm2_len, matmul_id_subgroup_q5_k_f16_f16acc_cm2_तथ्य, माप(vk_mat_mat_id_धकेलो_constants), 5, cm2_spec, true);
        create_mm_pipelines({GGML_TYPE_Q5_K, GGML_TYPE_F16, true, false}, tc_mmqid, "matmul_id_subgroup_q5_k_f16",        matmul_id_subgroup_q5_k_f16_cm2_len,        matmul_id_subgroup_q5_k_f16_cm2_तथ्य,        माप(vk_mat_mat_id_धकेलो_constants), 5, cm2_spec, true);
#घोषणा X_CM2_ID(TYPE, tstr) \
        { स्वतः tc = filter_tc(tc_mmqid, TYPE, true); \
          अगर (!tc.empty()) { \
              create_mm_pipelines({TYPE, GGML_TYPE_F16, true, true},  tc, "matmul_id_subgroup_" #tstr "_f16_f16acc", matmul_id_subgroup_##tstr##_f16_f16acc_cm2_len, matmul_id_subgroup_##tstr##_f16_f16acc_cm2_data, माप(vk_mat_mat_id_धकेलो_constants), 5, cm2_spec, true); \
              create_mm_pipelines({TYPE, GGML_TYPE_F16, true, false}, tc, "matmul_id_subgroup_" #tstr "_f16",        matmul_id_subgroup_##tstr##_f16_cm2_len,        matmul_id_subgroup_##tstr##_f16_cm2_data,        माप(vk_mat_mat_id_धकेलो_constants), 5, cm2_spec, true); \
          } }
        FOR_EACH_LUT_TYPE_NONFP4(X_CM2_ID)
#अगर defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
        अगर (device->ocp_fp4) {
#घोषणा X_CM2_ID_OCP(TYPE, tstr) \
            { स्वतः tc = filter_tc(tc_mmqid, TYPE, true); \
              अगर (!tc.empty()) { \
                  create_mm_pipelines({TYPE, GGML_TYPE_F16, true, true},  tc, "matmul_id_subgroup_" #tstr "_f16_ocp_f16acc", matmul_id_subgroup_##tstr##_f16_ocp_f16acc_cm2_len, matmul_id_subgroup_##tstr##_f16_ocp_f16acc_cm2_data, माप(vk_mat_mat_id_धकेलो_constants), 5, cm2_spec, true); \
                  create_mm_pipelines({TYPE, GGML_TYPE_F16, true, false}, tc, "matmul_id_subgroup_" #tstr "_f16_ocp",        matmul_id_subgroup_##tstr##_f16_ocp_cm2_len,        matmul_id_subgroup_##tstr##_f16_ocp_cm2_data,        माप(vk_mat_mat_id_धकेलो_constants), 5, cm2_spec, true); \
              } }
            FOR_EACH_LUT_FP4_TYPE(X_CM2_ID_OCP)
#अघोषित X_CM2_ID_OCP
        } अन्यथा
#पूर्ण_अगर
        {
            FOR_EACH_LUT_FP4_TYPE(X_CM2_ID)
        }
#अघोषित X_CM2_ID
    } अन्यथा
#पूर्ण_अगर  // defined(VK_NV_cooperative_matrix2) && defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
#अगर defined(VK_KHR_cooperative_matrix) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    अगर (device->coopmat_support) {
        spec_fn_t cm1_spec = [&](स्थिर std::vector<uint32_t>& wt, bool a) { वापस ggml_vk_mul_mm_spec(wt, a); };

        // Intel coopmat1 pins each pipeline's required subgroup size to its warptile WARP element.
        स्थिर bool cm1_pin = device->vendor_id == VK_VENDOR_ID_INTEL;

        // Intel coopmat1 uses a dedicated large-tile config for quant matmul_id.
        std::vector<vk_tile_config> tc_mmq_id = tc_mmq;
        अगर (cm1_pin) {
            tc_mmq_id[2] = { { 512, 128, 128, 32, 32, 32, 2, device->coopmat_m, device->coopmat_n, device->coopmat_k, 32 }, { 128, 128, 1 }, 32 };
        }

        स्वतः cm1_create = [&](vk_matmul_pipeline_कुंजी कुंजी, स्थिर std::vector<vk_tile_config>& tc_base,
                              स्थिर std::string& name, माप_प्रकार len, स्थिर व्योम* तथ्य, uint32_t pc_आकार, uint32_t pc) {
            स्वतः tc = filter_tc(tc_base, कुंजी.type_a, कुंजी.mul_mat_id);
            अगर (!tc.empty()) create_mm_pipelines(कुंजी, tc, name, len, तथ्य, pc_आकार, pc, cm1_spec, false, true, 0, true, cm1_pin);
        };
        स्वतः cm1_create_quant = [&](vk_matmul_pipeline_कुंजी कुंजी, स्थिर std::vector<vk_tile_config>& tc_base,
                                    स्थिर std::string& name, माप_प्रकार len, स्थिर व्योम* तथ्य, uint32_t pc_आकार, uint32_t pc) {
            spec_fn_t qs = [&, type_a=कुंजी.type_a](स्थिर std::vector<uint32_t>& wt, bool a) { वापस ggml_vk_mul_mm_spec_quant(wt, a, (uint32_t)type_a); };
            स्वतः tc = filter_tc(tc_base, कुंजी.type_a, कुंजी.mul_mat_id);
            अगर (!tc.empty()) create_mm_pipelines(कुंजी, tc, name, len, तथ्य, pc_आकार, pc, qs, false, true, 0, true, cm1_pin);
        };
        // int8 MMQ helper: per-type cm1 shader, warptile passed as-is (carries DEVICE_ARCH in
        // spec constant WARP_SIZE_IDX+1), subgroup size pinned to the warptile WARP element, no aligned variant.
        स्वतः cm1_create_mmq = [&](vk_matmul_pipeline_कुंजी कुंजी, स्थिर std::vector<vk_tile_config>& tc_base,
                                  स्थिर std::string& name, माप_प्रकार len, स्थिर व्योम* तथ्य, uint32_t pc_आकार, uint32_t pc) {
            spec_fn_t identity = [](स्थिर std::vector<uint32_t>& wt, bool) { वापस wt; };
            स्वतः tc = filter_tc(tc_base, कुंजी.type_a, कुंजी.mul_mat_id, true);
            अगर (!tc.empty()) create_mm_pipelines(कुंजी, tc, name, len, तथ्य, pc_आकार, pc, identity, false, false, 0, false, true);
        };

        std::vector<vk_tile_config> tc_mmq_cm1_int = {
            {s_warptile_mmq_cm1_int, s_mmq_wg_denoms, s_align},
            {m_warptile_mmq_cm1_int, m_mmq_wg_denoms, m_align},
            {l_warptile_mmq_cm1_int, l_mmq_wg_denoms, l_align},
        };
        std::vector<vk_tile_config> tc_mmq_cm1_int_k = {
            {s_warptile_mmq_cm1_int_k, s_mmq_cm1_wg_denoms_k, s_align},
            {m_warptile_mmq_cm1_int_k, m_mmq_cm1_wg_denoms_k, m_align},
            {l_warptile_mmq_cm1_int_k, l_mmq_cm1_wg_denoms_k, l_align},
        };

        // Some quants are not performant on RDNA4, those fall back to FP16 matmul
        स्थिर bool rdna3 = device->architecture == AMD_RDNA3;
        स्थिर bool rdna4 = device->architecture == AMD_RDNA4;

        cm1_create({GGML_TYPE_F32, GGML_TYPE_F32, false, false}, tc_mm, "matmul_f32_f32",     matmul_f32_f32_cm1_len,     matmul_f32_f32_cm1_तथ्य,     माप(vk_mat_mat_धकेलो_constants), 3);
        cm1_create({GGML_TYPE_F32, GGML_TYPE_F16, false, false}, tc_mm, "matmul_f32_f16",     matmul_f32_f16_cm1_len,     matmul_f32_f16_cm1_तथ्य,     माप(vk_mat_mat_धकेलो_constants), 3);
        अगर (device->coopmat_acc_f16_support) {
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F16, false, true},  tc_mm, "matmul_f16_f16acc", matmul_f16_f16acc_cm1_len, matmul_f16_f16acc_cm1_तथ्य, माप(vk_mat_mat_धकेलो_constants), 3);
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F32, false, true},  tc_mm, "matmul_f16_f32_f16acc", matmul_f16_f32_f16acc_cm1_len, matmul_f16_f32_f16acc_cm1_तथ्य, माप(vk_mat_mat_धकेलो_constants), 3);
        }
        अगर (device->coopmat_acc_f32_support) {
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F16, false, false}, tc_mm, "matmul_f16",      matmul_f16_cm1_len,      matmul_f16_cm1_तथ्य,      माप(vk_mat_mat_धकेलो_constants), 3);
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F32, false, false}, tc_mm, "matmul_f16_f32",  matmul_f16_f32_cm1_len,  matmul_f16_f32_cm1_तथ्य,  माप(vk_mat_mat_धकेलो_constants), 3);
        }
#अगर defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        अगर (device->coopmat_bf16_support) {
            cm1_create({GGML_TYPE_BF16, GGML_TYPE_BF16, false, false}, tc_mm, "matmul_bf16", matmul_bf16_cm1_len, matmul_bf16_cm1_तथ्य, माप(vk_mat_mat_धकेलो_constants), 3);
        }
#पूर्ण_अगर
        क्रम (स्थिर स्वतः type : non_lut_quant_types) {
            अगर (device->coopmat_acc_f16_support) {
                cm1_create_quant({type, GGML_TYPE_F32, false, true},  tc_mmq, "matmul_quant_f32_f16acc", matmul_quant_f32_f16acc_cm1_len, matmul_quant_f32_f16acc_cm1_तथ्य, माप(vk_mat_mat_धकेलो_constants), 3);
                cm1_create_quant({type, GGML_TYPE_F16, false, true},  tc_mmq, "matmul_quant_f16_f16acc", matmul_quant_f16_f16acc_cm1_len, matmul_quant_f16_f16acc_cm1_तथ्य, माप(vk_mat_mat_धकेलो_constants), 3);
            }
            अगर (device->coopmat_acc_f32_support) {
                cm1_create_quant({type, GGML_TYPE_F32, false, false}, tc_mmq, "matmul_quant_f32",        matmul_quant_f32_cm1_len,        matmul_quant_f32_cm1_तथ्य,        माप(vk_mat_mat_धकेलो_constants), 3);
                cm1_create_quant({type, GGML_TYPE_F16, false, false}, tc_mmq, "matmul_quant_f16",        matmul_quant_f16_cm1_len,        matmul_quant_f16_cm1_तथ्य,        माप(vk_mat_mat_धकेलो_constants), 3);
            }
        }
        // The _f16 variants provide the f16 B-type pipeline used when y_non_contig converts f32->f16.
#घोषणा X_CM1(TYPE, tstr) \
        अगर (device->coopmat_acc_f16_support) { \
            cm1_create({TYPE, GGML_TYPE_F32, false, true},  tc_mmq, "matmul_" #tstr "_f32_f16acc", matmul_##tstr##_f32_f16acc_cm1_len, matmul_##tstr##_f32_f16acc_cm1_data, माप(vk_mat_mat_धकेलो_constants), 3); \
            cm1_create({TYPE, GGML_TYPE_F16, false, true},  tc_mmq, "matmul_" #tstr "_f16_f16acc", matmul_##tstr##_f16_f16acc_cm1_len, matmul_##tstr##_f16_f16acc_cm1_data, माप(vk_mat_mat_धकेलो_constants), 3); \
        } \
        अगर (device->coopmat_acc_f32_support) { \
            cm1_create({TYPE, GGML_TYPE_F32, false, false}, tc_mmq, "matmul_" #tstr "_f32",        matmul_##tstr##_f32_cm1_len,        matmul_##tstr##_f32_cm1_data,        माप(vk_mat_mat_धकेलो_constants), 3); \
            cm1_create({TYPE, GGML_TYPE_F16, false, false}, tc_mmq, "matmul_" #tstr "_f16",        matmul_##tstr##_f16_cm1_len,        matmul_##tstr##_f16_cm1_data,        माप(vk_mat_mat_धकेलो_constants), 3); \
        }
        FOR_EACH_LUT_TYPE_NONFP4(X_CM1)
#अगर defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
        अगर (device->ocp_fp4) {
#घोषणा X_CM1_OCP(TYPE, tstr) \
            अगर (device->coopmat_acc_f16_support) { \
                cm1_create({TYPE, GGML_TYPE_F32, false, true},  tc_mmq, "matmul_" #tstr "_f32_ocp_f16acc", matmul_##tstr##_f32_ocp_f16acc_cm1_len, matmul_##tstr##_f32_ocp_f16acc_cm1_data, माप(vk_mat_mat_धकेलो_constants), 3); \
                cm1_create({TYPE, GGML_TYPE_F16, false, true},  tc_mmq, "matmul_" #tstr "_f16_ocp_f16acc", matmul_##tstr##_f16_ocp_f16acc_cm1_len, matmul_##tstr##_f16_ocp_f16acc_cm1_data, माप(vk_mat_mat_धकेलो_constants), 3); \
            } \
            अगर (device->coopmat_acc_f32_support) { \
                cm1_create({TYPE, GGML_TYPE_F32, false, false}, tc_mmq, "matmul_" #tstr "_f32_ocp",        matmul_##tstr##_f32_ocp_cm1_len,        matmul_##tstr##_f32_ocp_cm1_data,        माप(vk_mat_mat_धकेलो_constants), 3); \
                cm1_create({TYPE, GGML_TYPE_F16, false, false}, tc_mmq, "matmul_" #tstr "_f16_ocp",        matmul_##tstr##_f16_ocp_cm1_len,        matmul_##tstr##_f16_ocp_cm1_data,        माप(vk_mat_mat_धकेलो_constants), 3); \
            }
            FOR_EACH_LUT_FP4_TYPE(X_CM1_OCP)
#अघोषित X_CM1_OCP
        } अन्यथा
#पूर्ण_अगर
        {
            FOR_EACH_LUT_FP4_TYPE(X_CM1)
        }
#अघोषित X_CM1

        अगर (device->coopmat_int_support && (rdna3 || rdna4)) {
            cm1_create_mmq({GGML_TYPE_Q4_0,   GGML_TYPE_Q8_1, false, false}, tc_mmq_cm1_int,   "matmul_q4_0_q8_1",   matmul_q4_0_q8_1_cm1_len,   matmul_q4_0_q8_1_cm1_तथ्य,   माप(vk_mat_mat_धकेलो_constants), 3);
            अगर (!rdna4) { cm1_create_mmq({GGML_TYPE_Q4_1, GGML_TYPE_Q8_1, false, false}, tc_mmq_cm1_int,   "matmul_q4_1_q8_1",   matmul_q4_1_q8_1_cm1_len,   matmul_q4_1_q8_1_cm1_तथ्य,   माप(vk_mat_mat_धकेलो_constants), 3); }
            cm1_create_mmq({GGML_TYPE_Q5_0,   GGML_TYPE_Q8_1, false, false}, tc_mmq_cm1_int,   "matmul_q5_0_q8_1",   matmul_q5_0_q8_1_cm1_len,   matmul_q5_0_q8_1_cm1_तथ्य,   माप(vk_mat_mat_धकेलो_constants), 3);
            अगर (!rdna4) { cm1_create_mmq({GGML_TYPE_Q5_1, GGML_TYPE_Q8_1, false, false}, tc_mmq_cm1_int,   "matmul_q5_1_q8_1",   matmul_q5_1_q8_1_cm1_len,   matmul_q5_1_q8_1_cm1_तथ्य,   माप(vk_mat_mat_धकेलो_constants), 3); }
            cm1_create_mmq({GGML_TYPE_Q8_0,   GGML_TYPE_Q8_1, false, false}, tc_mmq_cm1_int,   "matmul_q8_0_q8_1",   matmul_q8_0_q8_1_cm1_len,   matmul_q8_0_q8_1_cm1_तथ्य,   माप(vk_mat_mat_धकेलो_constants), 3);
            cm1_create_mmq({GGML_TYPE_IQ4_NL, GGML_TYPE_Q8_1, false, false}, tc_mmq_cm1_int,   "matmul_iq4_nl_q8_1", matmul_iq4_nl_q8_1_cm1_len, matmul_iq4_nl_q8_1_cm1_तथ्य, माप(vk_mat_mat_धकेलो_constants), 3);
            cm1_create_mmq({GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_1, false, false}, tc_mmq_cm1_int,   "matmul_iq4_xs_q8_1", matmul_iq4_xs_q8_1_cm1_len, matmul_iq4_xs_q8_1_cm1_तथ्य, माप(vk_mat_mat_धकेलो_constants), 3);
            cm1_create_mmq({GGML_TYPE_MXFP4,  GGML_TYPE_Q8_1, false, false}, tc_mmq_cm1_int,   "matmul_mxfp4_q8_1",  matmul_mxfp4_q8_1_cm1_len,  matmul_mxfp4_q8_1_cm1_तथ्य,  माप(vk_mat_mat_धकेलो_constants), 3);
            cm1_create_mmq({GGML_TYPE_Q3_K,   GGML_TYPE_Q8_1, false, false}, tc_mmq_cm1_int_k, "matmul_q3_k_q8_1",   matmul_q3_k_q8_1_cm1_len,   matmul_q3_k_q8_1_cm1_तथ्य,   माप(vk_mat_mat_धकेलो_constants), 3);
            अगर (!rdna4) { cm1_create_mmq({GGML_TYPE_Q4_K, GGML_TYPE_Q8_1, false, false}, tc_mmq_cm1_int,   "matmul_q4_k_q8_1",   matmul_q4_k_q8_1_cm1_len,   matmul_q4_k_q8_1_cm1_तथ्य,   माप(vk_mat_mat_धकेलो_constants), 3); }
            अगर (!rdna4) { cm1_create_mmq({GGML_TYPE_Q5_K, GGML_TYPE_Q8_1, false, false}, tc_mmq_cm1_int,   "matmul_q5_k_q8_1",   matmul_q5_k_q8_1_cm1_len,   matmul_q5_k_q8_1_cm1_तथ्य,   माप(vk_mat_mat_धकेलो_constants), 3); }
            cm1_create_mmq({GGML_TYPE_Q6_K,   GGML_TYPE_Q8_1, false, false}, tc_mmq_cm1_int_k, "matmul_q6_k_q8_1",   matmul_q6_k_q8_1_cm1_len,   matmul_q6_k_q8_1_cm1_तथ्य,   माप(vk_mat_mat_धकेलो_constants), 3);
            अगर (!rdna4) { cm1_create_mmq({GGML_TYPE_NVFP4, GGML_TYPE_Q8_1, false, false}, tc_mmq_cm1_int_k, "matmul_nvfp4_q8_1",  matmul_nvfp4_q8_1_cm1_len,  matmul_nvfp4_q8_1_cm1_तथ्य,  माप(vk_mat_mat_धकेलो_constants), 3); }
        }

        GGML_ASSERT(device->subgroup_ballot);

        cm1_create({GGML_TYPE_F32, GGML_TYPE_F32, true, false}, tc_mm, "matmul_id_subgroup_f32_f32", matmul_id_subgroup_f32_f32_cm1_len, matmul_id_subgroup_f32_f32_cm1_तथ्य, माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती);
        अगर (device->coopmat_acc_f16_support) {
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F16, true, true},  tc_mm, "matmul_id_subgroup_f16_f16acc",     matmul_id_subgroup_f16_f16acc_cm1_len,     matmul_id_subgroup_f16_f16acc_cm1_तथ्य,     माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती);
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F32, true, true},  tc_mm, "matmul_id_subgroup_f16_f32_f16acc", matmul_id_subgroup_f16_f32_f16acc_cm1_len, matmul_id_subgroup_f16_f32_f16acc_cm1_तथ्य, माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती);
        }
        अगर (device->coopmat_acc_f32_support) {
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F16, true, false}, tc_mm, "matmul_id_subgroup_f16",     matmul_id_subgroup_f16_cm1_len,     matmul_id_subgroup_f16_cm1_तथ्य,     माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती);
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F32, true, false}, tc_mm, "matmul_id_subgroup_f16_f32", matmul_id_subgroup_f16_f32_cm1_len, matmul_id_subgroup_f16_f32_cm1_तथ्य, माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती);
        }
#अगर defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        अगर (device->coopmat_bf16_support) {
            cm1_create({GGML_TYPE_BF16, GGML_TYPE_BF16, true, false}, tc_mm, "matmul_id_subgroup_bf16", matmul_id_subgroup_bf16_cm1_len, matmul_id_subgroup_bf16_cm1_तथ्य, माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती);
        }
#पूर्ण_अगर
        क्रम (स्थिर स्वतः type : non_lut_quant_types) {
            अगर (device->coopmat_acc_f16_support) {
                cm1_create_quant({type, GGML_TYPE_F32, true, true},  tc_mmq_id, "matmul_id_subgroup_quant_f32_f16acc", matmul_id_subgroup_quant_f32_f16acc_cm1_len, matmul_id_subgroup_quant_f32_f16acc_cm1_तथ्य, माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती);
                cm1_create_quant({type, GGML_TYPE_F16, true, true},  tc_mmq_id, "matmul_id_subgroup_quant_f16_f16acc", matmul_id_subgroup_quant_f16_f16acc_cm1_len, matmul_id_subgroup_quant_f16_f16acc_cm1_तथ्य, माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती);
            }
            अगर (device->coopmat_acc_f32_support) {
                cm1_create_quant({type, GGML_TYPE_F32, true, false}, tc_mmq_id, "matmul_id_subgroup_quant_f32",        matmul_id_subgroup_quant_f32_cm1_len,        matmul_id_subgroup_quant_f32_cm1_तथ्य,        माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती);
                cm1_create_quant({type, GGML_TYPE_F16, true, false}, tc_mmq_id, "matmul_id_subgroup_quant_f16",        matmul_id_subgroup_quant_f16_cm1_len,        matmul_id_subgroup_quant_f16_cm1_तथ्य,        माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती);
            }
        }
        // The _f16 variants provide the f16 B-type pipeline used when y_non_contig converts f32->f16.
#घोषणा X_CM1_ID(TYPE, tstr) \
        अगर (device->coopmat_acc_f16_support) { \
            cm1_create({TYPE, GGML_TYPE_F32, true, true},  tc_mmq_id, "matmul_id_subgroup_" #tstr "_f32_f16acc", matmul_id_subgroup_##tstr##_f32_f16acc_cm1_len, matmul_id_subgroup_##tstr##_f32_f16acc_cm1_data, माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती); \
            cm1_create({TYPE, GGML_TYPE_F16, true, true},  tc_mmq_id, "matmul_id_subgroup_" #tstr "_f16_f16acc", matmul_id_subgroup_##tstr##_f16_f16acc_cm1_len, matmul_id_subgroup_##tstr##_f16_f16acc_cm1_data, माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती); \
        } \
        अगर (device->coopmat_acc_f32_support) { \
            cm1_create({TYPE, GGML_TYPE_F32, true, false}, tc_mmq_id, "matmul_id_subgroup_" #tstr "_f32",        matmul_id_subgroup_##tstr##_f32_cm1_len,        matmul_id_subgroup_##tstr##_f32_cm1_data,        माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती); \
            cm1_create({TYPE, GGML_TYPE_F16, true, false}, tc_mmq_id, "matmul_id_subgroup_" #tstr "_f16",        matmul_id_subgroup_##tstr##_f16_cm1_len,        matmul_id_subgroup_##tstr##_f16_cm1_data,        माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती); \
        }
        FOR_EACH_LUT_TYPE_NONFP4(X_CM1_ID)
#अगर defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
        अगर (device->ocp_fp4) {
#घोषणा X_CM1_ID_OCP(TYPE, tstr) \
            अगर (device->coopmat_acc_f16_support) { \
                cm1_create({TYPE, GGML_TYPE_F32, true, true},  tc_mmq_id, "matmul_id_subgroup_" #tstr "_f32_ocp_f16acc", matmul_id_subgroup_##tstr##_f32_ocp_f16acc_cm1_len, matmul_id_subgroup_##tstr##_f32_ocp_f16acc_cm1_data, माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती); \
                cm1_create({TYPE, GGML_TYPE_F16, true, true},  tc_mmq_id, "matmul_id_subgroup_" #tstr "_f16_ocp_f16acc", matmul_id_subgroup_##tstr##_f16_ocp_f16acc_cm1_len, matmul_id_subgroup_##tstr##_f16_ocp_f16acc_cm1_data, माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती); \
            } \
            अगर (device->coopmat_acc_f32_support) { \
                cm1_create({TYPE, GGML_TYPE_F32, true, false}, tc_mmq_id, "matmul_id_subgroup_" #tstr "_f32_ocp",        matmul_id_subgroup_##tstr##_f32_ocp_cm1_len,        matmul_id_subgroup_##tstr##_f32_ocp_cm1_data,        माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती); \
                cm1_create({TYPE, GGML_TYPE_F16, true, false}, tc_mmq_id, "matmul_id_subgroup_" #tstr "_f16_ocp",        matmul_id_subgroup_##tstr##_f16_ocp_cm1_len,        matmul_id_subgroup_##tstr##_f16_ocp_cm1_data,        माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती); \
            }
            FOR_EACH_LUT_FP4_TYPE(X_CM1_ID_OCP)
#अघोषित X_CM1_ID_OCP
        } अन्यथा
#पूर्ण_अगर
        {
            FOR_EACH_LUT_FP4_TYPE(X_CM1_ID)
        }
#अघोषित X_CM1_ID

        अगर (device->coopmat_int_support && (rdna3 || rdna4)) {
            cm1_create_mmq({GGML_TYPE_Q4_0,   GGML_TYPE_Q8_1, true, false}, tc_mmq_cm1_int,   "matmul_id_subgroup_q4_0_q8_1",   matmul_id_subgroup_q4_0_q8_1_cm1_len,   matmul_id_subgroup_q4_0_q8_1_cm1_तथ्य,   माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती);
            cm1_create_mmq({GGML_TYPE_Q4_1,   GGML_TYPE_Q8_1, true, false}, tc_mmq_cm1_int,   "matmul_id_subgroup_q4_1_q8_1",   matmul_id_subgroup_q4_1_q8_1_cm1_len,   matmul_id_subgroup_q4_1_q8_1_cm1_तथ्य,   माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती);
            cm1_create_mmq({GGML_TYPE_Q5_0,   GGML_TYPE_Q8_1, true, false}, tc_mmq_cm1_int,   "matmul_id_subgroup_q5_0_q8_1",   matmul_id_subgroup_q5_0_q8_1_cm1_len,   matmul_id_subgroup_q5_0_q8_1_cm1_तथ्य,   माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती);
            cm1_create_mmq({GGML_TYPE_Q5_1,   GGML_TYPE_Q8_1, true, false}, tc_mmq_cm1_int,   "matmul_id_subgroup_q5_1_q8_1",   matmul_id_subgroup_q5_1_q8_1_cm1_len,   matmul_id_subgroup_q5_1_q8_1_cm1_तथ्य,   माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती);
            cm1_create_mmq({GGML_TYPE_Q8_0,   GGML_TYPE_Q8_1, true, false}, tc_mmq_cm1_int,   "matmul_id_subgroup_q8_0_q8_1",   matmul_id_subgroup_q8_0_q8_1_cm1_len,   matmul_id_subgroup_q8_0_q8_1_cm1_तथ्य,   माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती);
            cm1_create_mmq({GGML_TYPE_IQ4_NL, GGML_TYPE_Q8_1, true, false}, tc_mmq_cm1_int,   "matmul_id_subgroup_iq4_nl_q8_1", matmul_id_subgroup_iq4_nl_q8_1_cm1_len, matmul_id_subgroup_iq4_nl_q8_1_cm1_तथ्य, माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती);
            cm1_create_mmq({GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_1, true, false}, tc_mmq_cm1_int,   "matmul_id_subgroup_iq4_xs_q8_1", matmul_id_subgroup_iq4_xs_q8_1_cm1_len, matmul_id_subgroup_iq4_xs_q8_1_cm1_तथ्य, माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती);
            cm1_create_mmq({GGML_TYPE_MXFP4,  GGML_TYPE_Q8_1, true, false}, tc_mmq_cm1_int,   "matmul_id_subgroup_mxfp4_q8_1",  matmul_id_subgroup_mxfp4_q8_1_cm1_len,  matmul_id_subgroup_mxfp4_q8_1_cm1_तथ्य,  माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती);
            cm1_create_mmq({GGML_TYPE_Q3_K,   GGML_TYPE_Q8_1, true, false}, tc_mmq_cm1_int_k, "matmul_id_subgroup_q3_k_q8_1",   matmul_id_subgroup_q3_k_q8_1_cm1_len,   matmul_id_subgroup_q3_k_q8_1_cm1_तथ्य,   माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती);
            cm1_create_mmq({GGML_TYPE_Q4_K,   GGML_TYPE_Q8_1, true, false}, tc_mmq_cm1_int,   "matmul_id_subgroup_q4_k_q8_1",   matmul_id_subgroup_q4_k_q8_1_cm1_len,   matmul_id_subgroup_q4_k_q8_1_cm1_तथ्य,   माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती);
            cm1_create_mmq({GGML_TYPE_Q5_K,   GGML_TYPE_Q8_1, true, false}, tc_mmq_cm1_int,   "matmul_id_subgroup_q5_k_q8_1",   matmul_id_subgroup_q5_k_q8_1_cm1_len,   matmul_id_subgroup_q5_k_q8_1_cm1_तथ्य,   माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती);
            cm1_create_mmq({GGML_TYPE_Q6_K,   GGML_TYPE_Q8_1, true, false}, tc_mmq_cm1_int_k, "matmul_id_subgroup_q6_k_q8_1",   matmul_id_subgroup_q6_k_q8_1_cm1_len,   matmul_id_subgroup_q6_k_q8_1_cm1_तथ्य,   माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती);
            अगर (!rdna4) { cm1_create_mmq({GGML_TYPE_NVFP4, GGML_TYPE_Q8_1, true, false}, tc_mmq_cm1_int_k, "matmul_id_subgroup_nvfp4_q8_1",  matmul_id_subgroup_nvfp4_q8_1_cm1_len,  matmul_id_subgroup_nvfp4_q8_1_cm1_तथ्य,  माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती); }
        }
    } अन्यथा
#पूर्ण_अगर  // defined(VK_KHR_cooperative_matrix) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    {
        // Helper for subgroup path with dot2 selection and filtering
        स्वतः sg_create = [&](vk_matmul_pipeline_कुंजी कुंजी, स्थिर std::vector<vk_tile_config>& tc_base,
                             स्थिर std::string& name, माप_प्रकार len, स्थिर व्योम* तथ्य, uint32_t pc_आकार, uint32_t pc,
                             uint32_t rsgs = 0) {
            स्वतः tc = filter_tc(tc_base, कुंजी.type_a, कुंजी.mul_mat_id);
            अगर (!tc.empty()) create_mm_pipelines(कुंजी, tc, name, len, तथ्य, pc_आकार, pc,
                [&](स्थिर std::vector<uint32_t>& wt, bool a) { वापस ggml_vk_mul_mm_spec(wt, a); },
                false, rsgs > 0, rsgs);
        };
        स्वतः sg_create_quant = [&](vk_matmul_pipeline_कुंजी कुंजी, स्थिर std::vector<vk_tile_config>& tc_base,
                                   स्थिर std::string& name, माप_प्रकार len, स्थिर व्योम* तथ्य, uint32_t pc_आकार, uint32_t pc,
                                   uint32_t rsgs = 0) {
            स्वतः tc = filter_tc(tc_base, कुंजी.type_a, कुंजी.mul_mat_id);
            अगर (!tc.empty()) {
                spec_fn_t qs = [&, type_a=कुंजी.type_a](स्थिर std::vector<uint32_t>& wt, bool a) { वापस ggml_vk_mul_mm_spec_quant(wt, a, (uint32_t)type_a); };
                create_mm_pipelines(कुंजी, tc, name, len, तथ्य, pc_आकार, pc, qs, false, rsgs > 0, rsgs);
            }
        };
#अगर defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
        स्वतः sg_create_mmq = [&](vk_matmul_pipeline_कुंजी कुंजी, स्थिर std::vector<vk_tile_config>& tc_base,
                                 स्थिर std::string& name, माप_प्रकार len, स्थिर व्योम* तथ्य, uint32_t pc_आकार, uint32_t pc,
                                 uint32_t rsgs = 0) {
            स्वतः tc = filter_tc(tc_base, कुंजी.type_a, कुंजी.mul_mat_id, true);
            अगर (!tc.empty()) {
                spec_fn_t identity = [](स्थिर std::vector<uint32_t>& wt, bool) { वापस wt; };
                create_mm_pipelines(कुंजी, tc, name, len, तथ्य, pc_आकार, pc, identity, false, rsgs > 0, rsgs, false);
            }
        };
#पूर्ण_अगर

        std::vector<vk_tile_config> tc_id = {{s_warptile_id, s_wg_denoms, s_align}, {m_warptile_id, m_wg_denoms, m_align}, {l_warptile_id, l_wg_denoms, l_align}};
        std::vector<vk_tile_config> tc_mmqid = {{s_warptile_mmqid, s_mmq_wg_denoms, s_align}, {m_warptile_mmqid, m_mmq_wg_denoms, m_align}, {l_warptile_mmqid, l_mmq_wg_denoms, l_align}};

        अगर (device->fp16) {
            // FP16 subgroup path - with dot2 runtime selection
            #घोषणा SPV_DOT2(NAME) (device->dot2_f16 ? NAME ## _dot2_len : NAME ## _len), (device->dot2_f16 ? NAME ## dot2_तथ्य : NAME ## तथ्य_2)
            #घोषणा SPV_DOT2_F16ACC(NAME) (device->dot2_f16 ? NAME ## _dot2_f16acc_len : NAME ## _f16acc_len), (device->dot2_f16 ? NAME ## dot2_f16acc_तथ्य : NAME ## f16acc_तथ्य)

            sg_create({GGML_TYPE_F32, GGML_TYPE_F32, false, false}, tc_mm, "matmul_f32_f32", SPV_DOT2(matmul_f32_f32), माप(vk_mat_mat_धकेलो_constants), 3);
            sg_create({GGML_TYPE_F32, GGML_TYPE_F16, false, false}, tc_mm, "matmul_f32_f16", SPV_DOT2(matmul_f32_f16), माप(vk_mat_mat_धकेलो_constants), 3);
            sg_create({GGML_TYPE_F16, GGML_TYPE_F16, false, true},  tc_mm, "matmul_f16_f16acc",     SPV_DOT2_F16ACC(matmul_f16),     माप(vk_mat_mat_धकेलो_constants), 3);
            sg_create({GGML_TYPE_F16, GGML_TYPE_F16, false, false}, tc_mm, "matmul_f16",            SPV_DOT2(matmul_f16),            माप(vk_mat_mat_धकेलो_constants), 3);
            sg_create({GGML_TYPE_F16, GGML_TYPE_F32, false, true},  tc_mm, "matmul_f16_f32_f16acc", SPV_DOT2_F16ACC(matmul_f16_f32), माप(vk_mat_mat_धकेलो_constants), 3);
            sg_create({GGML_TYPE_F16, GGML_TYPE_F32, false, false}, tc_mm, "matmul_f16_f32",        SPV_DOT2(matmul_f16_f32),        माप(vk_mat_mat_धकेलो_constants), 3);
            // BF16 - no dot2
            sg_create({GGML_TYPE_BF16, GGML_TYPE_BF16, false, false}, tc_mm, "matmul_bf16", matmul_bf16_len, matmul_bf16_तथ्य, माप(vk_mat_mat_धकेलो_constants), 3);

            क्रम (स्थिर स्वतः type : non_lut_quant_types) {
                sg_create_quant({type, GGML_TYPE_F32, false, true},  tc_mmq, "matmul_quant_f32_f16acc", SPV_DOT2_F16ACC(matmul_quant_f32), माप(vk_mat_mat_धकेलो_constants), 3);
                sg_create_quant({type, GGML_TYPE_F32, false, false}, tc_mmq, "matmul_quant_f32",        SPV_DOT2(matmul_quant_f32),        माप(vk_mat_mat_धकेलो_constants), 3);
            }
    #घोषणा X_SG(TYPE, tstr) \
            sg_create({TYPE, GGML_TYPE_F32, false, true},  tc_mmq, "matmul_" #tstr "_f32_f16acc", SPV_DOT2_F16ACC(matmul_##tstr##_f32), माप(vk_mat_mat_धकेलो_constants), 3); \
            sg_create({TYPE, GGML_TYPE_F32, false, false}, tc_mmq, "matmul_" #tstr "_f32",        SPV_DOT2(matmul_##tstr##_f32),        माप(vk_mat_mat_धकेलो_constants), 3);
            FOR_EACH_LUT_TYPE(X_SG)
#अघोषित X_SG

#अगर defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
            अगर (device->integer_dot_product) {
                std::vector<vk_tile_config> tc_mmq_int = {{s_warptile_mmq_int, s_mmq_wg_denoms, s_align}, {m_warptile_mmq_int, m_mmq_wg_denoms, m_align}, {l_warptile_mmq_int, l_mmq_wg_denoms, l_align}};
                std::vector<vk_tile_config> tc_mmq_int_k = {{s_warptile_mmq_int_k, s_mmq_wg_denoms, s_align}, {m_warptile_mmq_int_k, m_mmq_wg_denoms, m_align}, {l_warptile_mmq_int_k, l_mmq_wg_denoms, l_align}};
                sg_create_mmq({GGML_TYPE_Q2_0, GGML_TYPE_Q8_1, false, false}, tc_mmq_int, "matmul_q2_0_q8_1", matmul_q2_0_q8_1_len, matmul_q2_0_q8_1_तथ्य, माप(vk_mat_mat_धकेलो_constants), 3);
                sg_create_mmq({GGML_TYPE_Q4_0, GGML_TYPE_Q8_1, false, false}, tc_mmq_int, "matmul_q4_0_q8_1", matmul_q4_0_q8_1_len, matmul_q4_0_q8_1_तथ्य, माप(vk_mat_mat_धकेलो_constants), 3);
                sg_create_mmq({GGML_TYPE_Q4_1, GGML_TYPE_Q8_1, false, false}, tc_mmq_int, "matmul_q4_1_q8_1", matmul_q4_1_q8_1_len, matmul_q4_1_q8_1_तथ्य, माप(vk_mat_mat_धकेलो_constants), 3);
                sg_create_mmq({GGML_TYPE_Q5_0, GGML_TYPE_Q8_1, false, false}, tc_mmq_int, "matmul_q5_0_q8_1", matmul_q5_0_q8_1_len, matmul_q5_0_q8_1_तथ्य, माप(vk_mat_mat_धकेलो_constants), 3);
                sg_create_mmq({GGML_TYPE_Q5_1, GGML_TYPE_Q8_1, false, false}, tc_mmq_int, "matmul_q5_1_q8_1", matmul_q5_1_q8_1_len, matmul_q5_1_q8_1_तथ्य, माप(vk_mat_mat_धकेलो_constants), 3);
                sg_create_mmq({GGML_TYPE_Q8_0, GGML_TYPE_Q8_1, false, false}, tc_mmq_int, "matmul_q8_0_q8_1", matmul_q8_0_q8_1_len, matmul_q8_0_q8_1_तथ्य, माप(vk_mat_mat_धकेलो_constants), 3);
                sg_create_mmq({GGML_TYPE_MXFP4, GGML_TYPE_Q8_1, false, false}, tc_mmq_int, "matmul_mxfp4_q8_1", matmul_mxfp4_q8_1_len, matmul_mxfp4_q8_1_तथ्य, माप(vk_mat_mat_धकेलो_constants), 3);
                sg_create_mmq({GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_1, false, false}, tc_mmq_int, "matmul_iq4_xs_q8_1", matmul_iq4_xs_q8_1_len, matmul_iq4_xs_q8_1_तथ्य, माप(vk_mat_mat_धकेलो_constants), 3);
                sg_create_mmq({GGML_TYPE_Q2_K, GGML_TYPE_Q8_1, false, false}, tc_mmq_int_k, "matmul_q2_k_q8_1", matmul_q2_k_q8_1_len, matmul_q2_k_q8_1_तथ्य, माप(vk_mat_mat_धकेलो_constants), 3);
                sg_create_mmq({GGML_TYPE_Q3_K, GGML_TYPE_Q8_1, false, false}, tc_mmq_int_k, "matmul_q3_k_q8_1", matmul_q3_k_q8_1_len, matmul_q3_k_q8_1_तथ्य, माप(vk_mat_mat_धकेलो_constants), 3);
                sg_create_mmq({GGML_TYPE_Q4_K, GGML_TYPE_Q8_1, false, false}, tc_mmq_int_k, "matmul_q4_k_q8_1", matmul_q4_k_q8_1_len, matmul_q4_k_q8_1_तथ्य, माप(vk_mat_mat_धकेलो_constants), 3);
                sg_create_mmq({GGML_TYPE_Q5_K, GGML_TYPE_Q8_1, false, false}, tc_mmq_int_k, "matmul_q5_k_q8_1", matmul_q5_k_q8_1_len, matmul_q5_k_q8_1_तथ्य, माप(vk_mat_mat_धकेलो_constants), 3);
                sg_create_mmq({GGML_TYPE_Q6_K, GGML_TYPE_Q8_1, false, false}, tc_mmq_int_k, "matmul_q6_k_q8_1", matmul_q6_k_q8_1_len, matmul_q6_k_q8_1_तथ्य, माप(vk_mat_mat_धकेलो_constants), 3);
                sg_create_mmq({GGML_TYPE_IQ3_S, GGML_TYPE_Q8_1, false, false}, tc_mmq_int_k, "matmul_iq3_s_q8_1", matmul_iq3_s_q8_1_len, matmul_iq3_s_q8_1_तथ्य, माप(vk_mat_mat_धकेलो_constants), 3);
            }
#पूर्ण_अगर

            अगर (device->subgroup_ballot && device->subgroup_require_full_support && subgroup_min_आकार_16) {
                sg_create({GGML_TYPE_F32, GGML_TYPE_F32, true, false}, tc_id, "matmul_id_subgroup_f32_f32", SPV_DOT2(matmul_id_subgroup_f32_f32), माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती, mul_mat_subgroup_आकार_16);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F16, true, true},  tc_id, "matmul_id_subgroup_f16_f16acc",     SPV_DOT2_F16ACC(matmul_id_subgroup_f16),     माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती, mul_mat_subgroup_आकार_16);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F16, true, false}, tc_id, "matmul_id_subgroup_f16",            SPV_DOT2(matmul_id_subgroup_f16),            माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती, mul_mat_subgroup_आकार_16);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F32, true, true},  tc_id, "matmul_id_subgroup_f16_f32_f16acc", SPV_DOT2_F16ACC(matmul_id_subgroup_f16_f32), माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती, mul_mat_subgroup_आकार_16);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F32, true, false}, tc_id, "matmul_id_subgroup_f16_f32",        SPV_DOT2(matmul_id_subgroup_f16_f32),        माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती, mul_mat_subgroup_आकार_16);
                // BF16 id - no dot2
                sg_create({GGML_TYPE_BF16, GGML_TYPE_BF16, true, false}, tc_id, "matmul_id_subgroup_bf16", matmul_id_subgroup_bf16_len, matmul_id_subgroup_bf16_तथ्य, माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती, mul_mat_subgroup_आकार_16);
                क्रम (स्थिर स्वतः type : non_lut_quant_types) {
                    sg_create_quant({type, GGML_TYPE_F32, true, true},  tc_mmqid, "matmul_id_subgroup_quant_f32_f16acc", SPV_DOT2_F16ACC(matmul_id_subgroup_quant_f32), माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती, mul_mat_subgroup_आकार);
                    sg_create_quant({type, GGML_TYPE_F32, true, false}, tc_mmqid, "matmul_id_subgroup_quant_f32",        SPV_DOT2(matmul_id_subgroup_quant_f32),        माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती, mul_mat_subgroup_आकार);
                }
        #घोषणा X_SG_ID_SUB(TYPE, tstr) \
                sg_create({TYPE, GGML_TYPE_F32, true, true},  tc_mmqid, "matmul_id_subgroup_" #tstr "_f32_f16acc", SPV_DOT2_F16ACC(matmul_id_subgroup_##tstr##_f32), माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती, mul_mat_subgroup_आकार); \
                sg_create({TYPE, GGML_TYPE_F32, true, false}, tc_mmqid, "matmul_id_subgroup_" #tstr "_f32",        SPV_DOT2(matmul_id_subgroup_##tstr##_f32),        माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती, mul_mat_subgroup_आकार);
                FOR_EACH_LUT_TYPE(X_SG_ID_SUB)
#अघोषित X_SG_ID_SUB
#अगर defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
                अगर (device->integer_dot_product) {
                    std::vector<vk_tile_config> tc_mmqid_int = {{s_warptile_mmqid_int, s_mmq_wg_denoms, s_align}, {m_warptile_mmqid_int, m_mmq_wg_denoms, m_align}, {l_warptile_mmqid_int, l_mmq_wg_denoms, l_align}};
                    std::vector<vk_tile_config> tc_mmqid_int_k = {{s_warptile_mmqid_int_k, s_mmq_wg_denoms, s_align}, {m_warptile_mmqid_int_k, m_mmq_wg_denoms, m_align}, {l_warptile_mmqid_int_k, l_mmq_wg_denoms, l_align}};
                    sg_create_mmq({GGML_TYPE_Q2_0, GGML_TYPE_Q8_1, true, false}, tc_mmqid_int, "matmul_id_subgroup_q2_0_q8_1", matmul_id_subgroup_q2_0_q8_1_len, matmul_id_subgroup_q2_0_q8_1_तथ्य, माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती, mul_mat_subgroup_आकार);
                    sg_create_mmq({GGML_TYPE_Q4_0, GGML_TYPE_Q8_1, true, false}, tc_mmqid_int, "matmul_id_subgroup_q4_0_q8_1", matmul_id_subgroup_q4_0_q8_1_len, matmul_id_subgroup_q4_0_q8_1_तथ्य, माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती, mul_mat_subgroup_आकार);
                    sg_create_mmq({GGML_TYPE_Q4_1, GGML_TYPE_Q8_1, true, false}, tc_mmqid_int, "matmul_id_subgroup_q4_1_q8_1", matmul_id_subgroup_q4_1_q8_1_len, matmul_id_subgroup_q4_1_q8_1_तथ्य, माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती, mul_mat_subgroup_आकार);
                    sg_create_mmq({GGML_TYPE_Q5_0, GGML_TYPE_Q8_1, true, false}, tc_mmqid_int, "matmul_id_subgroup_q5_0_q8_1", matmul_id_subgroup_q5_0_q8_1_len, matmul_id_subgroup_q5_0_q8_1_तथ्य, माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती, mul_mat_subgroup_आकार);
                    sg_create_mmq({GGML_TYPE_Q5_1, GGML_TYPE_Q8_1, true, false}, tc_mmqid_int, "matmul_id_subgroup_q5_1_q8_1", matmul_id_subgroup_q5_1_q8_1_len, matmul_id_subgroup_q5_1_q8_1_तथ्य, माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती, mul_mat_subgroup_आकार);
                    sg_create_mmq({GGML_TYPE_Q8_0, GGML_TYPE_Q8_1, true, false}, tc_mmqid_int, "matmul_id_subgroup_q8_0_q8_1", matmul_id_subgroup_q8_0_q8_1_len, matmul_id_subgroup_q8_0_q8_1_तथ्य, माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती, mul_mat_subgroup_आकार);
                    sg_create_mmq({GGML_TYPE_MXFP4, GGML_TYPE_Q8_1, true, false}, tc_mmqid_int, "matmul_id_subgroup_mxfp4_q8_1", matmul_id_subgroup_mxfp4_q8_1_len, matmul_id_subgroup_mxfp4_q8_1_तथ्य, माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती, mul_mat_subgroup_आकार);
                    sg_create_mmq({GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_1, true, false}, tc_mmqid_int, "matmul_id_subgroup_iq4_xs_q8_1", matmul_id_subgroup_iq4_xs_q8_1_len, matmul_id_subgroup_iq4_xs_q8_1_तथ्य, माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती, mul_mat_subgroup_आकार);
                    sg_create_mmq({GGML_TYPE_Q2_K, GGML_TYPE_Q8_1, true, false}, tc_mmqid_int_k, "matmul_id_subgroup_q2_k_q8_1", matmul_id_subgroup_q2_k_q8_1_len, matmul_id_subgroup_q2_k_q8_1_तथ्य, माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती, mul_mat_subgroup_आकार_16);
                    sg_create_mmq({GGML_TYPE_Q3_K, GGML_TYPE_Q8_1, true, false}, tc_mmqid_int_k, "matmul_id_subgroup_q3_k_q8_1", matmul_id_subgroup_q3_k_q8_1_len, matmul_id_subgroup_q3_k_q8_1_तथ्य, माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती, mul_mat_subgroup_आकार_16);
                    sg_create_mmq({GGML_TYPE_Q4_K, GGML_TYPE_Q8_1, true, false}, tc_mmqid_int_k, "matmul_id_subgroup_q4_k_q8_1", matmul_id_subgroup_q4_k_q8_1_len, matmul_id_subgroup_q4_k_q8_1_तथ्य, माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती, mul_mat_subgroup_आकार_16);
                    sg_create_mmq({GGML_TYPE_Q5_K, GGML_TYPE_Q8_1, true, false}, tc_mmqid_int_k, "matmul_id_subgroup_q5_k_q8_1", matmul_id_subgroup_q5_k_q8_1_len, matmul_id_subgroup_q5_k_q8_1_तथ्य, माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती, mul_mat_subgroup_आकार_16);
                    sg_create_mmq({GGML_TYPE_Q6_K, GGML_TYPE_Q8_1, true, false}, tc_mmqid_int_k, "matmul_id_subgroup_q6_k_q8_1", matmul_id_subgroup_q6_k_q8_1_len, matmul_id_subgroup_q6_k_q8_1_तथ्य, माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती, mul_mat_subgroup_आकार_16);
                    sg_create_mmq({GGML_TYPE_IQ3_S, GGML_TYPE_Q8_1, true, false}, tc_mmqid_int_k, "matmul_id_subgroup_iq3_s_q8_1", matmul_id_subgroup_iq3_s_q8_1_len, matmul_id_subgroup_iq3_s_q8_1_तथ्य, माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती, mul_mat_subgroup_आकार_16);
                }
#पूर्ण_अगर
            } अन्यथा {
                sg_create({GGML_TYPE_F32, GGML_TYPE_F32, true, false}, tc_mm, "matmul_id_f32_f32", SPV_DOT2(matmul_id_f32_f32), माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F16, true, true},  tc_mm, "matmul_id_f16_f16acc",     SPV_DOT2_F16ACC(matmul_id_f16),     माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F16, true, false}, tc_mm, "matmul_id_f16",            SPV_DOT2(matmul_id_f16),            माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F32, true, true},  tc_mm, "matmul_id_f16_f32_f16acc", SPV_DOT2_F16ACC(matmul_id_f16_f32), माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F32, true, false}, tc_mm, "matmul_id_f16_f32",        SPV_DOT2(matmul_id_f16_f32),        माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती);
                // BF16 id - no dot2
                sg_create({GGML_TYPE_BF16, GGML_TYPE_BF16, true, false}, tc_mm, "matmul_id_bf16", matmul_id_bf16_len, matmul_id_bf16_तथ्य, माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती);
                क्रम (स्थिर स्वतः type : non_lut_quant_types) {
                    sg_create_quant({type, GGML_TYPE_F32, true, true},  tc_mmqid, "matmul_id_quant_f32_f16acc", SPV_DOT2_F16ACC(matmul_id_quant_f32), माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती);
                    sg_create_quant({type, GGML_TYPE_F32, true, false}, tc_mmqid, "matmul_id_quant_f32",        SPV_DOT2(matmul_id_quant_f32),        माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती);
                }
        #घोषणा X_SG_ID(TYPE, tstr) \
                sg_create({TYPE, GGML_TYPE_F32, true, true},  tc_mmqid, "matmul_id_" #tstr "_f32_f16acc", SPV_DOT2_F16ACC(matmul_id_##tstr##_f32), माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती); \
                sg_create({TYPE, GGML_TYPE_F32, true, false}, tc_mmqid, "matmul_id_" #tstr "_f32",        SPV_DOT2(matmul_id_##tstr##_f32),        माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती);
                FOR_EACH_LUT_TYPE(X_SG_ID)
#अघोषित X_SG_ID
#अगर defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
                अगर (device->integer_dot_product) {
                    std::vector<vk_tile_config> tc_mmqid_int = {{s_warptile_mmqid_int, s_mmq_wg_denoms, s_align}, {m_warptile_mmqid_int, m_mmq_wg_denoms, m_align}, {l_warptile_mmqid_int, l_mmq_wg_denoms, l_align}};
                    std::vector<vk_tile_config> tc_mmqid_int_k = {{s_warptile_mmqid_int_k, s_mmq_wg_denoms, s_align}, {m_warptile_mmqid_int_k, m_mmq_wg_denoms, m_align}, {l_warptile_mmqid_int_k, l_mmq_wg_denoms, l_align}};
                    sg_create_mmq({GGML_TYPE_Q2_0, GGML_TYPE_Q8_1, true, false}, tc_mmqid_int, "matmul_id_q2_0_q8_1", matmul_id_q2_0_q8_1_len, matmul_id_q2_0_q8_1_तथ्य, माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती);
                    sg_create_mmq({GGML_TYPE_Q4_0, GGML_TYPE_Q8_1, true, false}, tc_mmqid_int, "matmul_id_q4_0_q8_1", matmul_id_q4_0_q8_1_len, matmul_id_q4_0_q8_1_तथ्य, माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती);
                    sg_create_mmq({GGML_TYPE_Q4_1, GGML_TYPE_Q8_1, true, false}, tc_mmqid_int, "matmul_id_q4_1_q8_1", matmul_id_q4_1_q8_1_len, matmul_id_q4_1_q8_1_तथ्य, माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती);
                    sg_create_mmq({GGML_TYPE_Q5_0, GGML_TYPE_Q8_1, true, false}, tc_mmqid_int, "matmul_id_q5_0_q8_1", matmul_id_q5_0_q8_1_len, matmul_id_q5_0_q8_1_तथ्य, माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती);
                    sg_create_mmq({GGML_TYPE_Q5_1, GGML_TYPE_Q8_1, true, false}, tc_mmqid_int, "matmul_id_q5_1_q8_1", matmul_id_q5_1_q8_1_len, matmul_id_q5_1_q8_1_तथ्य, माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती);
                    sg_create_mmq({GGML_TYPE_Q8_0, GGML_TYPE_Q8_1, true, false}, tc_mmqid_int, "matmul_id_q8_0_q8_1", matmul_id_q8_0_q8_1_len, matmul_id_q8_0_q8_1_तथ्य, माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती);
                    sg_create_mmq({GGML_TYPE_MXFP4, GGML_TYPE_Q8_1, true, false}, tc_mmqid_int, "matmul_id_mxfp4_q8_1", matmul_id_mxfp4_q8_1_len, matmul_id_mxfp4_q8_1_तथ्य, माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती);
                    sg_create_mmq({GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_1, true, false}, tc_mmqid_int, "matmul_id_iq4_xs_q8_1", matmul_id_iq4_xs_q8_1_len, matmul_id_iq4_xs_q8_1_तथ्य, माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती);
                    sg_create_mmq({GGML_TYPE_Q2_K, GGML_TYPE_Q8_1, true, false}, tc_mmqid_int_k, "matmul_id_q2_k_q8_1", matmul_id_q2_k_q8_1_len, matmul_id_q2_k_q8_1_तथ्य, माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती);
                    sg_create_mmq({GGML_TYPE_Q3_K, GGML_TYPE_Q8_1, true, false}, tc_mmqid_int_k, "matmul_id_q3_k_q8_1", matmul_id_q3_k_q8_1_len, matmul_id_q3_k_q8_1_तथ्य, माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती);
                    sg_create_mmq({GGML_TYPE_Q4_K, GGML_TYPE_Q8_1, true, false}, tc_mmqid_int_k, "matmul_id_q4_k_q8_1", matmul_id_q4_k_q8_1_len, matmul_id_q4_k_q8_1_तथ्य, माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती);
                    sg_create_mmq({GGML_TYPE_Q5_K, GGML_TYPE_Q8_1, true, false}, tc_mmqid_int_k, "matmul_id_q5_k_q8_1", matmul_id_q5_k_q8_1_len, matmul_id_q5_k_q8_1_तथ्य, माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती);
                    sg_create_mmq({GGML_TYPE_Q6_K, GGML_TYPE_Q8_1, true, false}, tc_mmqid_int_k, "matmul_id_q6_k_q8_1", matmul_id_q6_k_q8_1_len, matmul_id_q6_k_q8_1_तथ्य, माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती);
                    sg_create_mmq({GGML_TYPE_IQ3_S, GGML_TYPE_Q8_1, true, false}, tc_mmqid_int_k, "matmul_id_iq3_s_q8_1", matmul_id_iq3_s_q8_1_len, matmul_id_iq3_s_q8_1_तथ्य, माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती);
                }
#पूर्ण_अगर
            }
            #अघोषित SPV_DOT2
            #अघोषित SPV_DOT2_F16ACC
        } अन्यथा {
            // FP32-only fallback path
            sg_create({GGML_TYPE_F32, GGML_TYPE_F32, false, false}, tc_mm, "matmul_f32_f32", matmul_f32_f32_fp32_len, matmul_f32_f32_fp32_तथ्य, माप(vk_mat_mat_धकेलो_constants), 3);
            sg_create({GGML_TYPE_F32, GGML_TYPE_F16, false, false}, tc_mm, "matmul_f32_f16", matmul_f32_f16_fp32_len, matmul_f32_f16_fp32_तथ्य, माप(vk_mat_mat_धकेलो_constants), 3);
            sg_create({GGML_TYPE_F16, GGML_TYPE_F16, false, false}, tc_mm, "matmul_f16",     matmul_f16_fp32_len,     matmul_f16_fp32_तथ्य,     माप(vk_mat_mat_धकेलो_constants), 3);
            sg_create({GGML_TYPE_F16, GGML_TYPE_F32, false, false}, tc_mm, "matmul_f16_f32", matmul_f16_f32_fp32_len, matmul_f16_f32_fp32_तथ्य, माप(vk_mat_mat_धकेलो_constants), 3);
            sg_create({GGML_TYPE_BF16, GGML_TYPE_BF16, false, false}, tc_mm, "matmul_bf16",  matmul_bf16_fp32_len,    matmul_bf16_fp32_तथ्य,    माप(vk_mat_mat_धकेलो_constants), 3);

            क्रम (स्थिर स्वतः type : non_lut_quant_types) {
                sg_create_quant({type, GGML_TYPE_F32, false, false}, tc_mmq, "matmul_quant_f32", matmul_quant_f32_fp32_len, matmul_quant_f32_fp32_तथ्य, माप(vk_mat_mat_धकेलो_constants), 3);
            }
    #घोषणा X_SG_FP32(TYPE, tstr) \
            sg_create({TYPE, GGML_TYPE_F32, false, false}, tc_mmq, "matmul_" #tstr "_f32", matmul_##tstr##_f32_fp32_len, matmul_##tstr##_f32_fp32_data, माप(vk_mat_mat_धकेलो_constants), 3);
            FOR_EACH_LUT_TYPE(X_SG_FP32)
#अघोषित X_SG_FP32

#अगर defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
            अगर (device->integer_dot_product) {
                std::vector<vk_tile_config> tc_mmq_int = {{s_warptile_mmq_int, s_mmq_wg_denoms, s_align}, {m_warptile_mmq_int, m_mmq_wg_denoms, m_align}, {l_warptile_mmq_int, l_mmq_wg_denoms, l_align}};
                std::vector<vk_tile_config> tc_mmq_int_k = {{s_warptile_mmq_int_k, s_mmq_wg_denoms, s_align}, {m_warptile_mmq_int_k, m_mmq_wg_denoms, m_align}, {l_warptile_mmq_int_k, l_mmq_wg_denoms, l_align}};
                sg_create_mmq({GGML_TYPE_Q2_0, GGML_TYPE_Q8_1, false, false}, tc_mmq_int, "matmul_q2_0_q8_1", matmul_q2_0_q8_1_fp32_len, matmul_q2_0_q8_1_fp32_तथ्य, माप(vk_mat_mat_धकेलो_constants), 3);
                sg_create_mmq({GGML_TYPE_Q4_0, GGML_TYPE_Q8_1, false, false}, tc_mmq_int, "matmul_q4_0_q8_1", matmul_q4_0_q8_1_fp32_len, matmul_q4_0_q8_1_fp32_तथ्य, माप(vk_mat_mat_धकेलो_constants), 3);
                sg_create_mmq({GGML_TYPE_Q4_1, GGML_TYPE_Q8_1, false, false}, tc_mmq_int, "matmul_q4_1_q8_1", matmul_q4_1_q8_1_fp32_len, matmul_q4_1_q8_1_fp32_तथ्य, माप(vk_mat_mat_धकेलो_constants), 3);
                sg_create_mmq({GGML_TYPE_Q5_0, GGML_TYPE_Q8_1, false, false}, tc_mmq_int, "matmul_q5_0_q8_1", matmul_q5_0_q8_1_fp32_len, matmul_q5_0_q8_1_fp32_तथ्य, माप(vk_mat_mat_धकेलो_constants), 3);
                sg_create_mmq({GGML_TYPE_Q5_1, GGML_TYPE_Q8_1, false, false}, tc_mmq_int, "matmul_q5_1_q8_1", matmul_q5_1_q8_1_fp32_len, matmul_q5_1_q8_1_fp32_तथ्य, माप(vk_mat_mat_धकेलो_constants), 3);
                sg_create_mmq({GGML_TYPE_Q8_0, GGML_TYPE_Q8_1, false, false}, tc_mmq_int, "matmul_q8_0_q8_1", matmul_q8_0_q8_1_fp32_len, matmul_q8_0_q8_1_fp32_तथ्य, माप(vk_mat_mat_धकेलो_constants), 3);
                sg_create_mmq({GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_1, false, false}, tc_mmq_int, "matmul_iq4_xs_q8_1", matmul_iq4_xs_q8_1_fp32_len, matmul_iq4_xs_q8_1_fp32_तथ्य, माप(vk_mat_mat_धकेलो_constants), 3);
                sg_create_mmq({GGML_TYPE_Q2_K, GGML_TYPE_Q8_1, false, false}, tc_mmq_int_k, "matmul_q2_k_q8_1", matmul_q2_k_q8_1_fp32_len, matmul_q2_k_q8_1_fp32_तथ्य, माप(vk_mat_mat_धकेलो_constants), 3);
                sg_create_mmq({GGML_TYPE_Q3_K, GGML_TYPE_Q8_1, false, false}, tc_mmq_int_k, "matmul_q3_k_q8_1", matmul_q3_k_q8_1_fp32_len, matmul_q3_k_q8_1_fp32_तथ्य, माप(vk_mat_mat_धकेलो_constants), 3);
                sg_create_mmq({GGML_TYPE_Q4_K, GGML_TYPE_Q8_1, false, false}, tc_mmq_int_k, "matmul_q4_k_q8_1", matmul_q4_k_q8_1_fp32_len, matmul_q4_k_q8_1_fp32_तथ्य, माप(vk_mat_mat_धकेलो_constants), 3);
                sg_create_mmq({GGML_TYPE_Q5_K, GGML_TYPE_Q8_1, false, false}, tc_mmq_int_k, "matmul_q5_k_q8_1", matmul_q5_k_q8_1_fp32_len, matmul_q5_k_q8_1_fp32_तथ्य, माप(vk_mat_mat_धकेलो_constants), 3);
                sg_create_mmq({GGML_TYPE_Q6_K, GGML_TYPE_Q8_1, false, false}, tc_mmq_int_k, "matmul_q6_k_q8_1", matmul_q6_k_q8_1_fp32_len, matmul_q6_k_q8_1_fp32_तथ्य, माप(vk_mat_mat_धकेलो_constants), 3);
                sg_create_mmq({GGML_TYPE_IQ3_S, GGML_TYPE_Q8_1, false, false}, tc_mmq_int_k, "matmul_iq3_s_q8_1", matmul_iq3_s_q8_1_fp32_len, matmul_iq3_s_q8_1_fp32_तथ्य, माप(vk_mat_mat_धकेलो_constants), 3);
            }
#पूर्ण_अगर

            अगर (device->subgroup_ballot && device->subgroup_require_full_support && subgroup_min_आकार_16) {
                sg_create({GGML_TYPE_F32, GGML_TYPE_F32, true, false}, tc_id, "matmul_id_subgroup_f32_f32", matmul_id_subgroup_f32_f32_fp32_len, matmul_id_subgroup_f32_f32_fp32_तथ्य, माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती, mul_mat_subgroup_आकार_16);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F16, true, false}, tc_id, "matmul_id_subgroup_f16",     matmul_id_subgroup_f16_fp32_len,     matmul_id_subgroup_f16_fp32_तथ्य,     माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती, mul_mat_subgroup_आकार_16);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F32, true, false}, tc_id, "matmul_id_subgroup_f16_f32", matmul_id_subgroup_f16_f32_fp32_len, matmul_id_subgroup_f16_f32_fp32_तथ्य, माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती, mul_mat_subgroup_आकार_16);
                sg_create({GGML_TYPE_BF16, GGML_TYPE_BF16, true, false}, tc_id, "matmul_id_subgroup_bf16",  matmul_id_subgroup_bf16_fp32_len,    matmul_id_subgroup_bf16_fp32_तथ्य,    माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती, mul_mat_subgroup_आकार_16);
                क्रम (स्थिर स्वतः type : non_lut_quant_types) {
                    sg_create_quant({type, GGML_TYPE_F32, true, false}, tc_mmqid, "matmul_id_subgroup_quant_f32", matmul_id_subgroup_quant_f32_fp32_len, matmul_id_subgroup_quant_f32_fp32_तथ्य, माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती, mul_mat_subgroup_आकार);
                }
        #घोषणा X_SG_ID_SUB_FP32(TYPE, tstr) \
                sg_create({TYPE, GGML_TYPE_F32, true, false}, tc_mmqid, "matmul_id_subgroup_" #tstr "_f32", matmul_id_subgroup_##tstr##_f32_fp32_len, matmul_id_subgroup_##tstr##_f32_fp32_data, माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती, mul_mat_subgroup_आकार);
                FOR_EACH_LUT_TYPE(X_SG_ID_SUB_FP32)
#अघोषित X_SG_ID_SUB_FP32
            } अन्यथा {
                sg_create({GGML_TYPE_F32, GGML_TYPE_F32, true, false}, tc_mm, "matmul_id_f32_f32", matmul_id_f32_f32_fp32_len, matmul_id_f32_f32_fp32_तथ्य, माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F16, true, false}, tc_mm, "matmul_id_f16",     matmul_id_f16_fp32_len,     matmul_id_f16_fp32_तथ्य,     माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F32, true, false}, tc_mm, "matmul_id_f16_f32", matmul_id_f16_f32_fp32_len, matmul_id_f16_f32_fp32_तथ्य, माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती);
                sg_create({GGML_TYPE_BF16, GGML_TYPE_BF16, true, false}, tc_mm, "matmul_id_bf16",  matmul_id_bf16_fp32_len,    matmul_id_bf16_fp32_तथ्य,    माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती);
                क्रम (स्थिर स्वतः type : non_lut_quant_types) {
                    sg_create_quant({type, GGML_TYPE_F32, true, false}, tc_mmqid, "matmul_id_quant_f32", matmul_id_quant_f32_fp32_len, matmul_id_quant_f32_fp32_तथ्य, माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती);
                }
        #घोषणा X_SG_ID_FP32(TYPE, tstr) \
                sg_create({TYPE, GGML_TYPE_F32, true, false}, tc_mmqid, "matmul_id_" #tstr "_f32", matmul_id_##tstr##_f32_fp32_len, matmul_id_##tstr##_f32_fp32_data, माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती);
                FOR_EACH_LUT_TYPE(X_SG_ID_FP32)
#अघोषित X_SG_ID_FP32
            }
        }
    }
#अघोषित FOR_EACH_LUT_TYPE
#अघोषित FOR_EACH_LUT_TYPE_NONFP4
#अघोषित FOR_EACH_LUT_FP4_TYPE
    // BF16 fallback for coopmat devices without bf16 coopmat support
    अगर ((device->coopmat2 || device->coopmat_support)
#अगर defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        && !device->coopmat_bf16_support
#पूर्ण_अगर
        ) {
        स्थिर uint32_t s_warptile_wm_bf16 = device->subgroup_आकार == 8 ? 8 : 32;
        std::vector<vk_tile_config> tc_bf16_fb = {
            {{ subgroup_आकार_32, 32, 32, 16, s_warptile_wm_bf16, 32, 2, 2, 2, 1, subgroup_आकार_8 }, {32, 32, 1}, s_align},
            {{ 128, 64, 64, 16, mm_warp_8, 32, 2, 4, 2, 1, mm_warp_8 }, {64, 64, 1}, m_align},
            {{ 128, 128, 128, 16, mm_warp_8 * 2, 64, 2, 4, 4, 1, mm_warp_8 }, {128, 128, 1}, l_align},
        };
        स्वतः tc_bf16_filtered = filter_tc(tc_bf16_fb, GGML_TYPE_BF16, false);
        स्वतः tc_bf16_id_filtered = filter_tc(tc_bf16_fb, GGML_TYPE_BF16, true);
        spec_fn_t bf16_spec = [&](स्थिर std::vector<uint32_t>& wt, bool a) { वापस ggml_vk_mul_mm_spec(wt, a); };
        अगर (!tc_bf16_filtered.empty()) {
            create_mm_pipelines({GGML_TYPE_BF16, GGML_TYPE_BF16, false, false}, tc_bf16_filtered, "matmul_bf16", matmul_bf16_fp32_len, matmul_bf16_fp32_तथ्य, माप(vk_mat_mat_धकेलो_constants), 3, bf16_spec);
        }
        अगर (!tc_bf16_id_filtered.empty()) {
            create_mm_pipelines({GGML_TYPE_BF16, GGML_TYPE_BF16, true, false}, tc_bf16_id_filtered, "matmul_id_bf16", matmul_id_bf16_fp32_len, matmul_id_bf16_fp32_तथ्य, माप(vk_mat_mat_id_धकेलो_constants), mul_mat_id_param_गिनती, bf16_spec);
        }
    }

    // Set up tile selector functions
    अगर (device->coopmat2) {
        device->matmul_tile_selector = [](uint32_t m, uint32_t n, uint32_t /*k*/, uint32_t shader_core_गिनती,
                                          स्थिर std::vector<vk_matmul_pipeline_pair>& configs) -> uint32_t {
            अगर (configs.size() <= 1) वापस 0;
            uint32_t last = (uint32_t)configs.size() - 1;
            अगर (configs.size() == 2) {
                uint32_t crossover = configs[0].unaligned->wg_denoms[1];
                वापस (n > crossover) ? 1 : 0;
            }
            // 3+ configs: s=0, m=1, l=2
            स्थिर uint32_t tiles_l = CEIL_DIV(m, configs[last].unaligned->wg_denoms[0]) * CEIL_DIV(n, configs[last].unaligned->wg_denoms[1]);
            स्थिर uint32_t tiles_m = CEIL_DIV(m, configs[1].unaligned->wg_denoms[0]) * CEIL_DIV(n, configs[1].unaligned->wg_denoms[1]);
            uint32_t crossover_large = configs[1].unaligned->wg_denoms[1];
            bool prefer_large = tiles_m > shader_core_गिनती || tiles_l > shader_core_गिनती ||
                                (tiles_l <= shader_core_गिनती / 3 && tiles_m > shader_core_गिनती / 2);
            अगर (n > crossover_large && prefer_large) वापस last;
            uint32_t crossover_medium_m = configs[0].unaligned->wg_denoms[0];
            uint32_t crossover_medium_n = configs[0].unaligned->wg_denoms[1];
            अगर (m > crossover_medium_m && n > crossover_medium_n) वापस 1;
            वापस 0;
        };
        device->matmul_id_tile_selector = [](uint32_t /*m*/, uint32_t n, uint32_t /*k*/, uint32_t /*shader_core_count*/,
                                             स्थिर std::vector<vk_matmul_pipeline_pair>& configs) -> uint32_t {
            अगर (configs.size() <= 1) वापस 0;
            uint32_t last = (uint32_t)configs.size() - 1;
            अगर (configs.size() == 2) {
                uint32_t crossover = configs[0].unaligned->wg_denoms[1];
                वापस (n > crossover) ? 1 : 0;
            }
            uint32_t crossover_large = configs[1].unaligned->wg_denoms[1];
            अगर (n > crossover_large) वापस last;
            uint32_t crossover_medium = configs[0].unaligned->wg_denoms[1];
            अगर (n > crossover_medium) वापस 1;
            वापस 0;
        };
    } अन्यथा {
        device->matmul_tile_selector = [](uint32_t m, uint32_t n, uint32_t /*k*/, uint32_t /*shader_core_count*/,
                                          स्थिर std::vector<vk_matmul_pipeline_pair>& configs) -> uint32_t {
            अगर (configs.size() <= 1) वापस 0;
            अगर (m <= 32 || n <= 32) वापस 0;
            अगर (configs.size() == 2) वापस 1;
            अगर (m <= 64 || n <= 64) वापस 1;
            वापस (uint32_t)configs.size() - 1;
        };
        device->matmul_id_tile_selector = device->matmul_tile_selector;
    }

    // mul mat vec

    // the number of rows computed per shader depends on GPU model and quant
    uint32_t rm_stdq = 1;
    uint32_t rm_kq = 2;
    uint32_t rm_stdq_int = 1;
    uint32_t rm_kq_int = 1;
    स्वतः स्थिर &rm_iq_int = [](uint32_t i) { वापस i == 0 ? 8u : 4u; };
    अगर (device->vendor_id == VK_VENDOR_ID_AMD) {
        अगर (device->architecture == AMD_GCN) {
            rm_stdq = 2;
            rm_kq = 4;
            rm_stdq_int = 4;
        }
    } अन्यथा अगर (device->vendor_id == VK_VENDOR_ID_INTEL) {
        rm_stdq = 2;
        rm_stdq_int = 2;
    }
    // RDNA3/4: above four columns, static 4 rows for all types bench faster than the default
    स्थिर bool is_rdna3_or_4 = device->vendor_id == VK_VENDOR_ID_AMD && (device->architecture == AMD_RDNA3 || device->architecture == AMD_RDNA4);
    स्वतः स्थिर &rm_int_n = [&](uint32_t rows, uint32_t i) { वापस (is_rdna3_or_4 && i >= 4) ? 4u : rows; };
    // RDNA3/4 and NVIDIA except pre-Turing: use 4 rows for MUL_MAT_ID MMVQ.
    स्वतः स्थिर &rm_id = [&](uint32_t rows) {
        अगर (device->vendor_id == VK_VENDOR_ID_NVIDIA &&
            device->architecture != vk_device_architecture::NVIDIA_PRE_TURING) {
            वापस 4u;
        }
        वापस is_rdna3_or_4 ? 4u : rows;
    };
    uint32_t rm_iq = 2 * rm_kq;

    स्थिर bool use_subgroups = device->subgroup_arithmetic;
    // The Imagination proprietary compiler rejects the subgroup-only dequant mul_mat_vec
    // shaders that require a subgroup size >= 16; fall back to shared-memory reduction.
    स्थिर bool is_imagination_proprietary =
        device->driver_id == vk::DriverId::eImaginationProprietary;
    // Ensure a subgroup size >= 16 is available
    स्थिर bool use_subgroups16 = use_subgroups && subgroup_min_आकार_16 && !is_imagination_proprietary;

    स्थिर uint32_t subgroup_आकार = (device->vendor_id == VK_VENDOR_ID_INTEL && device->subgroup_आकार_control && device->subgroup_min_आकार <= 16 && device->subgroup_max_आकार >= 16) ? 16 : device->subgroup_आकार;
    स्थिर uint32_t subgroup_size16 = std::max(subgroup_आकार, 16u);

    स्थिर uint32_t force_subgroup_आकार = use_subgroups ? subgroup_आकार : 0;
    स्थिर uint32_t force_subgroup_size16 = use_subgroups16 ? subgroup_size16 : 0;
    अटल constexpr uint32_t mul_mat_vec_अंक_bindings = 5;
    अटल constexpr uint32_t mul_mat_vec_id_अंक_bindings = 6;

#अगर defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
#घोषणा OCP_DMMV_LEN(NAME, REDUC)  (device->ocp_fp4 ? NAME ## _ocp_len[REDUC]  : NAME ## _len[REDUC])
#घोषणा OCP_DMMV_तथ्य(NAME, REDUC) (device->ocp_fp4 ? NAME ## ocp_तथ्य[REDUC] : NAME ## तथ्य_2[REDUC])
#अन्यथा
#घोषणा OCP_DMMV_LEN(NAME, REDUC)  NAME ## _len[REDUC]
#घोषणा OCP_DMMV_तथ्य(NAME, REDUC) NAME ## तथ्य_2[REDUC]
#पूर्ण_अगर

    क्रम (uint32_t w = 0; w < DMMV_WG_आकार_गिनती; ++w) {
        स्थिर uint32_t wg_आकार_subgroup   = (w == DMMV_WG_आकार_SUBGROUP) ? subgroup_आकार : (subgroup_आकार * 4);
        स्थिर uint32_t wg_आकार_subgroup16 = (w == DMMV_WG_आकार_SUBGROUP) ? subgroup_size16 : (subgroup_size16 * 4);

        स्थिर shader_reduction_mode reduc = (use_subgroups && w == DMMV_WG_आकार_SUBGROUP) ? SHADER_REDUCTION_MODE_SUBGROUP :
                                            (use_subgroups && w == DMMV_WG_आकार_LARGE) ? SHADER_REDUCTION_MODE_HYBRID :
                                            SHADER_REDUCTION_MODE_SHMEM;

        स्थिर shader_reduction_mode reduc16 = (use_subgroups16 && w == DMMV_WG_आकार_SUBGROUP) ? SHADER_REDUCTION_MODE_SUBGROUP :
                                              (use_subgroups16 && w == DMMV_WG_आकार_LARGE) ? SHADER_REDUCTION_MODE_HYBRID :
                                              SHADER_REDUCTION_MODE_SHMEM;

        क्रम (uint32_t i = 0; i < mul_mat_vec_max_cols; ++i) {
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_F32 ][i], "mul_mat_vec_f32_f32_f32",  arr_dmmv_f32_f32_f32_len[reduc],  arr_dmmv_f32_f32_f32_तथ्य[reduc],  "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {1, 1, 1}, {wg_आकार_subgroup, 1, i+1}, 1, false, use_subgroups, force_subgroup_आकार);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_F16 ][i], "mul_mat_vec_f16_f32_f32",  arr_dmmv_f16_f32_f32_len[reduc],  arr_dmmv_f16_f32_f32_तथ्य[reduc],  "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {2, 1, 1}, {wg_आकार_subgroup, 2, i+1}, 1, false, use_subgroups, force_subgroup_आकार);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_BF16][i], "mul_mat_vec_bf16_f32_f32", arr_dmmv_bf16_f32_f32_len[reduc], arr_dmmv_bf16_f32_f32_तथ्य[reduc], "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {2, 1, 1}, {wg_आकार_subgroup, 2, i+1}, 1, false, use_subgroups, force_subgroup_आकार);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q1_0][i], "mul_mat_vec_q1_0_f32_f32", arr_dmmv_q1_0_f32_f32_len[reduc], arr_dmmv_q1_0_f32_f32_तथ्य[reduc], "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {2*rm_stdq, 1, 1}, {wg_आकार_subgroup, 2*rm_stdq, i+1}, 1, true, use_subgroups, force_subgroup_आकार);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q2_0][i], "mul_mat_vec_q2_0_f32_f32", arr_dmmv_q2_0_f32_f32_len[reduc], arr_dmmv_q2_0_f32_f32_तथ्य[reduc], "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {2*rm_stdq, 1, 1}, {wg_आकार_subgroup, 2*rm_stdq, i+1}, 1, true, use_subgroups, force_subgroup_आकार);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q4_0][i], "mul_mat_vec_q4_0_f32_f32", arr_dmmv_q4_0_f32_f32_len[reduc], arr_dmmv_q4_0_f32_f32_तथ्य[reduc], "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {2*rm_stdq, 1, 1}, {wg_आकार_subgroup, 2*rm_stdq, i+1}, 1, true, use_subgroups, force_subgroup_आकार);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q4_1][i], "mul_mat_vec_q4_1_f32_f32", arr_dmmv_q4_1_f32_f32_len[reduc], arr_dmmv_q4_1_f32_f32_तथ्य[reduc], "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {2*rm_stdq, 1, 1}, {wg_आकार_subgroup, 2*rm_stdq, i+1}, 1, true, use_subgroups, force_subgroup_आकार);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q5_0][i], "mul_mat_vec_q5_0_f32_f32", arr_dmmv_q5_0_f32_f32_len[reduc], arr_dmmv_q5_0_f32_f32_तथ्य[reduc], "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {2*rm_stdq, 1, 1}, {wg_आकार_subgroup, 2*rm_stdq, i+1}, 1, true, use_subgroups, force_subgroup_आकार);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q5_1][i], "mul_mat_vec_q5_1_f32_f32", arr_dmmv_q5_1_f32_f32_len[reduc], arr_dmmv_q5_1_f32_f32_तथ्य[reduc], "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {2*rm_stdq, 1, 1}, {wg_आकार_subgroup, 2*rm_stdq, i+1}, 1, true, use_subgroups, force_subgroup_आकार);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q8_0][i], "mul_mat_vec_q8_0_f32_f32", arr_dmmv_q8_0_f32_f32_len[reduc], arr_dmmv_q8_0_f32_f32_तथ्य[reduc], "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {1*rm_stdq, 1, 1}, {wg_आकार_subgroup, 1*rm_stdq, i+1}, 1, true, use_subgroups, force_subgroup_आकार);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q2_K][i], "mul_mat_vec_q2_k_f32_f32", arr_dmmv_q2_k_f32_f32_len[reduc16], arr_dmmv_q2_k_f32_f32_तथ्य[reduc16], "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {rm_kq, 1, 1}, {wg_आकार_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q3_K][i], "mul_mat_vec_q3_k_f32_f32", arr_dmmv_q3_k_f32_f32_len[reduc16], arr_dmmv_q3_k_f32_f32_तथ्य[reduc16], "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {rm_kq, 1, 1}, {wg_आकार_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q4_K][i], "mul_mat_vec_q4_k_f32_f32", arr_dmmv_q4_k_f32_f32_len[reduc16], arr_dmmv_q4_k_f32_f32_तथ्य[reduc16], "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {rm_kq, 1, 1}, {wg_आकार_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q5_K][i], "mul_mat_vec_q5_k_f32_f32", arr_dmmv_q5_k_f32_f32_len[reduc16], arr_dmmv_q5_k_f32_f32_तथ्य[reduc16], "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {rm_kq, 1, 1}, {wg_आकार_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q6_K][i], "mul_mat_vec_q6_k_f32_f32", arr_dmmv_q6_k_f32_f32_len[reduc16], arr_dmmv_q6_k_f32_f32_तथ्य[reduc16], "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {rm_kq, 1, 1}, {wg_आकार_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_TQ1_0][i], "mul_mat_vec_tq1_0_f32_f32", arr_dmmv_tq1_0_f32_f32_len[reduc16], arr_dmmv_tq1_0_f32_f32_तथ्य[reduc16], "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {rm_kq, 1, 1}, {wg_आकार_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_TQ2_0][i], "mul_mat_vec_tq2_0_f32_f32", arr_dmmv_tq2_0_f32_f32_len[reduc16], arr_dmmv_tq2_0_f32_f32_तथ्य[reduc16], "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {rm_kq, 1, 1}, {wg_आकार_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ1_S][i],   "mul_mat_vec_iq1_s_f32_f32",   arr_dmmv_iq1_s_f32_f32_len[reduc16],   arr_dmmv_iq1_s_f32_f32_तथ्य[reduc16],   "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {rm_iq, 1, 1}, {wg_आकार_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ1_M][i],   "mul_mat_vec_iq1_m_f32_f32",   arr_dmmv_iq1_m_f32_f32_len[reduc16],   arr_dmmv_iq1_m_f32_f32_तथ्य[reduc16],   "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {rm_iq, 1, 1}, {wg_आकार_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ2_XXS][i], "mul_mat_vec_iq2_xxs_f32_f32", arr_dmmv_iq2_xxs_f32_f32_len[reduc16], arr_dmmv_iq2_xxs_f32_f32_तथ्य[reduc16], "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {rm_iq, 1, 1}, {wg_आकार_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ2_XS][i],  "mul_mat_vec_iq2_xs_f32_f32",  arr_dmmv_iq2_xs_f32_f32_len[reduc16],  arr_dmmv_iq2_xs_f32_f32_तथ्य[reduc16],  "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {rm_iq, 1, 1}, {wg_आकार_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ2_S][i],   "mul_mat_vec_iq2_s_f32_f32",   arr_dmmv_iq2_s_f32_f32_len[reduc16],   arr_dmmv_iq2_s_f32_f32_तथ्य[reduc16],   "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {rm_iq, 1, 1}, {wg_आकार_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ3_XXS][i], "mul_mat_vec_iq3_xxs_f32_f32", arr_dmmv_iq3_xxs_f32_f32_len[reduc16], arr_dmmv_iq3_xxs_f32_f32_तथ्य[reduc16], "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {rm_iq, 1, 1}, {wg_आकार_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ3_S][i],   "mul_mat_vec_iq3_s_f32_f32",   arr_dmmv_iq3_s_f32_f32_len[reduc16],   arr_dmmv_iq3_s_f32_f32_तथ्य[reduc16],   "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {rm_iq, 1, 1}, {wg_आकार_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ4_XS][i],  "mul_mat_vec_iq4_xs_f32_f32",  arr_dmmv_iq4_xs_f32_f32_len[reduc16],  arr_dmmv_iq4_xs_f32_f32_तथ्य[reduc16],  "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {rm_iq, 1, 1}, {wg_आकार_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ4_NL][i],  "mul_mat_vec_iq4_nl_f32_f32",  arr_dmmv_iq4_nl_f32_f32_len[reduc16],  arr_dmmv_iq4_nl_f32_f32_तथ्य[reduc16],  "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {rm_iq, 1, 1}, {wg_आकार_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_MXFP4][i],   "mul_mat_vec_mxfp4_f32_f32",   OCP_DMMV_LEN(arr_dmmv_mxfp4_f32_f32, reduc16), OCP_DMMV_तथ्य(arr_dmmv_mxfp4_f32_f32, reduc16), "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {rm_iq, 1, 1}, {wg_आकार_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_NVFP4][i],   "mul_mat_vec_nvfp4_f32_f32",   OCP_DMMV_LEN(arr_dmmv_nvfp4_f32_f32, reduc16), OCP_DMMV_तथ्य(arr_dmmv_nvfp4_f32_f32, reduc16), "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {rm_iq, 1, 1}, {wg_आकार_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);

            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_F32 ][i], "mul_mat_vec_f32_f16_f32",  arr_dmmv_f32_f16_f32_len[reduc],  arr_dmmv_f32_f16_f32_तथ्य[reduc],  "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {1, 1, 1}, {wg_आकार_subgroup, 1, i+1}, 1, false, use_subgroups, force_subgroup_आकार);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_F16 ][i], "mul_mat_vec_f16_f16_f32",  arr_dmmv_f16_f16_f32_len[reduc],  arr_dmmv_f16_f16_f32_तथ्य[reduc],  "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {2, 1, 1}, {wg_आकार_subgroup, 2, i+1}, 1, false, use_subgroups, force_subgroup_आकार);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_BF16][i], "mul_mat_vec_bf16_f16_f32", arr_dmmv_bf16_f16_f32_len[reduc], arr_dmmv_bf16_f16_f32_तथ्य[reduc], "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {2, 1, 1}, {wg_आकार_subgroup, 2, i+1}, 1, false, use_subgroups, force_subgroup_आकार);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q1_0][i], "mul_mat_vec_q1_0_f16_f32", arr_dmmv_q1_0_f16_f32_len[reduc], arr_dmmv_q1_0_f16_f32_तथ्य[reduc], "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {2*rm_stdq, 1, 1}, {wg_आकार_subgroup, 2*rm_stdq, i+1}, 1, true, use_subgroups, force_subgroup_आकार);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q2_0][i], "mul_mat_vec_q2_0_f16_f32", arr_dmmv_q2_0_f16_f32_len[reduc], arr_dmmv_q2_0_f16_f32_तथ्य[reduc], "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {2*rm_stdq, 1, 1}, {wg_आकार_subgroup, 2*rm_stdq, i+1}, 1, true, use_subgroups, force_subgroup_आकार);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q4_0][i], "mul_mat_vec_q4_0_f16_f32", arr_dmmv_q4_0_f16_f32_len[reduc], arr_dmmv_q4_0_f16_f32_तथ्य[reduc], "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {2*rm_stdq, 1, 1}, {wg_आकार_subgroup, 2*rm_stdq, i+1}, 1, true, use_subgroups, force_subgroup_आकार);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q4_1][i], "mul_mat_vec_q4_1_f16_f32", arr_dmmv_q4_1_f16_f32_len[reduc], arr_dmmv_q4_1_f16_f32_तथ्य[reduc], "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {2*rm_stdq, 1, 1}, {wg_आकार_subgroup, 2*rm_stdq, i+1}, 1, true, use_subgroups, force_subgroup_आकार);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q5_0][i], "mul_mat_vec_q5_0_f16_f32", arr_dmmv_q5_0_f16_f32_len[reduc], arr_dmmv_q5_0_f16_f32_तथ्य[reduc], "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {2*rm_stdq, 1, 1}, {wg_आकार_subgroup, 2*rm_stdq, i+1}, 1, true, use_subgroups, force_subgroup_आकार);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q5_1][i], "mul_mat_vec_q5_1_f16_f32", arr_dmmv_q5_1_f16_f32_len[reduc], arr_dmmv_q5_1_f16_f32_तथ्य[reduc], "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {2*rm_stdq, 1, 1}, {wg_आकार_subgroup, 2*rm_stdq, i+1}, 1, true, use_subgroups, force_subgroup_आकार);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q8_0][i], "mul_mat_vec_q8_0_f16_f32", arr_dmmv_q8_0_f16_f32_len[reduc], arr_dmmv_q8_0_f16_f32_तथ्य[reduc], "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {1*rm_stdq, 1, 1}, {wg_आकार_subgroup, 1*rm_stdq, i+1}, 1, true, use_subgroups, force_subgroup_आकार);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q2_K][i], "mul_mat_vec_q2_k_f16_f32", arr_dmmv_q2_k_f16_f32_len[reduc16], arr_dmmv_q2_k_f16_f32_तथ्य[reduc16], "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {rm_kq, 1, 1}, {wg_आकार_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q3_K][i], "mul_mat_vec_q3_k_f16_f32", arr_dmmv_q3_k_f16_f32_len[reduc16], arr_dmmv_q3_k_f16_f32_तथ्य[reduc16], "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {rm_kq, 1, 1}, {wg_आकार_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q4_K][i], "mul_mat_vec_q4_k_f16_f32", arr_dmmv_q4_k_f16_f32_len[reduc16], arr_dmmv_q4_k_f16_f32_तथ्य[reduc16], "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {rm_kq, 1, 1}, {wg_आकार_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q5_K][i], "mul_mat_vec_q5_k_f16_f32", arr_dmmv_q5_k_f16_f32_len[reduc16], arr_dmmv_q5_k_f16_f32_तथ्य[reduc16], "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {rm_kq, 1, 1}, {wg_आकार_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q6_K][i], "mul_mat_vec_q6_k_f16_f32", arr_dmmv_q6_k_f16_f32_len[reduc16], arr_dmmv_q6_k_f16_f32_तथ्य[reduc16], "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {rm_kq, 1, 1}, {wg_आकार_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_TQ1_0][i], "mul_mat_vec_tq1_0_f16_f32", arr_dmmv_tq1_0_f16_f32_len[reduc16], arr_dmmv_tq1_0_f16_f32_तथ्य[reduc16], "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {rm_kq, 1, 1}, {wg_आकार_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_TQ2_0][i], "mul_mat_vec_tq2_0_f16_f32", arr_dmmv_tq2_0_f16_f32_len[reduc16], arr_dmmv_tq2_0_f16_f32_तथ्य[reduc16], "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {rm_kq, 1, 1}, {wg_आकार_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ1_S][i],   "mul_mat_vec_iq1_s_f16_f32",   arr_dmmv_iq1_s_f16_f32_len[reduc16],   arr_dmmv_iq1_s_f16_f32_तथ्य[reduc16],   "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {rm_iq, 1, 1}, {wg_आकार_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ1_M][i],   "mul_mat_vec_iq1_m_f16_f32",   arr_dmmv_iq1_m_f16_f32_len[reduc16],   arr_dmmv_iq1_m_f16_f32_तथ्य[reduc16],   "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {rm_iq, 1, 1}, {wg_आकार_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ2_XXS][i], "mul_mat_vec_iq2_xxs_f16_f32", arr_dmmv_iq2_xxs_f16_f32_len[reduc16], arr_dmmv_iq2_xxs_f16_f32_तथ्य[reduc16], "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {rm_iq, 1, 1}, {wg_आकार_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ2_XS][i],  "mul_mat_vec_iq2_xs_f16_f32",  arr_dmmv_iq2_xs_f16_f32_len[reduc16],  arr_dmmv_iq2_xs_f16_f32_तथ्य[reduc16],  "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {rm_iq, 1, 1}, {wg_आकार_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ2_S][i],   "mul_mat_vec_iq2_s_f16_f32",   arr_dmmv_iq2_s_f16_f32_len[reduc16],   arr_dmmv_iq2_s_f16_f32_तथ्य[reduc16],   "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {rm_iq, 1, 1}, {wg_आकार_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ3_XXS][i], "mul_mat_vec_iq3_xxs_f16_f32", arr_dmmv_iq3_xxs_f16_f32_len[reduc16], arr_dmmv_iq3_xxs_f16_f32_तथ्य[reduc16], "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {rm_iq, 1, 1}, {wg_आकार_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ3_S][i],   "mul_mat_vec_iq3_s_f16_f32",   arr_dmmv_iq3_s_f16_f32_len[reduc16],   arr_dmmv_iq3_s_f16_f32_तथ्य[reduc16],   "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {rm_iq, 1, 1}, {wg_आकार_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ4_XS][i],  "mul_mat_vec_iq4_xs_f16_f32",  arr_dmmv_iq4_xs_f16_f32_len[reduc16],  arr_dmmv_iq4_xs_f16_f32_तथ्य[reduc16],  "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {rm_iq, 1, 1}, {wg_आकार_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ4_NL][i],  "mul_mat_vec_iq4_nl_f16_f32",  arr_dmmv_iq4_nl_f16_f32_len[reduc16],  arr_dmmv_iq4_nl_f16_f32_तथ्य[reduc16],  "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {rm_iq, 1, 1}, {wg_आकार_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_MXFP4][i],   "mul_mat_vec_mxfp4_f16_f32",   OCP_DMMV_LEN(arr_dmmv_mxfp4_f16_f32, reduc16), OCP_DMMV_तथ्य(arr_dmmv_mxfp4_f16_f32, reduc16), "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {rm_iq, 1, 1}, {wg_आकार_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_NVFP4][i],   "mul_mat_vec_nvfp4_f16_f32",   OCP_DMMV_LEN(arr_dmmv_nvfp4_f16_f32, reduc16), OCP_DMMV_तथ्य(arr_dmmv_nvfp4_f16_f32, reduc16), "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {rm_iq, 1, 1}, {wg_आकार_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);

#अगर defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
            अगर (device->integer_dot_product) {
                स्थिर uint32_t subgroup_आकार_int = (device->vendor_id == VK_VENDOR_ID_INTEL && device->subgroup_आकार_control) ? device->subgroup_min_आकार : device->subgroup_आकार;
                स्थिर uint32_t wg_आकार_subgroup_int = (w == DMMV_WG_आकार_SUBGROUP) ? subgroup_आकार_int : (subgroup_आकार_int * 4);

                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q2_0][i], "mul_mat_vec_q2_0_q8_1_f32", arr_dmmv_q2_0_q8_1_f32_len[reduc], arr_dmmv_q2_0_q8_1_f32_तथ्य[reduc], "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {rm_int_n(2*rm_kq_int, i), 1, 1}, {wg_आकार_subgroup_int, rm_int_n(2*rm_kq_int, i), i+1}, 1, true, use_subgroups, subgroup_आकार_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q4_0][i], "mul_mat_vec_q4_0_q8_1_f32", arr_dmmv_q4_0_q8_1_f32_len[reduc], arr_dmmv_q4_0_q8_1_f32_तथ्य[reduc], "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {rm_int_n(1*rm_stdq_int, i), 1, 1}, {wg_आकार_subgroup_int, rm_int_n(1*rm_stdq_int, i), i+1}, 1, true, use_subgroups, subgroup_आकार_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q4_1][i], "mul_mat_vec_q4_1_q8_1_f32", arr_dmmv_q4_1_q8_1_f32_len[reduc], arr_dmmv_q4_1_q8_1_f32_तथ्य[reduc], "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {rm_int_n(1*rm_stdq_int, i), 1, 1}, {wg_आकार_subgroup_int, rm_int_n(1*rm_stdq_int, i), i+1}, 1, true, use_subgroups, subgroup_आकार_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q5_0][i], "mul_mat_vec_q5_0_q8_1_f32", arr_dmmv_q5_0_q8_1_f32_len[reduc], arr_dmmv_q5_0_q8_1_f32_तथ्य[reduc], "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {rm_int_n(1*rm_stdq_int, i), 1, 1}, {wg_आकार_subgroup_int, rm_int_n(1*rm_stdq_int, i), i+1}, 1, true, use_subgroups, subgroup_आकार_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q5_1][i], "mul_mat_vec_q5_1_q8_1_f32", arr_dmmv_q5_1_q8_1_f32_len[reduc], arr_dmmv_q5_1_q8_1_f32_तथ्य[reduc], "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {rm_int_n(1*rm_stdq_int, i), 1, 1}, {wg_आकार_subgroup_int, rm_int_n(1*rm_stdq_int, i), i+1}, 1, true, use_subgroups, subgroup_आकार_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q8_0][i], "mul_mat_vec_q8_0_q8_1_f32", arr_dmmv_q8_0_q8_1_f32_len[reduc], arr_dmmv_q8_0_q8_1_f32_तथ्य[reduc], "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {rm_int_n(1*rm_stdq_int, i), 1, 1}, {wg_आकार_subgroup_int, rm_int_n(1*rm_stdq_int, i), i+1}, 1, true, use_subgroups, subgroup_आकार_int);

                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_MXFP4][i], "mul_mat_vec_mxfp4_q8_1_f32", arr_dmmv_mxfp4_q8_1_f32_len[reduc], arr_dmmv_mxfp4_q8_1_f32_तथ्य[reduc], "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {rm_int_n(2*rm_stdq_int, i), 1, 1}, {wg_आकार_subgroup_int, rm_int_n(2*rm_stdq_int, i), i+1}, 1, true, use_subgroups, subgroup_आकार_int);

                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q2_K][i], "mul_mat_vec_q2_k_q8_1_f32", arr_dmmv_q2_k_q8_1_f32_len[reduc], arr_dmmv_q2_k_q8_1_f32_तथ्य[reduc], "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {rm_int_n(2*rm_kq_int, i), 1, 1}, {wg_आकार_subgroup_int, rm_int_n(2*rm_kq_int, i), i+1}, 1, true, use_subgroups, subgroup_आकार_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q3_K][i], "mul_mat_vec_q3_k_q8_1_f32", arr_dmmv_q3_k_q8_1_f32_len[reduc], arr_dmmv_q3_k_q8_1_f32_तथ्य[reduc], "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {rm_int_n(1*rm_kq_int, i), 1, 1}, {wg_आकार_subgroup_int, rm_int_n(1*rm_kq_int, i), i+1}, 1, true, use_subgroups, subgroup_आकार_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q4_K][i], "mul_mat_vec_q4_k_q8_1_f32", arr_dmmv_q4_k_q8_1_f32_len[reduc], arr_dmmv_q4_k_q8_1_f32_तथ्य[reduc], "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {rm_int_n(1*rm_kq_int, i), 1, 1}, {wg_आकार_subgroup_int, rm_int_n(1*rm_kq_int, i), i+1}, 1, true, use_subgroups, subgroup_आकार_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q5_K][i], "mul_mat_vec_q5_k_q8_1_f32", arr_dmmv_q5_k_q8_1_f32_len[reduc], arr_dmmv_q5_k_q8_1_f32_तथ्य[reduc], "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {rm_int_n(1*rm_kq_int, i), 1, 1}, {wg_आकार_subgroup_int, rm_int_n(1*rm_kq_int, i), i+1}, 1, true, use_subgroups, subgroup_आकार_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q6_K][i], "mul_mat_vec_q6_k_q8_1_f32", arr_dmmv_q6_k_q8_1_f32_len[reduc], arr_dmmv_q6_k_q8_1_f32_तथ्य[reduc], "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {rm_int_n(1*rm_kq_int, i), 1, 1}, {wg_आकार_subgroup_int, rm_int_n(1*rm_kq_int, i), i+1}, 1, true, use_subgroups, subgroup_आकार_int);

                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_IQ1_S][i], "mul_mat_vec_iq1_s_q8_1_f32", arr_dmmv_iq1_s_q8_1_f32_len[reduc], arr_dmmv_iq1_s_q8_1_f32_तथ्य[reduc], "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {1*rm_iq_int(i), 1, 1}, {wg_आकार_subgroup_int, 1*rm_iq_int(i), i+1}, 1, true, use_subgroups, subgroup_आकार_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_IQ1_M][i], "mul_mat_vec_iq1_m_q8_1_f32", arr_dmmv_iq1_m_q8_1_f32_len[reduc], arr_dmmv_iq1_m_q8_1_f32_तथ्य[reduc], "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {1*rm_iq_int(i), 1, 1}, {wg_आकार_subgroup_int, 1*rm_iq_int(i), i+1}, 1, true, use_subgroups, subgroup_आकार_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_IQ4_XS][i], "mul_mat_vec_iq4_xs_q8_1_f32", arr_dmmv_iq4_xs_q8_1_f32_len[reduc], arr_dmmv_iq4_xs_q8_1_f32_तथ्य[reduc], "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_धकेलो_constants), {rm_int_n(1*rm_stdq_int, i), 1, 1}, {wg_आकार_subgroup_int, rm_int_n(1*rm_stdq_int, i), i+1}, 1, true, use_subgroups, subgroup_आकार_int);

            }
#पूर्ण_अगर // GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT
        }

        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_F32 ], "mul_mat_vec_id_f32_f32",        arr_dmmv_id_f32_f32_f32_len[reduc],     arr_dmmv_id_f32_f32_f32_तथ्य[reduc],     "main", mul_mat_vec_id_अंक_bindings, माप(vk_mat_vec_id_धकेलो_constants), {1, 1, 1}, {wg_आकार_subgroup, 1}, 1, false, use_subgroups, force_subgroup_आकार);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_F16 ], "mul_mat_vec_id_f16_f32",        arr_dmmv_id_f16_f32_f32_len[reduc],     arr_dmmv_id_f16_f32_f32_तथ्य[reduc],     "main", mul_mat_vec_id_अंक_bindings, माप(vk_mat_vec_id_धकेलो_constants), {2, 1, 1}, {wg_आकार_subgroup, 2}, 1, false, use_subgroups, force_subgroup_आकार);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_BF16], "mul_mat_vec_id_bf16_f32",       arr_dmmv_id_bf16_f32_f32_len[reduc],    arr_dmmv_id_bf16_f32_f32_तथ्य[reduc],    "main", mul_mat_vec_id_अंक_bindings, माप(vk_mat_vec_id_धकेलो_constants), {2, 1, 1}, {wg_आकार_subgroup, 2}, 1, false, use_subgroups, force_subgroup_आकार);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q1_0], "mul_mat_vec_id_q1_0_f32",       arr_dmmv_id_q1_0_f32_f32_len[reduc],    arr_dmmv_id_q1_0_f32_f32_तथ्य[reduc],    "main", mul_mat_vec_id_अंक_bindings, माप(vk_mat_vec_id_धकेलो_constants), {2*rm_stdq, 1, 1}, {wg_आकार_subgroup, 2*rm_stdq}, 1, true, use_subgroups, force_subgroup_आकार);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q2_0], "mul_mat_vec_id_q2_0_f32",       arr_dmmv_id_q2_0_f32_f32_len[reduc],    arr_dmmv_id_q2_0_f32_f32_तथ्य[reduc],    "main", mul_mat_vec_id_अंक_bindings, माप(vk_mat_vec_id_धकेलो_constants), {2*rm_stdq, 1, 1}, {wg_आकार_subgroup, 2*rm_stdq}, 1, true, use_subgroups, force_subgroup_आकार);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q4_0], "mul_mat_vec_id_q4_0_f32",       arr_dmmv_id_q4_0_f32_f32_len[reduc],    arr_dmmv_id_q4_0_f32_f32_तथ्य[reduc],    "main", mul_mat_vec_id_अंक_bindings, माप(vk_mat_vec_id_धकेलो_constants), {2*rm_stdq, 1, 1}, {wg_आकार_subgroup, 2*rm_stdq}, 1, true, use_subgroups, force_subgroup_आकार);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q4_1], "mul_mat_vec_id_q4_1_f32",       arr_dmmv_id_q4_1_f32_f32_len[reduc],    arr_dmmv_id_q4_1_f32_f32_तथ्य[reduc],    "main", mul_mat_vec_id_अंक_bindings, माप(vk_mat_vec_id_धकेलो_constants), {2*rm_stdq, 1, 1}, {wg_आकार_subgroup, 2*rm_stdq}, 1, true, use_subgroups, force_subgroup_आकार);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q5_0], "mul_mat_vec_id_q5_0_f32",       arr_dmmv_id_q5_0_f32_f32_len[reduc],    arr_dmmv_id_q5_0_f32_f32_तथ्य[reduc],    "main", mul_mat_vec_id_अंक_bindings, माप(vk_mat_vec_id_धकेलो_constants), {2*rm_stdq, 1, 1}, {wg_आकार_subgroup, 2*rm_stdq}, 1, true, use_subgroups, force_subgroup_आकार);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q5_1], "mul_mat_vec_id_q5_1_f32",       arr_dmmv_id_q5_1_f32_f32_len[reduc],    arr_dmmv_id_q5_1_f32_f32_तथ्य[reduc],    "main", mul_mat_vec_id_अंक_bindings, माप(vk_mat_vec_id_धकेलो_constants), {2*rm_stdq, 1, 1}, {wg_आकार_subgroup, 2*rm_stdq}, 1, true, use_subgroups, force_subgroup_आकार);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q8_0], "mul_mat_vec_id_q8_0_f32",       arr_dmmv_id_q8_0_f32_f32_len[reduc],    arr_dmmv_id_q8_0_f32_f32_तथ्य[reduc],    "main", mul_mat_vec_id_अंक_bindings, माप(vk_mat_vec_id_धकेलो_constants), {1*rm_stdq, 1, 1}, {wg_आकार_subgroup, 1*rm_stdq}, 1, true, use_subgroups, force_subgroup_आकार);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q2_K], "mul_mat_vec_id_q2_k_f32",       arr_dmmv_id_q2_k_f32_f32_len[reduc16],    arr_dmmv_id_q2_k_f32_f32_तथ्य[reduc16],    "main", mul_mat_vec_id_अंक_bindings, माप(vk_mat_vec_id_धकेलो_constants), {rm_kq, 1, 1}, {wg_आकार_subgroup16, rm_kq}, 1, true, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q3_K], "mul_mat_vec_id_q3_k_f32",       arr_dmmv_id_q3_k_f32_f32_len[reduc16],    arr_dmmv_id_q3_k_f32_f32_तथ्य[reduc16],    "main", mul_mat_vec_id_अंक_bindings, माप(vk_mat_vec_id_धकेलो_constants), {rm_kq, 1, 1}, {wg_आकार_subgroup16, rm_kq}, 1, true, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q4_K], "mul_mat_vec_id_q4_k_f32",       arr_dmmv_id_q4_k_f32_f32_len[reduc16],    arr_dmmv_id_q4_k_f32_f32_तथ्य[reduc16],    "main", mul_mat_vec_id_अंक_bindings, माप(vk_mat_vec_id_धकेलो_constants), {rm_kq, 1, 1}, {wg_आकार_subgroup16, rm_kq}, 1, true, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q5_K], "mul_mat_vec_id_q5_k_f32",       arr_dmmv_id_q5_k_f32_f32_len[reduc16],    arr_dmmv_id_q5_k_f32_f32_तथ्य[reduc16],    "main", mul_mat_vec_id_अंक_bindings, माप(vk_mat_vec_id_धकेलो_constants), {rm_kq, 1, 1}, {wg_आकार_subgroup16, rm_kq}, 1, true, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q6_K], "mul_mat_vec_id_q6_k_f32",       arr_dmmv_id_q6_k_f32_f32_len[reduc16],    arr_dmmv_id_q6_k_f32_f32_तथ्य[reduc16],    "main", mul_mat_vec_id_अंक_bindings, माप(vk_mat_vec_id_धकेलो_constants), {rm_kq, 1, 1}, {wg_आकार_subgroup16, rm_kq}, 1, true, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_TQ1_0], "mul_mat_vec_id_tq1_0_f32",     arr_dmmv_id_tq1_0_f32_f32_len[reduc16],   arr_dmmv_id_tq1_0_f32_f32_तथ्य[reduc16],   "main", mul_mat_vec_id_अंक_bindings, माप(vk_mat_vec_id_धकेलो_constants), {rm_kq, 1, 1}, {wg_आकार_subgroup16, rm_kq}, 1, true, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_TQ2_0], "mul_mat_vec_id_tq2_0_f32",     arr_dmmv_id_tq2_0_f32_f32_len[reduc16],   arr_dmmv_id_tq2_0_f32_f32_तथ्य[reduc16],   "main", mul_mat_vec_id_अंक_bindings, माप(vk_mat_vec_id_धकेलो_constants), {rm_kq, 1, 1}, {wg_आकार_subgroup16, rm_kq}, 1, true, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ1_S],   "mul_mat_vec_id_iq1_s_f32",   arr_dmmv_id_iq1_s_f32_f32_len[reduc16],   arr_dmmv_id_iq1_s_f32_f32_तथ्य[reduc16],   "main", mul_mat_vec_id_अंक_bindings, माप(vk_mat_vec_id_धकेलो_constants), {rm_iq, 1, 1}, {wg_आकार_subgroup16, rm_iq}, 1, true, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ1_M],   "mul_mat_vec_id_iq1_m_f32",   arr_dmmv_id_iq1_m_f32_f32_len[reduc16],   arr_dmmv_id_iq1_m_f32_f32_तथ्य[reduc16],   "main", mul_mat_vec_id_अंक_bindings, माप(vk_mat_vec_id_धकेलो_constants), {rm_iq, 1, 1}, {wg_आकार_subgroup16, rm_iq}, 1, true, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ2_XXS], "mul_mat_vec_id_iq2_xxs_f32", arr_dmmv_id_iq2_xxs_f32_f32_len[reduc16], arr_dmmv_id_iq2_xxs_f32_f32_तथ्य[reduc16], "main", mul_mat_vec_id_अंक_bindings, माप(vk_mat_vec_id_धकेलो_constants), {rm_iq, 1, 1}, {wg_आकार_subgroup16, rm_iq}, 1, true, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ2_XS],  "mul_mat_vec_id_iq2_xs_f32",  arr_dmmv_id_iq2_xs_f32_f32_len[reduc16],  arr_dmmv_id_iq2_xs_f32_f32_तथ्य[reduc16],  "main", mul_mat_vec_id_अंक_bindings, माप(vk_mat_vec_id_धकेलो_constants), {rm_iq, 1, 1}, {wg_आकार_subgroup16, rm_iq}, 1, true, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ2_S],   "mul_mat_vec_id_iq2_s_f32",   arr_dmmv_id_iq2_s_f32_f32_len[reduc16],   arr_dmmv_id_iq2_s_f32_f32_तथ्य[reduc16],   "main", mul_mat_vec_id_अंक_bindings, माप(vk_mat_vec_id_धकेलो_constants), {rm_iq, 1, 1}, {wg_आकार_subgroup16, rm_iq}, 1, true, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ3_XXS], "mul_mat_vec_id_iq3_xxs_f32", arr_dmmv_id_iq3_xxs_f32_f32_len[reduc16], arr_dmmv_id_iq3_xxs_f32_f32_तथ्य[reduc16], "main", mul_mat_vec_id_अंक_bindings, माप(vk_mat_vec_id_धकेलो_constants), {rm_iq, 1, 1}, {wg_आकार_subgroup16, rm_iq}, 1, true, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ3_S],   "mul_mat_vec_id_iq3_s_f32",   arr_dmmv_id_iq3_s_f32_f32_len[reduc16],   arr_dmmv_id_iq3_s_f32_f32_तथ्य[reduc16],   "main", mul_mat_vec_id_अंक_bindings, माप(vk_mat_vec_id_धकेलो_constants), {rm_iq, 1, 1}, {wg_आकार_subgroup16, rm_iq}, 1, true, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ4_XS],  "mul_mat_vec_id_iq4_xs_f32",  arr_dmmv_id_iq4_xs_f32_f32_len[reduc16],  arr_dmmv_id_iq4_xs_f32_f32_तथ्य[reduc16],  "main", mul_mat_vec_id_अंक_bindings, माप(vk_mat_vec_id_धकेलो_constants), {rm_iq, 1, 1}, {wg_आकार_subgroup16, rm_iq}, 1, true, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ4_NL],  "mul_mat_vec_id_iq4_nl_f32",  arr_dmmv_id_iq4_nl_f32_f32_len[reduc16],  arr_dmmv_id_iq4_nl_f32_f32_तथ्य[reduc16],  "main", mul_mat_vec_id_अंक_bindings, माप(vk_mat_vec_id_धकेलो_constants), {rm_iq, 1, 1}, {wg_आकार_subgroup16, rm_iq}, 1, true, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_MXFP4],   "mul_mat_vec_id_mxfp4_f32",   OCP_DMMV_LEN(arr_dmmv_id_mxfp4_f32_f32, reduc16), OCP_DMMV_तथ्य(arr_dmmv_id_mxfp4_f32_f32, reduc16), "main", mul_mat_vec_id_अंक_bindings, माप(vk_mat_vec_id_धकेलो_constants), {rm_iq, 1, 1}, {wg_आकार_subgroup16, rm_iq}, 1, true, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_NVFP4],   "mul_mat_vec_id_nvfp4_f32",   OCP_DMMV_LEN(arr_dmmv_id_nvfp4_f32_f32, reduc16), OCP_DMMV_तथ्य(arr_dmmv_id_nvfp4_f32_f32, reduc16), "main", mul_mat_vec_id_अंक_bindings, माप(vk_mat_vec_id_धकेलो_constants), {rm_iq, 1, 1}, {wg_आकार_subgroup16, rm_iq}, 1, true, use_subgroups16, force_subgroup_size16);

#अगर defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
        अगर (device->integer_dot_product) {
            स्थिर uint32_t subgroup_आकार_int = (device->vendor_id == VK_VENDOR_ID_INTEL && device->subgroup_आकार_control) ? device->subgroup_min_आकार : device->subgroup_आकार;
            स्थिर uint32_t wg_आकार_subgroup_int = (w == DMMV_WG_आकार_SUBGROUP) ? subgroup_आकार_int : (subgroup_आकार_int * 4);

            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q2_0], "mul_mat_vec_id_q2_0_q8_1_f32", arr_dmmv_id_q2_0_q8_1_f32_len[reduc], arr_dmmv_id_q2_0_q8_1_f32_तथ्य[reduc], "main", mul_mat_vec_id_अंक_bindings, माप(vk_mat_vec_id_धकेलो_constants), {rm_id(2*rm_kq_int), 1, 1}, {wg_आकार_subgroup_int, rm_id(2*rm_kq_int)}, 1, true, use_subgroups, subgroup_आकार_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q4_0], "mul_mat_vec_id_q4_0_q8_1_f32", arr_dmmv_id_q4_0_q8_1_f32_len[reduc], arr_dmmv_id_q4_0_q8_1_f32_तथ्य[reduc], "main", mul_mat_vec_id_अंक_bindings, माप(vk_mat_vec_id_धकेलो_constants), {rm_id(1*rm_stdq_int), 1, 1}, {wg_आकार_subgroup_int, rm_id(1*rm_stdq_int)}, 1, true, use_subgroups, subgroup_आकार_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q4_1], "mul_mat_vec_id_q4_1_q8_1_f32", arr_dmmv_id_q4_1_q8_1_f32_len[reduc], arr_dmmv_id_q4_1_q8_1_f32_तथ्य[reduc], "main", mul_mat_vec_id_अंक_bindings, माप(vk_mat_vec_id_धकेलो_constants), {rm_id(1*rm_stdq_int), 1, 1}, {wg_आकार_subgroup_int, rm_id(1*rm_stdq_int)}, 1, true, use_subgroups, subgroup_आकार_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q5_0], "mul_mat_vec_id_q5_0_q8_1_f32", arr_dmmv_id_q5_0_q8_1_f32_len[reduc], arr_dmmv_id_q5_0_q8_1_f32_तथ्य[reduc], "main", mul_mat_vec_id_अंक_bindings, माप(vk_mat_vec_id_धकेलो_constants), {rm_id(1*rm_stdq_int), 1, 1}, {wg_आकार_subgroup_int, rm_id(1*rm_stdq_int)}, 1, true, use_subgroups, subgroup_आकार_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q5_1], "mul_mat_vec_id_q5_1_q8_1_f32", arr_dmmv_id_q5_1_q8_1_f32_len[reduc], arr_dmmv_id_q5_1_q8_1_f32_तथ्य[reduc], "main", mul_mat_vec_id_अंक_bindings, माप(vk_mat_vec_id_धकेलो_constants), {rm_id(1*rm_stdq_int), 1, 1}, {wg_आकार_subgroup_int, rm_id(1*rm_stdq_int)}, 1, true, use_subgroups, subgroup_आकार_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q8_0], "mul_mat_vec_id_q8_0_q8_1_f32", arr_dmmv_id_q8_0_q8_1_f32_len[reduc], arr_dmmv_id_q8_0_q8_1_f32_तथ्य[reduc], "main", mul_mat_vec_id_अंक_bindings, माप(vk_mat_vec_id_धकेलो_constants), {rm_id(1*rm_stdq_int), 1, 1}, {wg_आकार_subgroup_int, rm_id(1*rm_stdq_int)}, 1, true, use_subgroups, subgroup_आकार_int);

            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_MXFP4], "mul_mat_vec_id_mxfp4_q8_1_f32", arr_dmmv_id_mxfp4_q8_1_f32_len[reduc], arr_dmmv_id_mxfp4_q8_1_f32_तथ्य[reduc], "main", mul_mat_vec_id_अंक_bindings, माप(vk_mat_vec_id_धकेलो_constants), {rm_id(2*rm_stdq_int), 1, 1}, {wg_आकार_subgroup_int, rm_id(2*rm_stdq_int)}, 1, true, use_subgroups, subgroup_आकार_int);

            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q2_K], "mul_mat_vec_id_q2_k_q8_1_f32", arr_dmmv_id_q2_k_q8_1_f32_len[reduc], arr_dmmv_id_q2_k_q8_1_f32_तथ्य[reduc], "main", mul_mat_vec_id_अंक_bindings, माप(vk_mat_vec_id_धकेलो_constants), {rm_id(2*rm_kq_int), 1, 1}, {wg_आकार_subgroup_int, rm_id(2*rm_kq_int)}, 1, true, use_subgroups, subgroup_आकार_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q3_K], "mul_mat_vec_id_q3_k_q8_1_f32", arr_dmmv_id_q3_k_q8_1_f32_len[reduc], arr_dmmv_id_q3_k_q8_1_f32_तथ्य[reduc], "main", mul_mat_vec_id_अंक_bindings, माप(vk_mat_vec_id_धकेलो_constants), {rm_id(1*rm_kq_int), 1, 1}, {wg_आकार_subgroup_int, rm_id(1*rm_kq_int)}, 1, true, use_subgroups, subgroup_आकार_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q4_K], "mul_mat_vec_id_q4_k_q8_1_f32", arr_dmmv_id_q4_k_q8_1_f32_len[reduc], arr_dmmv_id_q4_k_q8_1_f32_तथ्य[reduc], "main", mul_mat_vec_id_अंक_bindings, माप(vk_mat_vec_id_धकेलो_constants), {rm_id(1*rm_kq_int), 1, 1}, {wg_आकार_subgroup_int, rm_id(1*rm_kq_int)}, 1, true, use_subgroups, subgroup_आकार_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q5_K], "mul_mat_vec_id_q5_k_q8_1_f32", arr_dmmv_id_q5_k_q8_1_f32_len[reduc], arr_dmmv_id_q5_k_q8_1_f32_तथ्य[reduc], "main", mul_mat_vec_id_अंक_bindings, माप(vk_mat_vec_id_धकेलो_constants), {rm_id(1*rm_kq_int), 1, 1}, {wg_आकार_subgroup_int, rm_id(1*rm_kq_int)}, 1, true, use_subgroups, subgroup_आकार_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q6_K], "mul_mat_vec_id_q6_k_q8_1_f32", arr_dmmv_id_q6_k_q8_1_f32_len[reduc], arr_dmmv_id_q6_k_q8_1_f32_तथ्य[reduc], "main", mul_mat_vec_id_अंक_bindings, माप(vk_mat_vec_id_धकेलो_constants), {rm_id(1*rm_kq_int), 1, 1}, {wg_आकार_subgroup_int, rm_id(1*rm_kq_int)}, 1, true, use_subgroups, subgroup_आकार_int);

            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_IQ1_S], "mul_mat_vec_id_iq1_s_q8_1_f32", arr_dmmv_id_iq1_s_q8_1_f32_len[reduc], arr_dmmv_id_iq1_s_q8_1_f32_तथ्य[reduc], "main", mul_mat_vec_id_अंक_bindings, माप(vk_mat_vec_id_धकेलो_constants), {1*rm_iq_int(0), 1, 1}, {wg_आकार_subgroup_int, 1*rm_iq_int(0)}, 1, true, use_subgroups, subgroup_आकार_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_IQ1_M], "mul_mat_vec_id_iq1_m_q8_1_f32", arr_dmmv_id_iq1_m_q8_1_f32_len[reduc], arr_dmmv_id_iq1_m_q8_1_f32_तथ्य[reduc], "main", mul_mat_vec_id_अंक_bindings, माप(vk_mat_vec_id_धकेलो_constants), {1*rm_iq_int(0), 1, 1}, {wg_आकार_subgroup_int, 1*rm_iq_int(0)}, 1, true, use_subgroups, subgroup_आकार_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_IQ4_XS], "mul_mat_vec_id_iq4_xs_q8_1_f32", arr_dmmv_id_iq4_xs_q8_1_f32_len[reduc], arr_dmmv_id_iq4_xs_q8_1_f32_तथ्य[reduc], "main", mul_mat_vec_id_अंक_bindings, माप(vk_mat_vec_id_धकेलो_constants), {rm_id(1*rm_stdq_int), 1, 1}, {wg_आकार_subgroup_int, rm_id(1*rm_stdq_int)}, 1, true, use_subgroups, subgroup_आकार_int);
        }
#पूर्ण_अगर // GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT
    }

#अघोषित OCP_DMMV_तथ्य
#अघोषित OCP_DMMV_LEN

#अगर !defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
    GGML_UNUSED(rm_stdq_int);
    GGML_UNUSED(rm_kq_int);
    GGML_UNUSED(is_rdna3_or_4);
    GGML_UNUSED(rm_int_n);
    GGML_UNUSED(rm_id);
    GGML_UNUSED(rm_iq_int);
#पूर्ण_अगर

    // dequant shaders
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_F32 ], "f32_to_f16",   dequant_f32_len,  dequant_f32_तथ्य,  "main", 2, 5 * माप(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q1_0], "dequant_q1_0", dequant_q1_0_len, dequant_q1_0_तथ्य, "main", 2, 5 * माप(uint32_t), {256 * 8, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q2_0], "dequant_q2_0", dequant_q2_0_len, dequant_q2_0_तथ्य, "main", 2, 5 * माप(uint32_t), {256 * 4, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q4_0], "dequant_q4_0", dequant_q4_0_len, dequant_q4_0_तथ्य, "main", 2, 5 * माप(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q4_1], "dequant_q4_1", dequant_q4_1_len, dequant_q4_1_तथ्य, "main", 2, 5 * माप(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q5_0], "dequant_q5_0", dequant_q5_0_len, dequant_q5_0_तथ्य, "main", 2, 5 * माप(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q5_1], "dequant_q5_1", dequant_q5_1_len, dequant_q5_1_तथ्य, "main", 2, 5 * माप(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q8_0], "dequant_q8_0", dequant_q8_0_len, dequant_q8_0_तथ्य, "main", 2, 5 * माप(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant_transpose[GGML_TYPE_Q8_0], "dequant_q8_0_transpose", dequant_q8_0_transpose_len, dequant_q8_0_transpose_तथ्य, "main", 2, 5 * माप(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q2_K], "dequant_q2_k", dequant_q2_k_len, dequant_q2_k_तथ्य, "main", 2, 5 * माप(uint32_t), {256 * 64, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q3_K], "dequant_q3_k", dequant_q3_k_len, dequant_q3_k_तथ्य, "main", 2, 5 * माप(uint32_t), {256 * 64, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q4_K], "dequant_q4_k", dequant_q4_k_len, dequant_q4_k_तथ्य, "main", 2, 5 * माप(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q5_K], "dequant_q5_k", dequant_q5_k_len, dequant_q5_k_तथ्य, "main", 2, 5 * माप(uint32_t), {256 * 64, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q6_K], "dequant_q6_k", dequant_q6_k_len, dequant_q6_k_तथ्य, "main", 2, 5 * माप(uint32_t), {256 * 64, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_TQ1_0], "dequant_tq1_0", dequant_tq1_0_len, dequant_tq1_0_तथ्य, "main", 2, 5 * माप(uint32_t), {256 * 4, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_TQ2_0], "dequant_tq2_0", dequant_tq2_0_len, dequant_tq2_0_तथ्य, "main", 2, 5 * माप(uint32_t), {256 * 64, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ1_S],   "dequant_iq1_s",   dequant_iq1_s_len,   dequant_iq1_s_तथ्य,   "main", 2, 5 * माप(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ1_M],   "dequant_iq1_m",   dequant_iq1_m_len,   dequant_iq1_m_तथ्य,   "main", 2, 5 * माप(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ2_XXS], "dequant_iq2_xxs", dequant_iq2_xxs_len, dequant_iq2_xxs_तथ्य, "main", 2, 5 * माप(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ2_XS],  "dequant_iq2_xs",  dequant_iq2_xs_len,  dequant_iq2_xs_तथ्य,  "main", 2, 5 * माप(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ2_S],   "dequant_iq2_s",   dequant_iq2_s_len,   dequant_iq2_s_तथ्य,   "main", 2, 5 * माप(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ3_XXS], "dequant_iq3_xxs", dequant_iq3_xxs_len, dequant_iq3_xxs_तथ्य, "main", 2, 5 * माप(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ3_S],   "dequant_iq3_s",   dequant_iq3_s_len,   dequant_iq3_s_तथ्य,   "main", 2, 5 * माप(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ4_XS],  "dequant_iq4_xs",  dequant_iq4_xs_len,  dequant_iq4_xs_तथ्य,  "main", 2, 5 * माप(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ4_NL],  "dequant_iq4_nl",  dequant_iq4_nl_len,  dequant_iq4_nl_तथ्य,  "main", 2, 5 * माप(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_MXFP4],   "dequant_mxfp4",   dequant_mxfp4_len,   dequant_mxfp4_तथ्य,   "main", 2, 5 * माप(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_NVFP4],   "dequant_nvfp4",   dequant_nvfp4_len,   dequant_nvfp4_तथ्य,   "main", 2, 5 * माप(uint32_t), {256 * 16, 1, 1}, {}, 1);

    // get_rows
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_F32 ], "get_rows_f32",  get_rows_f32_len,  get_rows_f32_तथ्य,  "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_F16 ], "get_rows_f16",  get_rows_f16_len,  get_rows_f16_तथ्य,  "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_BF16], "get_rows_bf16", get_rows_bf16_len, get_rows_bf16_तथ्य, "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q1_0], "get_rows_q1_0", get_rows_q1_0_len, get_rows_q1_0_तथ्य, "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q2_0], "get_rows_q2_0", get_rows_q2_0_len, get_rows_q2_0_तथ्य, "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q4_0], "get_rows_q4_0", get_rows_q4_0_len, get_rows_q4_0_तथ्य, "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q4_1], "get_rows_q4_1", get_rows_q4_1_len, get_rows_q4_1_तथ्य, "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q5_0], "get_rows_q5_0", get_rows_q5_0_len, get_rows_q5_0_तथ्य, "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q5_1], "get_rows_q5_1", get_rows_q5_1_len, get_rows_q5_1_तथ्य, "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q8_0], "get_rows_q8_0", get_rows_q8_0_len, get_rows_q8_0_तथ्य, "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q2_K], "get_rows_q2_k", get_rows_q2_k_len, get_rows_q2_k_तथ्य, "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q3_K], "get_rows_q3_k", get_rows_q3_k_len, get_rows_q3_k_तथ्य, "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q4_K], "get_rows_q4_k", get_rows_q4_k_len, get_rows_q4_k_तथ्य, "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q5_K], "get_rows_q5_k", get_rows_q5_k_len, get_rows_q5_k_तथ्य, "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q6_K], "get_rows_q6_k", get_rows_q6_k_len, get_rows_q6_k_तथ्य, "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_TQ1_0], "get_rows_tq1_0", get_rows_tq1_0_len, get_rows_tq1_0_तथ्य, "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_TQ2_0], "get_rows_tq2_0", get_rows_tq2_0_len, get_rows_tq2_0_तथ्य, "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_IQ1_S],   "get_rows_iq1_s",   get_rows_iq1_s_len,   get_rows_iq1_s_तथ्य,   "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_IQ1_M],   "get_rows_iq1_m",   get_rows_iq1_m_len,   get_rows_iq1_m_तथ्य,   "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_IQ2_XXS], "get_rows_iq2_xxs", get_rows_iq2_xxs_len, get_rows_iq2_xxs_तथ्य, "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_IQ2_XS],  "get_rows_iq2_xs",  get_rows_iq2_xs_len,  get_rows_iq2_xs_तथ्य,  "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_IQ2_S],   "get_rows_iq2_s",   get_rows_iq2_s_len,   get_rows_iq2_s_तथ्य,   "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_IQ3_XXS], "get_rows_iq3_xxs", get_rows_iq3_xxs_len, get_rows_iq3_xxs_तथ्य, "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_IQ3_S],   "get_rows_iq3_s",   get_rows_iq3_s_len,   get_rows_iq3_s_तथ्य,   "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_IQ4_XS],  "get_rows_iq4_xs",  get_rows_iq4_xs_len,  get_rows_iq4_xs_तथ्य,  "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_IQ4_NL],  "get_rows_iq4_nl",  get_rows_iq4_nl_len,  get_rows_iq4_nl_तथ्य,  "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_MXFP4],   "get_rows_mxfp4",   get_rows_mxfp4_len,   get_rows_mxfp4_तथ्य,   "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_NVFP4],   "get_rows_nvfp4",   get_rows_nvfp4_len,   get_rows_nvfp4_तथ्य,   "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_I32],     "get_rows_i32",     get_rows_i32_len,     get_rows_i32_तथ्य,     "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {1024, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_F32 ], "get_rows_f32_f32",  get_rows_f32_f32_len,  get_rows_f32_f32_तथ्य,  "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_F16 ], "get_rows_f16_f32",  get_rows_f16_f32_len,  get_rows_f16_f32_तथ्य,  "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_BF16], "get_rows_bf16_f32", get_rows_bf16_f32_len, get_rows_bf16_f32_तथ्य, "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q1_0], "get_rows_q1_0_f32", get_rows_q1_0_f32_len, get_rows_q1_0_f32_तथ्य, "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q2_0], "get_rows_q2_0_f32", get_rows_q2_0_f32_len, get_rows_q2_0_f32_तथ्य, "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q4_0], "get_rows_q4_0_f32", get_rows_q4_0_f32_len, get_rows_q4_0_f32_तथ्य, "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q4_1], "get_rows_q4_1_f32", get_rows_q4_1_f32_len, get_rows_q4_1_f32_तथ्य, "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q5_0], "get_rows_q5_0_f32", get_rows_q5_0_f32_len, get_rows_q5_0_f32_तथ्य, "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q5_1], "get_rows_q5_1_f32", get_rows_q5_1_f32_len, get_rows_q5_1_f32_तथ्य, "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q8_0], "get_rows_q8_0_f32", get_rows_q8_0_f32_len, get_rows_q8_0_f32_तथ्य, "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q2_K], "get_rows_q2_k_f32", get_rows_q2_k_f32_len, get_rows_q2_k_f32_तथ्य, "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q3_K], "get_rows_q3_k_f32", get_rows_q3_k_f32_len, get_rows_q3_k_f32_तथ्य, "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q4_K], "get_rows_q4_k_f32", get_rows_q4_k_f32_len, get_rows_q4_k_f32_तथ्य, "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q5_K], "get_rows_q5_k_f32", get_rows_q5_k_f32_len, get_rows_q5_k_f32_तथ्य, "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q6_K], "get_rows_q6_k_f32", get_rows_q6_k_f32_len, get_rows_q6_k_f32_तथ्य, "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_TQ1_0], "get_rows_tq1_0_f32", get_rows_tq1_0_f32_len, get_rows_tq1_0_f32_तथ्य, "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_TQ2_0], "get_rows_tq2_0_f32", get_rows_tq2_0_f32_len, get_rows_tq2_0_f32_तथ्य, "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_IQ1_S],   "get_rows_iq1_s_f32",   get_rows_iq1_s_f32_len,   get_rows_iq1_s_f32_तथ्य,   "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_IQ1_M],   "get_rows_iq1_m_f32",   get_rows_iq1_m_f32_len,   get_rows_iq1_m_f32_तथ्य,   "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_IQ2_XXS], "get_rows_iq2_xxs_f32", get_rows_iq2_xxs_f32_len, get_rows_iq2_xxs_f32_तथ्य, "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_IQ2_XS],  "get_rows_iq2_xs_f32",  get_rows_iq2_xs_f32_len,  get_rows_iq2_xs_f32_तथ्य,  "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_IQ2_S],   "get_rows_iq2_s_f32",   get_rows_iq2_s_f32_len,   get_rows_iq2_s_f32_तथ्य,   "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_IQ3_XXS], "get_rows_iq3_xxs_f32", get_rows_iq3_xxs_f32_len, get_rows_iq3_xxs_f32_तथ्य, "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_IQ3_S],   "get_rows_iq3_s_f32",   get_rows_iq3_s_f32_len,   get_rows_iq3_s_f32_तथ्य,   "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_IQ4_XS],  "get_rows_iq4_xs_f32",  get_rows_iq4_xs_f32_len,  get_rows_iq4_xs_f32_तथ्य,  "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_IQ4_NL],  "get_rows_iq4_nl_f32",  get_rows_iq4_nl_f32_len,  get_rows_iq4_nl_f32_तथ्य,  "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_MXFP4],   "get_rows_mxfp4_f32",   get_rows_mxfp4_f32_len,   get_rows_mxfp4_f32_तथ्य,   "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_NVFP4],   "get_rows_nvfp4_f32",   get_rows_nvfp4_f32_len,   get_rows_nvfp4_f32_तथ्य,   "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_back_f32, "get_rows_back_f32", get_rows_back_f32_len, get_rows_back_f32_तथ्य, "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {256, 1, 1}, {}, 1, true);

    ggml_vk_create_pipeline(device, device->pipeline_matmul_split_k_reduce, "split_k_reduce", split_k_reduce_len, split_k_reduce_तथ्य, "main", 2, 2 * माप(uint32_t), {256 * 4, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_flash_attn_split_k_reduce, "fa_split_k_reduce", fa_split_k_reduce_len, fa_split_k_reduce_तथ्य, "main", 3, माप(vk_op_flash_attn_split_k_reduce_धकेलो_constants), {1, device->subgroup_आकार, 1}, {device->subgroup_आकार}, 1, true);

#अगर defined(VK_KHR_cooperative_matrix) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    अगर (device->vendor_id == VK_VENDOR_ID_INTEL && (device->architecture == INTEL_XE2 || (device->architecture == INTEL_XE1 && device->coopmat_support && device->uma))) {
        स्वतः upper_power_of_2 = [&](uint32_t in) {
            GGML_ASSERT(in != 0);
            अगर (in <= 1) वापस 1u;
            uint32_t ret = in - 1;
            ret |= ret >> 1;
            ret |= ret >> 2;
            ret |= ret >> 4;
            ret |= ret >> 8;
            ret |= ret >> 16;
            वापस ret + 1;
            };

        uint32_t xe_native_sub_group_आकार = 16;
        अगर (device->architecture == INTEL_XE1) {
            xe_native_sub_group_आकार = 8;
        }

        क्रम (स्वतः& it : device->pipeline_xe_fa_decode_dual_phases) {
            स्थिर uint32_t split_p_chunk = 32;
            स्वतः HdQk = it.first;
            स्वतः& pipelines = it.second;
            uint32_t शीर्ष_dim_qk = std::लो<0>(HdQk);
            uint32_t शीर्ष_dim_pv = std::लो<1>(HdQk);
            uint32_t gqa_ratio = std::लो<2>(HdQk);
            uint32_t q_len = std::लो<3>(HdQk);
            स्थिर uint32_t out_dim_per_wg = gqa_ratio > 16 ? 8 : 16;
            uint32_t aligned_q_len = upper_power_of_2(q_len);
            uint32_t group_sz_ph1 = std::min(std::max(aligned_q_len * xe_native_sub_group_आकार, 64u), 256u);
            uint32_t out_per_wg_ph1 = std::min(q_len, 256u / xe_native_sub_group_आकार);
            uint32_t aligned_gqa_ratio = upper_power_of_2(gqa_ratio);
            uint32_t split_p_per_iter_ph2 = 256;
            uint32_t split_p_per_warp = 16;
            uint32_t group_sz_ph2 = (split_p_per_iter_ph2 / split_p_per_warp) * xe_native_sub_group_आकार;
            uint32_t out_per_wg_ph2 = std::min(std::max(16u / aligned_gqa_ratio, 1u), q_len);
            ggml_vk_create_pipeline(device, pipelines.first, "xe_fa_decode_ph1", fa_decode_ph1_cm1_len, fa_decode_ph1_cm1_तथ्य, "main", 5, माप(vk_fa_xe_opt_धकेलो_constants), { 1, 32, 1 }, { group_sz_ph1, gqa_ratio, शीर्ष_dim_qk, xe_native_sub_group_आकार, split_p_chunk, out_per_wg_ph1 }, 1, false, true, xe_native_sub_group_आकार);
            ggml_vk_create_pipeline(device, pipelines.second, "xe_fa_decode_ph2", fa_decode_ph2_cm1_len, fa_decode_ph2_cm1_तथ्य, "main", 5, माप(vk_fa_xe_opt_धकेलो_constants), { 1, 1, 1 }, { group_sz_ph2, gqa_ratio, शीर्ष_dim_pv, out_per_wg_ph2, xe_native_sub_group_आकार, split_p_per_iter_ph2, split_p_chunk, out_dim_per_wg }, 1, false, true, xe_native_sub_group_आकार);
        }
    }
#पूर्ण_अगर

    क्रम (स्वतः &it : device->pipeline_fa_mask_opt) {
        स्वतः BrBc = it.first;
        ggml_vk_create_pipeline(device, it.second, "fa_mask_opt", fa_mask_opt_len, fa_mask_opt_तथ्य, "main", 2, माप(vk_op_flash_attn_mask_opt_धकेलो_constants), {1, 1, 1}, {128, 128 / device->subgroup_आकार, BrBc.first, BrBc.second}, 1, true, true, device->subgroup_आकार);
    }

    {
        // Large workgroup so the per-row KV scan parallelizes; capped to device limits.
        स्थिर uint32_t compact_max = std::min({1024u, device->properties.limits.maxComputeWorkGroupInvocations, device->properties.limits.max_Compute_Work_Group_आकार[0]});

        // Fast ballot prefix-sum path when the device supports full subgroups; otherwise
        // a shared-memory prefix-sum fallback. Both emit a deterministic ascending list.
        device->fa_sparse_compact_use_subgroups = device->subgroup_ballot && device->subgroup_require_full_support;
        अगर (device->fa_sparse_compact_use_subgroups) {
            स्थिर uint32_t compact_wg = std::max(device->subgroup_आकार, (compact_max / device->subgroup_आकार) * device->subgroup_आकार);
            स्थिर uint32_t compact_अंक_sg = compact_wg / device->subgroup_आकार;
            ggml_vk_create_pipeline(device, device->pipeline_fa_sparse_compact_subgroup, "fa_sparse_compact_subgroup", fa_sparse_compact_subgroup_len, fa_sparse_compact_subgroup_तथ्य, "main", 2, माप(vk_op_flash_attn_sparse_compact_धकेलो_constants), {1, 1, 1}, {compact_wg, compact_अंक_sg}, 1, true, true, device->subgroup_आकार);
        } अन्यथा {
            ggml_vk_create_pipeline(device, device->pipeline_fa_sparse_compact, "fa_sparse_compact", fa_sparse_compact_len, fa_sparse_compact_तथ्य, "main", 2, माप(vk_op_flash_attn_sparse_compact_धकेलो_constants), {1, 1, 1}, {compact_max}, 1, true);
        }
    }

    अगर (device->subgroup_clustered && device->subgroup_require_full_support) {
        ggml_vk_create_pipeline(device, device->pipeline_quantize_q8_1_x4, "quantize_q8_1_x4", quantize_q8_1_x4_subgroup_len, quantize_q8_1_x4_subgroup_तथ्य, "main", 2, माप(vk_quantize_q8_1_धकेलो_constants), {32 * device->subgroup_आकार / 8, 1, 1}, { device->subgroup_आकार }, 1, true, true);
    } अन्यथा {
        ggml_vk_create_pipeline(device, device->pipeline_quantize_q8_1_x4, "quantize_q8_1_x4", quantize_q8_1_x4_len, quantize_q8_1_x4_तथ्य, "main", 2, माप(vk_quantize_q8_1_धकेलो_constants), {32 * device->subgroup_आकार / 8, 1, 1}, { device->subgroup_आकार }, 1);
    }

    क्रम (uint32_t i = 0; i < p021_max_gqa_ratio; ++i) {
        अगर (device->subgroup_arithmetic && device->subgroup_require_full_support) {
            ggml_vk_create_pipeline2(device, device->pipeline_mul_mat_vec_p021_f16_f32[i], "mul_mat_vec_p021_f16_f32"+std::to_string(i+1), mul_mat_vec_p021_f16_f32_subgroup_add_len, mul_mat_vec_p021_f16_f32_subgroup_add_तथ्य, "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_p021_धकेलो_constants), {1, 1, 1}, {device->subgroup_आकार, i + 1}, 1, true, true);
        } अन्यथा {
            ggml_vk_create_pipeline2(device, device->pipeline_mul_mat_vec_p021_f16_f32[i], "mul_mat_vec_p021_f16_f32"+std::to_string(i+1), mul_mat_vec_p021_f16_f32_len,              mul_mat_vec_p021_f16_f32_तथ्य,              "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_p021_धकेलो_constants), {1, 1, 1}, {device->subgroup_आकार, i + 1}, 1, true);
        }
    }
    ggml_vk_create_pipeline(device, device->pipeline_mul_mat_vec_nc_f16_f32, "mul_mat_vec_nc_f16_f32", mul_mat_vec_nc_f16_f32_len, mul_mat_vec_nc_f16_f32_तथ्य, "main", mul_mat_vec_अंक_bindings, माप(vk_mat_vec_nc_धकेलो_constants), {1, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_norm_f32, "norm_f32", norm_f32_len, norm_f32_तथ्य, "main", 2, माप(vk_op_unary_धकेलो_constants), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_group_norm_f32, "group_norm_f32", group_norm_f32_len, group_norm_f32_तथ्य, "main", 2, माप(vk_op_धकेलो_constants), {1, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_f32, "rms_norm_f32", rms_norm_f32_len, rms_norm_f32_तथ्य, "main", 4, माप(vk_op_द्विआधारी_धकेलो_constants), {1, 1, 1}, {0, 0}, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_f32, "rms_norm_mul_f32", rms_norm_f32_len, rms_norm_f32_तथ्य, "main", 4, माप(vk_op_द्विआधारी_धकेलो_constants), {1, 1, 1}, {0, 1}, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_add_f32, "rms_norm_mul_add_f32", rms_norm_mul_add_f32_len, rms_norm_mul_add_f32_तथ्य, "main", 5, माप(vk_op_द्विआधारी_धकेलो_constants), {1, 1, 1}, {0, 1, 0}, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_add_mul_f32, "rms_norm_mul_add_mul_f32", rms_norm_mul_add_f32_len, rms_norm_mul_add_f32_तथ्य, "main", 5, माप(vk_op_द्विआधारी_धकेलो_constants), {1, 1, 1}, {0, 1, 1}, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_add_partials_f32, "rms_norm_mul_add_partials_f32", rms_norm_mul_add_partials_f32_len, rms_norm_mul_add_partials_f32_तथ्य, "main", 6, माप(vk_op_द्विआधारी_धकेलो_constants), {1, 1, 1}, {0, 1, 0}, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_add_mul_partials_f32, "rms_norm_mul_add_mul_partials_f32", rms_norm_mul_add_partials_f32_len, rms_norm_mul_add_partials_f32_तथ्य, "main", 6, माप(vk_op_द्विआधारी_धकेलो_constants), {1, 1, 1}, {0, 1, 1}, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_set_rows_f32_f32, "rms_norm_set_rows_f32_f32", rms_norm_set_rows_f32_f32_len, rms_norm_set_rows_f32_f32_तथ्य, "main", 4, माप(vk_op_द्विआधारी_धकेलो_constants), {1, 1, 1}, {0, 0}, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_set_rows_f32_f16, "rms_norm_set_rows_f32_f16", rms_norm_set_rows_f32_f16_len, rms_norm_set_rows_f32_f16_तथ्य, "main", 4, माप(vk_op_द्विआधारी_धकेलो_constants), {1, 1, 1}, {0, 0}, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_partials_f32, "rms_norm_partials_f32", rms_norm_partials_f32_len, rms_norm_partials_f32_तथ्य, "main", 4, माप(vk_op_द्विआधारी_धकेलो_constants), {1, 1, 1}, {0, 0}, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_partials_f32, "rms_norm_mul_partials_f32", rms_norm_partials_f32_len, rms_norm_partials_f32_तथ्य, "main", 4, माप(vk_op_द्विआधारी_धकेलो_constants), {1, 1, 1}, {0, 1}, 1, true);

    अगर (माप(vk_op_rms_norm_mul_rope_धकेलो_constants) <= device->properties.limits.max_धकेलो_Constants_आकार) {
        ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_rope_f32_f32, "rms_norm_mul_rope_f32_f32", rms_norm_mul_rope_f32_f32_len, rms_norm_mul_rope_f32_f32_तथ्य, "main", 7, माप(vk_op_rms_norm_mul_rope_धकेलो_constants), {1, 1, 1}, {0, 1}, 1, true);
        ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_rope_f32_f16, "rms_norm_mul_rope_f32_f16", rms_norm_mul_rope_f32_f16_len, rms_norm_mul_rope_f32_f16_तथ्य, "main", 7, माप(vk_op_rms_norm_mul_rope_धकेलो_constants), {1, 1, 1}, {0, 1}, 1, true);
    }

    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_back_f32, "rms_norm_back_f32", rms_norm_back_f32_len, rms_norm_back_f32_तथ्य, "main", 3, माप(vk_op_धकेलो_constants), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_l2_norm_f32, "l2_norm_f32", l2_norm_f32_len, l2_norm_f32_तथ्य, "main", 2, माप(vk_op_unary_धकेलो_constants), {1, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_cpy_32, "cpy_32", cpy_32_len, cpy_32_तथ्य, "main", 2, माप(vk_op_unary_धकेलो_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_f16, "cpy_f32_f16", cpy_f32_f16_len, cpy_f32_f16_तथ्य, "main", 2, माप(vk_op_unary_धकेलो_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_16, "cpy_16", cpy_16_len, cpy_16_तथ्य, "main", 2, माप(vk_op_unary_धकेलो_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f16_f32, "cpy_f16_f32", cpy_f16_f32_len, cpy_f16_f32_तथ्य, "main", 2, माप(vk_op_unary_धकेलो_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_bf16,"cpy_f32_bf16",cpy_f32_bf16_len,cpy_f32_bf16_तथ्य,"main", 2, माप(vk_op_unary_धकेलो_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_bf16_f32,"cpy_bf16_f32",cpy_bf16_f32_len,cpy_bf16_f32_तथ्य,"main", 2, माप(vk_op_unary_धकेलो_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_bf16_f16,"cpy_bf16_f16",cpy_bf16_f16_len,cpy_bf16_f16_तथ्य,"main", 2, माप(vk_op_unary_धकेलो_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_i32_f32, "cpy_i32_f32", cpy_i32_f32_len, cpy_i32_f32_तथ्य, "main", 2, माप(vk_op_unary_धकेलो_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_i32, "cpy_f32_i32", cpy_f32_i32_len, cpy_f32_i32_तथ्य, "main", 2, माप(vk_op_unary_धकेलो_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_32, "contig_cpy_32", contig_cpy_32_len, contig_cpy_32_तथ्य, "main", 2, माप(vk_op_unary_धकेलो_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_f32_f16, "contig_cpy_f32_f16", contig_cpy_f32_f16_len, contig_cpy_f32_f16_तथ्य, "main", 2, माप(vk_op_unary_धकेलो_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_16, "contig_cpy_16", contig_cpy_16_len, contig_cpy_16_तथ्य, "main", 2, माप(vk_op_unary_धकेलो_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_f16_f32, "contig_cpy_f16_f32", contig_cpy_f16_f32_len, contig_cpy_f16_f32_तथ्य, "main", 2, माप(vk_op_unary_धकेलो_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_f32_bf16,"contig_cpy_f32_bf16",contig_cpy_f32_bf16_len,contig_cpy_f32_bf16_तथ्य,"main", 2, माप(vk_op_unary_धकेलो_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_bf16_f32,"contig_cpy_bf16_f32",contig_cpy_bf16_f32_len,contig_cpy_bf16_f32_तथ्य,"main", 2, माप(vk_op_unary_धकेलो_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_bf16_f16,"contig_cpy_bf16_f16",contig_cpy_bf16_f16_len,contig_cpy_bf16_f16_तथ्य,"main", 2, माप(vk_op_unary_धकेलो_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_i32_f32, "contig_cpy_i32_f32", contig_cpy_i32_f32_len, contig_cpy_i32_f32_तथ्य, "main", 2, माप(vk_op_unary_धकेलो_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_f32_i32, "contig_cpy_f32_i32", contig_cpy_f32_i32_len, contig_cpy_f32_i32_तथ्य, "main", 2, माप(vk_op_unary_धकेलो_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_cpy_transpose_32, "cpy_transpose_32", cpy_transpose_32_len, cpy_transpose_32_तथ्य, "main", 2, माप(vk_op_unary_धकेलो_constants), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_transpose_16, "cpy_transpose_16", cpy_transpose_16_len, cpy_transpose_16_तथ्य, "main", 2, माप(vk_op_unary_धकेलो_constants), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_transpose_02_32, "cpy_transpose_02_32", cpy_transpose_02_32_len, cpy_transpose_02_32_तथ्य, "main", 2, माप(vk_op_unary_धकेलो_constants), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_transpose_02_16, "cpy_transpose_02_16", cpy_transpose_02_16_len, cpy_transpose_02_16_तथ्य, "main", 2, माप(vk_op_unary_धकेलो_constants), {1, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q1_0], "cpy_f32_q1_0", cpy_f32_q1_0_len, cpy_f32_q1_0_तथ्य, "main", 2, माप(vk_op_unary_धकेलो_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q2_0], "cpy_f32_q2_0", cpy_f32_q2_0_len, cpy_f32_q2_0_तथ्य, "main", 2, माप(vk_op_unary_धकेलो_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q4_0], "cpy_f32_q4_0", cpy_f32_q4_0_len, cpy_f32_q4_0_तथ्य, "main", 2, माप(vk_op_unary_धकेलो_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q4_1], "cpy_f32_q4_1", cpy_f32_q4_1_len, cpy_f32_q4_1_तथ्य, "main", 2, माप(vk_op_unary_धकेलो_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q5_0], "cpy_f32_q5_0", cpy_f32_q5_0_len, cpy_f32_q5_0_तथ्य, "main", 2, माप(vk_op_unary_धकेलो_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q5_1], "cpy_f32_q5_1", cpy_f32_q5_1_len, cpy_f32_q5_1_तथ्य, "main", 2, माप(vk_op_unary_धकेलो_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q8_0], "cpy_f32_q8_0", cpy_f32_q8_0_len, cpy_f32_q8_0_तथ्य, "main", 2, माप(vk_op_unary_धकेलो_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_IQ4_NL], "cpy_f32_iq4_nl", cpy_f32_iq4_nl_len, cpy_f32_iq4_nl_तथ्य, "main", 2, माप(vk_op_unary_धकेलो_constants), {32, 1, 1}, {}, 1);

#घोषणा SET_ROWS(src_idx, src, itype) \
        ggml_vk_create_pipeline(device, device->pipeline_set_rows ## itype [src_idx][GGML_TYPE_F32],  "set_rows_" #src "_f32" #itype,  set_rows_ ## src ## _f32 ## itype ## _len,  set_rows_ ## src ## _f32 ## itype ## तथ्य_2,  "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {1, 1, 1}, {1}, 1, true); \
        ggml_vk_create_pipeline(device, device->pipeline_set_rows ## itype [src_idx][GGML_TYPE_F16],  "set_rows_" #src "_f16" #itype,  set_rows_ ## src ## _f16 ## itype ## _len,  set_rows_ ## src ## _f16 ## itype ## तथ्य_2,  "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {1, 1, 1}, {1}, 1, true); \
        ggml_vk_create_pipeline(device, device->pipeline_set_rows ## itype [src_idx][GGML_TYPE_BF16], "set_rows_" #src "_bf16" #itype, set_rows_ ## src ## _bf16 ## itype ## _len, set_rows_ ## src ## _bf16 ## itype ## तथ्य_2, "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {1, 1, 1}, {1}, 1, true); \
        ggml_vk_create_pipeline(device, device->pipeline_set_rows ## itype [src_idx][GGML_TYPE_Q1_0], "set_rows_" #src "_q1_0" #itype, set_rows_ ## src ## _q1_0 ## itype ## _len, set_rows_ ## src ## _q1_0 ## itype ## तथ्य_2, "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {1, 1, 1}, {1}, 1, true); \
        ggml_vk_create_pipeline(device, device->pipeline_set_rows ## itype [src_idx][GGML_TYPE_Q2_0], "set_rows_" #src "_q2_0" #itype, set_rows_ ## src ## _q2_0 ## itype ## _len, set_rows_ ## src ## _q2_0 ## itype ## तथ्य_2, "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {1, 1, 1}, {1}, 1, true); \
        ggml_vk_create_pipeline(device, device->pipeline_set_rows ## itype [src_idx][GGML_TYPE_Q4_0], "set_rows_" #src "_q4_0" #itype, set_rows_ ## src ## _q4_0 ## itype ## _len, set_rows_ ## src ## _q4_0 ## itype ## तथ्य_2, "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {1, 1, 1}, {1}, 1, true); \
        ggml_vk_create_pipeline(device, device->pipeline_set_rows ## itype [src_idx][GGML_TYPE_Q4_1], "set_rows_" #src "_q4_1" #itype, set_rows_ ## src ## _q4_1 ## itype ## _len, set_rows_ ## src ## _q4_1 ## itype ## तथ्य_2, "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {1, 1, 1}, {1}, 1, true); \
        ggml_vk_create_pipeline(device, device->pipeline_set_rows ## itype [src_idx][GGML_TYPE_Q5_0], "set_rows_" #src "_q5_0" #itype, set_rows_ ## src ## _q5_0 ## itype ## _len, set_rows_ ## src ## _q5_0 ## itype ## तथ्य_2, "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {1, 1, 1}, {1}, 1, true); \
        ggml_vk_create_pipeline(device, device->pipeline_set_rows ## itype [src_idx][GGML_TYPE_Q5_1], "set_rows_" #src "_q5_1" #itype, set_rows_ ## src ## _q5_1 ## itype ## _len, set_rows_ ## src ## _q5_1 ## itype ## तथ्य_2, "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {1, 1, 1}, {1}, 1, true); \
        ggml_vk_create_pipeline(device, device->pipeline_set_rows ## itype [src_idx][GGML_TYPE_Q8_0], "set_rows_" #src "_q8_0" #itype, set_rows_ ## src ## _q8_0 ## itype ## _len, set_rows_ ## src ## _q8_0 ## itype ## तथ्य_2, "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {1, 1, 1}, {1}, 1, true); \
        ggml_vk_create_pipeline(device, device->pipeline_set_rows ## itype [src_idx][GGML_TYPE_IQ4_NL], "set_rows_" #src "_iq4_nl" #itype, set_rows_ ## src ## _iq4_nl ## itype ## _len, set_rows_ ## src ## _iq4_nl ## itype ## तथ्य_2, "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {1, 1, 1}, {1}, 1, true);

    SET_ROWS(0, f32, _i32)
    SET_ROWS(0, f32, _i64)
    SET_ROWS(1, f16, _i32)
    SET_ROWS(1, f16, _i64)
#अघोषित SET_ROWS


    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q1_0], "cpy_q1_0_f32", cpy_q1_0_f32_len, cpy_q1_0_f32_तथ्य, "main", 2, माप(vk_op_unary_धकेलो_constants), {(uint32_t)ggml_blck_आकार(GGML_TYPE_Q1_0), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q2_0], "cpy_q2_0_f32", cpy_q2_0_f32_len, cpy_q2_0_f32_तथ्य, "main", 2, माप(vk_op_unary_धकेलो_constants), {(uint32_t)ggml_blck_आकार(GGML_TYPE_Q2_0), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q4_0], "cpy_q4_0_f32", cpy_q4_0_f32_len, cpy_q4_0_f32_तथ्य, "main", 2, माप(vk_op_unary_धकेलो_constants), {(uint32_t)ggml_blck_आकार(GGML_TYPE_Q4_0), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q4_1], "cpy_q4_1_f32", cpy_q4_1_f32_len, cpy_q4_1_f32_तथ्य, "main", 2, माप(vk_op_unary_धकेलो_constants), {(uint32_t)ggml_blck_आकार(GGML_TYPE_Q4_1), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q5_0], "cpy_q5_0_f32", cpy_q5_0_f32_len, cpy_q5_0_f32_तथ्य, "main", 2, माप(vk_op_unary_धकेलो_constants), {(uint32_t)ggml_blck_आकार(GGML_TYPE_Q5_0), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q5_1], "cpy_q5_1_f32", cpy_q5_1_f32_len, cpy_q5_1_f32_तथ्य, "main", 2, माप(vk_op_unary_धकेलो_constants), {(uint32_t)ggml_blck_आकार(GGML_TYPE_Q5_1), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q8_0], "cpy_q8_0_f32", cpy_q8_0_f32_len, cpy_q8_0_f32_तथ्य, "main", 2, माप(vk_op_unary_धकेलो_constants), {(uint32_t)ggml_blck_आकार(GGML_TYPE_Q8_0), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_IQ4_NL], "cpy_iq4_nl_f32", cpy_iq4_nl_f32_len, cpy_iq4_nl_f32_तथ्य, "main", 2, माप(vk_op_unary_धकेलो_constants), {(uint32_t)ggml_blck_आकार(GGML_TYPE_IQ4_NL), 1, 1}, {}, 1);

    स्वतः get_suffix = [](bool src0_f16, bool src1_f16, bool dst_f16) {
        std::string s;
        s += std::string(src0_f16 ? "_f16" : "_f32");
        s += std::string(src1_f16 ? "_f16" : "_f32");
        s += std::string(dst_f16 ? "_f16" : "_f32");
        वापस s;
    };

#घोषणा CREATE_द्विआधारी(name, namemod, spec, bindings) \
    क्रम (पूर्णांक s0 : {0,1}) क्रम (पूर्णांक s1 : {0,1}) क्रम (पूर्णांक d : {0,1}) \
        ggml_vk_create_pipeline2(device, device->pipeline_ ## name ## namemod[s0][s1][d], \
                                #name + get_suffix(s0, s1, d) + #namemod, name ## _len[s0][s1][d], name ## तथ्य_2[s0][s1][d], \
                                "main", (bindings), माप(vk_op_द्विआधारी_धकेलो_constants), {512, 1, 1}, spec, 1);

    CREATE_द्विआधारी(add, , {0}, 4)
    CREATE_द्विआधारी(add, _norepeat, {1}, 4)
    CREATE_द्विआधारी(sub, , {0}, 3)
    CREATE_द्विआधारी(sub, _norepeat, {1}, 3)
    CREATE_द्विआधारी(mul, , {0}, 3)
    CREATE_द्विआधारी(mul, _norepeat, {1}, 3)
    CREATE_द्विआधारी(भाग, , {0}, 3)
    CREATE_द्विआधारी(भाग, _norepeat, {1}, 3)
    CREATE_द्विआधारी(add_rms, , {0}, 4)
    CREATE_द्विआधारी(add_rms, _norepeat, {1}, 4)
#अघोषित CREATE_द्विआधारी

    अगर (device->multi_add) {
        क्रम (uint32_t i = 0; i < MAX_FUSED_ADDS; ++i) {
            ggml_vk_create_pipeline2(device, device->pipeline_multi_add[i],     "multi_add_f32_"     + std::to_string(i+1), multi_add_f32_len,     multi_add_f32_तथ्य,     "main", MAX_PARAMETER_गिनती, माप(vk_op_multi_add_धकेलो_constants), {512, 1, 1}, {i+2}, 1);
            ggml_vk_create_pipeline2(device, device->pipeline_multi_add_rms[i], "multi_add_rms_f32_" + std::to_string(i+1), multi_add_rms_f32_len, multi_add_rms_f32_तथ्य, "main", MAX_PARAMETER_गिनती, माप(vk_op_multi_add_धकेलो_constants), {512, 1, 1}, {i+2}, 1);
        }
    }

    ggml_vk_create_pipeline(device, device->pipeline_add_id_f32, "add_id_f32", add_id_f32_len, add_id_f32_तथ्य, "main", 4, माप(vk_op_add_id_धकेलो_constants), {1, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_acc_f32, "acc_f32", acc_f32_len, acc_f32_तथ्य, "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {512, 1, 1}, {0, 1}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_set_f32, "set_f32", acc_f32_len, acc_f32_तथ्य, "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {512, 1, 1}, {0, 0}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_concat_i8, "concat_i8", concat_i8_len, concat_i8_तथ्य, "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_concat_i16, "concat_i16", concat_i16_len, concat_i16_तथ्य, "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_concat_i32, "concat_i32", concat_i32_len, concat_i32_तथ्य, "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_concat_i64, "concat_i64", concat_i64_len, concat_i64_तथ्य, "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_upscale_nearest_f32, "upscale_f32", upscale_f32_len, upscale_f32_तथ्य, "main", 2, माप(vk_op_upscale_धकेलो_constants), {512, 1, 1}, {GGML_SCALE_MODE_NEAREST}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_upscale_bilinear_f32, "upscale_f32", upscale_f32_len, upscale_f32_तथ्य, "main", 2, माप(vk_op_upscale_धकेलो_constants), {512, 1, 1}, {GGML_SCALE_MODE_BILINEAR}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_upscale_bicubic_f32, "upscale_f32", upscale_f32_len, upscale_f32_तथ्य, "main", 2, माप(vk_op_upscale_धकेलो_constants), {512, 1, 1}, {GGML_SCALE_MODE_BICUBIC}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_upscale_bilinear_antialias_f32, "upscale_f32", upscale_f32_len, upscale_f32_तथ्य, "main", 2, माप(vk_op_upscale_धकेलो_constants), {512, 1, 1}, {GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ANTIALIAS}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_scale_f32, "scale_f32", scale_f32_len, scale_f32_तथ्य, "main", 2, माप(vk_op_unary_धकेलो_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_log[0], "log_f32", log_f32_len, log_f32_तथ्य, "main", 2, माप(vk_op_unary_धकेलो_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_log[1], "log_f16", log_f16_len, log_f16_तथ्य, "main", 2, माप(vk_op_unary_धकेलो_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_tri[0], "tri_f32", tri_f32_len, tri_f32_तथ्य, "main", 2, माप(vk_op_unary_धकेलो_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_tri[1], "tri_f16", tri_f16_len, tri_f16_तथ्य, "main", 2, माप(vk_op_unary_धकेलो_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_diag[0], "diag_f32", diag_f32_len, diag_f32_तथ्य, "main", 2, माप(vk_op_unary_धकेलो_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_diag[1], "diag_f16", diag_f16_len, diag_f16_तथ्य, "main", 2, माप(vk_op_unary_धकेलो_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_pad_f32, "pad_f32", pad_f32_len, pad_f32_तथ्य, "main", 2, माप(vk_op_pad_धकेलो_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_pad_reflect_1d_f32, "pad_reflect_1d_f32", pad_reflect_1d_f32_len, pad_reflect_1d_f32_तथ्य, "main", 2, माप(vk_op_unary_धकेलो_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_roll_f32, "roll_f32", roll_f32_len, roll_f32_तथ्य, "main", 2, माप(vk_op_unary_धकेलो_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_repeat_i32, "repeat_i32", repeat_i32_len, repeat_i32_तथ्य, "main", 2, माप(vk_op_unary_धकेलो_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_repeat_back_f32, "repeat_back_f32", repeat_back_f32_len, repeat_back_f32_तथ्य, "main", 2, माप(vk_op_unary_धकेलो_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_repeat_i16, "repeat_i16", repeat_i16_len, repeat_i16_तथ्य, "main", 2, माप(vk_op_unary_धकेलो_constants), {512, 1, 1}, {}, 1);

#घोषणा CREATE_UNARY(name)  \
    ggml_vk_create_pipeline(device, device->pipeline_ ## name [0], #name "_f32", name ## _f32_len, name ## f32_तथ्य, "main", 2, माप(vk_op_unary_धकेलो_constants), {512, 1, 1}, {}, 1);  \
    ggml_vk_create_pipeline(device, device->pipeline_ ## name [1], #name "_f16", name ## _f16_len, name ## f16_तथ्य, "main", 2, माप(vk_op_unary_धकेलो_constants), {512, 1, 1}, {}, 1);

    CREATE_UNARY(elu)
    CREATE_UNARY(gelu)
    CREATE_UNARY(gelu_erf)
    CREATE_UNARY(gelu_quick)
    CREATE_UNARY(silu)
    CREATE_UNARY(relu)
    CREATE_UNARY(sqr)
    CREATE_UNARY(वर्ग_मूल)
    CREATE_UNARY(sin)
    CREATE_UNARY(cos)
    CREATE_UNARY(clamp)
    CREATE_UNARY(leaky_relu)
    CREATE_UNARY(xielu)
    CREATE_UNARY(neg)
    CREATE_UNARY(tanh)
    CREATE_UNARY(sigmoid)
    CREATE_UNARY(hardsigmoid)
    CREATE_UNARY(hardswish)
    CREATE_UNARY(असल)
    CREATE_UNARY(softplus)
    CREATE_UNARY(step)
    CREATE_UNARY(round)
    CREATE_UNARY(उच्चमान)
    CREATE_UNARY(न्यूनमान)
    CREATE_UNARY(छाँटो)
    CREATE_UNARY(sgn)
    CREATE_UNARY(exp)
    CREATE_UNARY(expm1)
#अघोषित CREATE_UNARY

// spec constants: {norepeat, op_on_b}
#घोषणा CREATE_UNARY_MUL(name, idx) \
    क्रम (पूर्णांक dt = 0; dt < 2; ++dt) { \
        स्थिर माप_प्रकार len_ = dt ? name ## _mul_f16_len : name ## _mul_f32_len; \
        स्थिर अचिन्हित अक्षर * तथ्य_3 = dt ? name ## mul_f16_तथ्य : name ## mul_f32_तथ्य; \
        स्थिर std::string dts_ = dt ? "f16" : "f32"; \
        क्रम (पूर्णांक ob = 0; ob < 2; ++ob) \
            क्रम (पूर्णांक nr = 0; nr < 2; ++nr) \
                ggml_vk_create_pipeline(device, device->pipeline_unary_mul[(idx)][dt][nr][ob], \
                    (#name "_mul" + std::string(ob ? "_b" : "") + "_" + dts_ + (nr ? "_norepeat" : "")).c_str(), \
                    len_, तथ्य_3, "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {512, 1, 1}, \
                    { (uint32_t) nr, (uint32_t) ob }, 1); \
    }

    CREATE_UNARY_MUL(gelu, 0)
    CREATE_UNARY_MUL(sigmoid, 1)
    CREATE_UNARY_MUL(silu, 2)
    CREATE_UNARY_MUL(softplus, 3)
#अघोषित CREATE_UNARY_MUL

    ggml_vk_create_pipeline(device, device->pipeline_add1_f16_f16, "add1_f16_f16", add1_f16_f16_len, add1_f16_f16_तथ्य, "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_add1_f16_f32, "add1_f16_f32", add1_f16_f32_len, add1_f16_f32_तथ्य, "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_add1_f32_f32, "add1_f32_f32", add1_f32_f32_len, add1_f32_f32_तथ्य, "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_arange_f32, "arange_f32", arange_f32_len, arange_f32_तथ्य, "main", 1, माप(vk_op_धकेलो_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_fill_f32, "fill_f32", fill_f32_len, fill_f32_तथ्य, "main", 1, माप(vk_op_धकेलो_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_fill_f16, "fill_f16", fill_f16_len, fill_f16_तथ्य, "main", 1, माप(vk_op_धकेलो_constants), {512, 1, 1}, {}, 1);

#घोषणा CREATE_GLU(name)  \
    ggml_vk_create_pipeline(device, device->pipeline_ ## name [0], #name "_f32", name ## _f32_len, name ## f32_तथ्य, "main", 3, माप(vk_op_glu_धकेलो_constants), {512, 1, 1}, {}, 1, true);   \
    ggml_vk_create_pipeline(device, device->pipeline_ ## name [1], #name "_f16", name ## _f16_len, name ## f16_तथ्य, "main", 3, माप(vk_op_glu_धकेलो_constants), {512, 1, 1}, {}, 1, true);

    CREATE_GLU(geglu)
    CREATE_GLU(reglu)
    CREATE_GLU(swiglu)
    CREATE_GLU(swiglu_oai)
    CREATE_GLU(swiglu_clamp)
    CREATE_GLU(geglu_erf)
    CREATE_GLU(geglu_quick)
#अघोषित CREATE_GLU

    ggml_vk_create_pipeline(device, device->pipeline_silu_back_f32, "silu_back_f32", silu_back_f32_len, silu_back_f32_तथ्य, "main", 3, माप(vk_op_धकेलो_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_diag_mask_inf_f32, "diag_mask_inf_f32", diag_mask_inf_f32_len, diag_mask_inf_f32_तथ्य, "main", 2, माप(vk_op_diag_mask_धकेलो_constants), {1, 512, 1}, {}, 1, true);

    ggml_vk_create_pipeline(device, device->pipeline_soft_max_f32, "soft_max_f32", soft_max_f32_len, soft_max_f32_तथ्य, "main", 4, माप(vk_op_soft_max_धकेलो_constants), {1, 1, 1}, { device->subgroup_आकार }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_f32_wg512, "soft_max_f32_wg512", soft_max_f32_len, soft_max_f32_तथ्य, "main", 4, माप(vk_op_soft_max_धकेलो_constants), {1, 1, 1}, { 512 }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_f32_f16, "soft_max_f32_f16", soft_max_f32_f16_len, soft_max_f32_f16_तथ्य, "main", 4, माप(vk_op_soft_max_धकेलो_constants), {1, 1, 1}, { device->subgroup_आकार }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_f32_f16_wg512, "soft_max_f32_f16_wg512", soft_max_f32_f16_len, soft_max_f32_f16_तथ्य, "main", 4, माप(vk_op_soft_max_धकेलो_constants), {1, 1, 1}, { 512 }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_back_f32, "soft_max_back_f32", soft_max_back_f32_len, soft_max_back_f32_तथ्य, "main", 3, माप(vk_op_धकेलो_constants), {1, 1, 1}, { device->subgroup_आकार }, 1, true);

    ggml_vk_create_pipeline(device, device->pipeline_soft_max_large1_f32,     "soft_max_large1_f32",     soft_max_large1_f32_len,     soft_max_large1_f32_तथ्य,     "main", 6, माप(vk_op_soft_max_धकेलो_constants), {1, 1, 1}, { 128, 4 }, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_large2_f32,     "soft_max_large2_f32",     soft_max_large2_f32_len,     soft_max_large2_f32_तथ्य,     "main", 6, माप(vk_op_soft_max_धकेलो_constants), {1, 1, 1}, { 128, 4 }, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_large3_f32,     "soft_max_large3_f32",     soft_max_large3_f32_len,     soft_max_large3_f32_तथ्य,     "main", 6, माप(vk_op_soft_max_धकेलो_constants), {1, 1, 1}, { 128, 4 }, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_large1_f32_f16, "soft_max_large1_f32_f16", soft_max_large1_f32_f16_len, soft_max_large1_f32_f16_तथ्य, "main", 6, माप(vk_op_soft_max_धकेलो_constants), {1, 1, 1}, { 128, 4 }, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_large2_f32_f16, "soft_max_large2_f32_f16", soft_max_large2_f32_f16_len, soft_max_large2_f32_f16_तथ्य, "main", 6, माप(vk_op_soft_max_धकेलो_constants), {1, 1, 1}, { 128, 4 }, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_large3_f32_f16, "soft_max_large3_f32_f16", soft_max_large3_f32_f16_len, soft_max_large3_f32_f16_तथ्य, "main", 6, माप(vk_op_soft_max_धकेलो_constants), {1, 1, 1}, { 128, 4 }, 1, true);

    ggml_vk_create_pipeline(device, device->pipeline_rope_norm_f32, "rope_norm_f32", rope_norm_f32_len, rope_norm_f32_तथ्य, "main", 5, माप(vk_op_rope_धकेलो_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_neox_f32, "rope_neox_f32", rope_neox_f32_len, rope_neox_f32_तथ्य, "main", 5, माप(vk_op_rope_धकेलो_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_multi_f32, "rope_multi_f32", rope_multi_f32_len, rope_multi_f32_तथ्य, "main", 5, माप(vk_op_rope_धकेलो_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_vision_f32, "rope_vision_f32", rope_vision_f32_len, rope_vision_f32_तथ्य, "main", 5, माप(vk_op_rope_धकेलो_constants), {1, 512, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_rope_norm_f16, "rope_norm_f16", rope_norm_f16_len, rope_norm_f16_तथ्य, "main", 5, माप(vk_op_rope_धकेलो_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_neox_f16, "rope_neox_f16", rope_neox_f16_len, rope_neox_f16_तथ्य, "main", 5, माप(vk_op_rope_धकेलो_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_multi_f16, "rope_multi_f16", rope_multi_f16_len, rope_multi_f16_तथ्य, "main", 5, माप(vk_op_rope_धकेलो_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_vision_f16, "rope_vision_f16", rope_vision_f16_len, rope_vision_f16_तथ्य, "main", 5, माप(vk_op_rope_धकेलो_constants), {1, 512, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_rope_norm_f32_f16, "rope_norm_f32_f16", rope_norm_f32_f16_len, rope_norm_f32_f16_तथ्य, "main", 5, माप(vk_op_rope_धकेलो_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_neox_f32_f16, "rope_neox_f32_f16", rope_neox_f32_f16_len, rope_neox_f32_f16_तथ्य, "main", 5, माप(vk_op_rope_धकेलो_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_multi_f32_f16, "rope_multi_f32_f16", rope_multi_f32_f16_len, rope_multi_f32_f16_तथ्य, "main", 5, माप(vk_op_rope_धकेलो_constants), {1, 512, 1}, {}, 1);

    क्रम (uint32_t i = 0; i < अंक_argsort_pipelines; ++i) {
        uint32_t BLOCK_आकार = 1u << std::min(i, device->max_workgroup_आकार_log2);
        अगर (i <= device->max_workgroup_आकार_log2 &&
            2 * माप(पूर्णांक) * BLOCK_आकार <= device->properties.limits.max_Compute_Shared_Memory_आकार) {
            स्थिर uint32_t NCOLS_PADDED_LOG2 = i;
            ggml_vk_create_pipeline2(device, device->pipeline_argsort_f32[i], "argsort_f32_"+std::to_string(i), argsort_f32_len, argsort_f32_तथ्य, "main", 3, माप(vk_op_argsort_धकेलो_constants), {BLOCK_आकार, 1, 1}, {BLOCK_आकार, NCOLS_PADDED_LOG2}, 1, true);
        }
        स्थिर uint32_t WG_UNROLL_FACTOR = BLOCK_आकार > 1 ? 2 : 1;
        BLOCK_आकार /= WG_UNROLL_FACTOR;
        ggml_vk_create_pipeline2(device, device->pipeline_argsort_large_f32[i], "argsort_large_f32_"+std::to_string(i), argsort_large_f32_len, argsort_large_f32_तथ्य, "main", 3, माप(vk_op_argsort_धकेलो_constants), {BLOCK_आकार * WG_UNROLL_FACTOR, 1, 1}, {BLOCK_आकार, WG_UNROLL_FACTOR}, 1, true);
    }

    क्रम (uint32_t i = 0; i < अंक_topk_pipelines; ++i) {
        स्थिर uint32_t BLOCK_आकार = 1u << i;
        स्थिर uint32_t NCOLS_PADDED_LOG2 = i;
        अगर (i <= device->max_workgroup_आकार_log2) {
            uint32_t nary_shmem = 2 * माप(पूर्णांक) * BLOCK_आकार +
                                  माप(पूर्णांक) * device->subgroup_आकार +
                                  2 * माप(पूर्णांक) +
                                  2 * (BLOCK_आकार / device->subgroup_आकार) * माप(पूर्णांक);
            अगर (device->subgroup_arithmetic && device->subgroup_require_full_support && device->subgroup_shuffle && device->subgroup_ballot &&
                nary_shmem <= device->properties.limits.max_Compute_Shared_Memory_आकार) {
                ggml_vk_create_pipeline2(device, device->pipeline_topk_f32[i], "topk_f32_"+std::to_string(i), topk_nary_खोज_f32_len, topk_nary_खोज_f32_तथ्य, "main", 2, माप(vk_op_topk_धकेलो_constants), {BLOCK_आकार, 1, 1}, {BLOCK_आकार, device->subgroup_आकार, device->subgroup_आकार_log2}, 1, true, true, device->subgroup_आकार);
            } अन्यथा अगर (2 * माप(पूर्णांक) * BLOCK_आकार <= device->properties.limits.max_Compute_Shared_Memory_आकार) {
                ggml_vk_create_pipeline2(device, device->pipeline_topk_f32[i], "topk_f32_"+std::to_string(i), topk_argsort_f32_len, topk_argsort_f32_तथ्य, "main", 2, माप(vk_op_topk_धकेलो_constants), {BLOCK_आकार, 1, 1}, {BLOCK_आकार, NCOLS_PADDED_LOG2}, 1, true);
            }
        }
    }

    // large-k fallback: one workgroup per row, radix-select instead of a full sort. The QSA
    // variant (spec constant 1) additionally gathers the qwen4 indexer input on the fly.
    {
        स्थिर uint32_t BLOCK_आकार = 1u << std::min(10u, device->max_workgroup_आकार_log2);
        ggml_vk_create_pipeline2(device, device->pipeline_topk_radix_f32, "topk_radix_f32", topk_radix_select_f32_len, topk_radix_select_f32_तथ्य, "main", 5, माप(vk_op_topk_radix_धकेलो_constants), {BLOCK_आकार, 1, 1}, {BLOCK_आकार, 0}, 1, true);
        ggml_vk_create_pipeline2(device, device->pipeline_topk_radix_qsa, "topk_radix_qsa", topk_radix_select_f32_len, topk_radix_select_f32_तथ्य, "main", 5, माप(vk_op_topk_radix_धकेलो_constants), {BLOCK_आकार, 1, 1}, {BLOCK_आकार, 1}, 1, true);
    }

    ggml_vk_create_pipeline(device, device->pipeline_argmax_f32, "argmax_f32", argmax_f32_len, argmax_f32_तथ्य, "main", 2, माप(vk_op_धकेलो_constants), {1, 1, 1}, { device->subgroup_आकार }, 1);

    ggml_vk_create_pipeline(device, device->pipeline_योग_rows_f32, "sum_rows_f32", योग_rows_f32_len, योग_rows_f32_तथ्य, "main", 2, माप(vk_op_योग_rows_धकेलो_constants), {1, 1, 1}, { device->subgroup_आकार }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cross_entropy_loss_f32, "cross_entropy_loss_f32", cross_entropy_loss_f32_len, cross_entropy_loss_f32_तथ्य, "main", 3, माप(vk_op_धकेलो_constants), {1, 1, 1}, { device->subgroup_आकार }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cross_entropy_loss_f32_wg512, "cross_entropy_loss_f32_wg512", cross_entropy_loss_f32_len, cross_entropy_loss_f32_तथ्य, "main", 3, माप(vk_op_धकेलो_constants), {1, 1, 1}, { 512 }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cross_entropy_loss_back_f32, "cross_entropy_loss_back_f32", cross_entropy_loss_back_f32_len, cross_entropy_loss_back_f32_तथ्य, "main", 4, माप(vk_op_धकेलो_constants), {1, 1, 1}, { device->subgroup_आकार }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cross_entropy_loss_back_f32_wg512, "cross_entropy_loss_back_f32_wg512", cross_entropy_loss_back_f32_len, cross_entropy_loss_back_f32_तथ्य, "main", 4, माप(vk_op_धकेलो_constants), {1, 1, 1}, { 512 }, 1);
    // Intel Windows driver in range [32.0.101.8509, 32.0.101.8860) will crash when using fwht kernels so we gate that here
    स्थिर bool can_use_fwht = device->driver_id != vk::DriverId::eIntelProprietaryWindows ||
        !ggml_vk_intel_windows_driver_in_range(device->properties.driverVersion, 101, 8509, 101, 8860);
    अगर (can_use_fwht && device->subgroup_basic && device->subgroup_shuffle) {
        पूर्णांक idx = 0;
        क्रम (uint32_t n : {64, 128, 256, 512}) {
            अगर (device->subgroup_आकार <= n) {
                ggml_vk_create_pipeline(device, device->pipeline_fwht_f32[idx], "fwht_f32", fwht_f32_len, fwht_f32_तथ्य, "main", 2, माप(vk_op_fwht_धकेलो_constants), {1, 1, 1}, { device->subgroup_आकार, n }, 1, true, true, device->subgroup_आकार);
            }
            ++idx;
        }
    } अन्यथा अगर (can_use_fwht) {
        पूर्णांक idx = 0;
        क्रम (uint32_t n : {64, 128, 256, 512}) {
            स्थिर uint32_t block_आकार_2 = std::min(device->subgroup_आकार, n);
            ggml_vk_create_pipeline(device, device->pipeline_fwht_f32[idx], "fwht_shmem_f32", fwht_shmem_f32_len, fwht_shmem_f32_तथ्य, "main", 2, माप(vk_op_fwht_धकेलो_constants), {1, 1, 1}, { block_आकार_2, n }, 1);
            ++idx;
        }
    }

    स्थिर uint32_t cumsum_elem_per_thread = (device->vendor_id == VK_VENDOR_ID_AMD || device->vendor_id == VK_VENDOR_ID_INTEL) ? 2 : 4;
    ggml_vk_create_pipeline(device, device->pipeline_cumsum_f32,       "cumsum_f32", cumsum_f32_len, cumsum_f32_तथ्य, "main", 2, माप(vk_op_योग_rows_धकेलो_constants), {1, 1, 1}, { 256, device->subgroup_आकार, cumsum_elem_per_thread }, 1, true, true, device->subgroup_आकार);
    ggml_vk_create_pipeline(device, device->pipeline_cumsum_small_f32, "cumsum_f32", cumsum_f32_len, cumsum_f32_तथ्य, "main", 2, माप(vk_op_योग_rows_धकेलो_constants), {1, 1, 1}, { 128, device->subgroup_आकार, 1 }, 1, true, true, device->subgroup_आकार);
    ggml_vk_create_pipeline(device, device->pipeline_cumsum_multipass1_f32, "cumsum_multipass1_f32", cumsum_multipass1_f32_len, cumsum_multipass1_f32_तथ्य, "main", 3, माप(vk_op_योग_rows_धकेलो_constants), {256, 1, 1}, { 256, device->subgroup_आकार }, 1, true, true, device->subgroup_आकार);
    ggml_vk_create_pipeline(device, device->pipeline_cumsum_multipass2_f32, "cumsum_multipass2_f32", cumsum_multipass2_f32_len, cumsum_multipass2_f32_तथ्य, "main", 3, माप(vk_op_योग_rows_धकेलो_constants), {256, 1, 1}, { 256, device->subgroup_आकार }, 1, true, true, device->subgroup_आकार);

    ggml_vk_create_pipeline(device, device->pipeline_गिनती_equal_i32, "count_equal_i32", गिनती_equal_i32_len, गिनती_equal_i32_तथ्य, "main", 3, माप(vk_op_धकेलो_constants), {512, 1, 1}, { device->subgroup_आकार }, 1);

    अगर (device->subgroup_arithmetic && device->subgroup_vote && device->subgroup_require_full_support) {
        ggml_vk_create_pipeline(device, device->pipeline_गिनती_experts, "count_experts", गिनती_experts_subgroup_len, गिनती_experts_subgroup_तथ्य, "main", 2, माप(vk_op_गिनती_experts_धकेलो_constants), {1, 1, 1}, {}, 1, true, true);
    } अन्यथा {
        ggml_vk_create_pipeline(device, device->pipeline_गिनती_experts, "count_experts", गिनती_experts_len, गिनती_experts_तथ्य, "main", 2, माप(vk_op_गिनती_experts_धकेलो_constants), {1, 1, 1}, {}, 1, true);
    }

    // comb holds a token's 4x4 matrix in one 16-lane slice of a subgroup, so it
    // needs at least 16 lanes, pinned to a known size.
    अगर (device->subgroup_basic && device->subgroup_shuffle && device->subgroup_require_full_support && device->subgroup_आकार >= 16) {
        स्थिर uint32_t tokens_per_workgroup = 4 * (device->subgroup_आकार / 16);
        ggml_vk_create_pipeline(device, device->pipeline_dsv4_hc_comb_f32, "dsv4_hc_comb_f32", dsv4_hc_comb_f32_len, dsv4_hc_comb_f32_तथ्य, "main", 4, माप(vk_op_dsv4_hc_comb_धकेलो_constants), {tokens_per_workgroup, 1, 1}, { device->subgroup_आकार }, 1, true, true, device->subgroup_आकार);
    }

    ggml_vk_create_pipeline(device, device->pipeline_dsv4_hc_pre_f32,        "dsv4_hc_pre_f32",        dsv4_hc_pre_f32_len,  dsv4_hc_pre_f32_तथ्य,  "main", 3, माप(vk_op_dsv4_hc_pre_धकेलो_constants),  {256, 1, 1}, { 256, 0 }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dsv4_hc_pre_gated_f32,  "dsv4_hc_pre_gated_f32",  dsv4_hc_pre_f32_len,  dsv4_hc_pre_f32_तथ्य,  "main", 3, माप(vk_op_dsv4_hc_pre_धकेलो_constants),  {256, 1, 1}, { 256, 1 }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dsv4_hc_post_f32,       "dsv4_hc_post_f32",       dsv4_hc_post_f32_len, dsv4_hc_post_f32_तथ्य, "main", 5, माप(vk_op_dsv4_hc_post_धकेलो_constants), {256, 1, 1}, { 256, 1 }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dsv4_hc_post_nocomb_f32,"dsv4_hc_post_nocomb_f32",dsv4_hc_post_f32_len, dsv4_hc_post_f32_तथ्य, "main", 5, माप(vk_op_dsv4_hc_post_धकेलो_constants), {256, 1, 1}, { 256, 0 }, 1);

    क्रम (स्वतः &s : device->pipeline_solve_tri_f32) {
        स्थिर vk_solve_tri_pipeline_state &state = s.first;

        // Max number of rows to load at a time, limited by shared memory
        स्थिर uint32_t batch_N = device->properties.limits.max_Compute_Shared_Memory_आकार / ((state.N + state.K) * माप(भग्न));
        // Need at least K invocations, and prefer a minimum of 128 to spread out loading shared memory
        स्थिर uint32_t block_आकार_2 = std::max(128u, 1u << (uint32_t)ceilf(log2f(भग्न(state.K))));

        ggml_vk_create_pipeline(
            device, s.second, "solve_tri_f32",
            solve_tri_f32_len, solve_tri_f32_तथ्य, "main", 3,
            माप(vk_op_द्विआधारी_धकेलो_constants), {1, 1, 1}, { 0, state.N, state.K, batch_N, block_आकार_2 }, 1, true);
    }

#घोषणा IM2COL(bda) \
    ggml_vk_create_pipeline(device, device->pipeline_im2col_f32, "im2col_f32", im2col_f32 ## bda ## _len, im2col_f32 ## bda ## तथ्य_2, "main", 2, माप(vk_op_im2col_धकेलो_constants), {512, 1, 1}, { device->subgroup_आकार }, 1, true);   \
    ggml_vk_create_pipeline(device, device->pipeline_im2col_3d_f32, "im2col_3d_f32", im2col_3d_f32 ## bda ## _len, im2col_3d_f32 ## bda ## तथ्य_2, "main", 2, माप(vk_op_im2col_3d_धकेलो_constants), {512, 1, 1}, { 512 }, 1, true);      \
    ggml_vk_create_pipeline(device, device->pipeline_im2col_f32_f16, "im2col_f32_f16", im2col_f32_f16 ## bda ## _len, im2col_f32_f16 ## bda ## तथ्य_2, "main", 2, माप(vk_op_im2col_धकेलो_constants), {512, 1, 1}, { device->subgroup_आकार }, 1, true);   \
    ggml_vk_create_pipeline(device, device->pipeline_im2col_3d_f32_f16, "im2col_3d_f32_f16", im2col_3d_f32_f16 ## bda ## _len, im2col_3d_f32_f16 ## bda ## तथ्य_2, "main", 2, माप(vk_op_im2col_3d_धकेलो_constants), {512, 1, 1}, { 512 }, 1, true);
    अगर (device->shader_int64 && device->बफर_device_address) {
        IM2COL(_bda)
    } अन्यथा {
        IM2COL()
    }

    ggml_vk_create_pipeline(device, device->pipeline_timestep_embedding_f32, "timestep_embedding_f32", timestep_embedding_f32_len, timestep_embedding_f32_तथ्य, "main", 2, माप(vk_op_timestep_embedding_धकेलो_constants), {256, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_conv_transpose_1d_f32, "conv_transpose_1d_f32", conv_transpose_1d_f32_len, conv_transpose_1d_f32_तथ्य, "main", 3, माप(vk_op_conv_transpose_1d_धकेलो_constants), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_col2im_1d_f32,  "col2im_1d_f32",  col2im_1d_f32_len,  col2im_1d_f32_तथ्य,  "main", 2, माप(vk_op_col2im_1d_धकेलो_constants), {256, 1, 1}, {}, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_col2im_1d_f16,  "col2im_1d_f16",  col2im_1d_f16_len,  col2im_1d_f16_तथ्य,  "main", 2, माप(vk_op_col2im_1d_धकेलो_constants), {256, 1, 1}, {}, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_col2im_1d_bf16, "col2im_1d_bf16", col2im_1d_bf16_len, col2im_1d_bf16_तथ्य, "main", 2, माप(vk_op_col2im_1d_धकेलो_constants), {256, 1, 1}, {}, 1, true);

    ggml_vk_create_pipeline(device, device->pipeline_out_prod_f32, "out_prod_f32", out_prod_f32_len, out_prod_f32_तथ्य, "main", 3, माप(vk_op_द्विआधारी_धकेलो_constants), {256, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_snake_f32,  "snake_f32",  snake_f32_len,  snake_f32_तथ्य,  "main", 4, माप(vk_op_snake_धकेलो_constants), {256, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_snake_f16,  "snake_f16",  snake_f16_len,  snake_f16_तथ्य,  "main", 4, माप(vk_op_snake_धकेलो_constants), {256, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_snake_bf16, "snake_bf16", snake_bf16_len, snake_bf16_तथ्य, "main", 4, माप(vk_op_snake_धकेलो_constants), {256, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_pool1d_f32, "pool1d_f32", pool1d_f32_len, pool1d_f32_तथ्य, "main", 2, माप(vk_op_pool1d_धकेलो_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_pool2d_f32, "pool2d_f32", pool2d_f32_len, pool2d_f32_तथ्य, "main", 2, माप(vk_op_pool2d_धकेलो_constants), {1, 1, 1}, {512}, 1);
    अगर (device->subgroup_arithmetic && device->subgroup_require_full_support) {
        ggml_vk_create_pipeline(device, device->pipeline_pool2d_subgroup_f32, "pool2d_subgroup_f32", pool2d_subgroup_f32_len, pool2d_subgroup_f32_तथ्य, "main", 2, माप(vk_op_pool2d_धकेलो_constants), {1, 1, 1}, {128}, 1, false, true, device->subgroup_आकार);
    }

    ggml_vk_create_pipeline(device, device->pipeline_rwkv_wkv6_f32, "rwkv_wkv6_f32", rwkv_wkv6_f32_len, rwkv_wkv6_f32_तथ्य, "main", 7, माप(vk_op_rwkv_wkv6_धकेलो_constants), {1, 1, 1}, {device->subgroup_आकार}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_rwkv_wkv7_f32, "rwkv_wkv7_f32", rwkv_wkv7_f32_len, rwkv_wkv7_f32_तथ्य, "main", 8, माप(vk_op_rwkv_wkv7_धकेलो_constants), {1, 1, 1}, {device->subgroup_आकार}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_gated_linear_attn_f32, "gated_linear_attn_f32", gated_linear_attn_f32_len, gated_linear_attn_f32_तथ्य, "main", 6, माप(vk_op_gated_linear_attn_धकेलो_constants), {1, 1, 1}, {}, 1);

    क्रम (ggml_type k_type : lightning_indexer_k_types) {
        स्थिर std::string name = "lightning_indexer_" + std::string(ggml_type_name(k_type)) + "_k_f32";
        ggml_vk_create_pipeline(device, device->pipeline_lightning_indexer_f32[k_type], name.c_str(), lightning_indexer_f32_len, lightning_indexer_f32_तथ्य, "main", 5, माप(vk_op_lightning_indexer_धकेलो_constants), {1, 1, 1}, {(uint32_t)k_type, fa_block_bytes(k_type)}, 1, true);
    }

    {
        स्थिर uint32_t gdn_sizes[] = {16, 32, 64, 128};
        स्थिर अक्षर * gdn_names[][2] = {
            {"gated_delta_net_f32_d16",     "gated_delta_net_f32_d16_kda"},
            {"gated_delta_net_f32_d32",     "gated_delta_net_f32_d32_kda"},
            {"gated_delta_net_f32_d64",     "gated_delta_net_f32_d64_kda"},
            {"gated_delta_net_f32_d128",    "gated_delta_net_f32_d128_kda"},
        };
        क्रम (uint32_t si = 0; si < 4; si++) {
            स्थिर uint32_t S_V = gdn_sizes[si];
            GGML_ASSERT(is_pow2(S_V));

            // Intel Xe regresses at SIMD32 for this scan; prefer a narrower subgroup.
            uint32_t gdn_subgroup_आकार = device->subgroup_आकार;
            अगर (device->vendor_id == VK_VENDOR_ID_INTEL && device->subgroup_आकार_control &&
                device->subgroup_min_आकार <= 16u && device->subgroup_max_आकार >= 16u) {
                gdn_subgroup_आकार = 16u;
            }
            uint32_t lanes_per_column;
            अगर (device->vendor_id == VK_VENDOR_ID_INTEL) {
                // Intel Xe: full-width reduction (min rows/lane) ~10x over the COLS_PER_WG=8 rule.
                lanes_per_column = std::min(gdn_subgroup_आकार, S_V);
            } अन्यथा अगर (S_V >= 128u && device->subgroup_clustered) {
                // COLS_PER_WG=8: measured optimum (Ampere sg32->4, Vega20 sg64->8).
                lanes_per_column = std::max(1u, gdn_subgroup_आकार / 8u);
            } अन्यथा {
                // Use largest power-of-two that divides both S_V and subgroup_size so that
                // (1) S_V % lanes_per_column == 0 and (2) S_V % (subgroup_size / lanes_per_column) == 0.
                // This means we don't need extra bounds checking logic in the shader.
                lanes_per_column = std::min(S_V, gdn_subgroup_आकार);
            }

            // gated_delta_net.comp relies on S_V % COLS_PER_WG == 0 and
            // S_V % LANES_PER_COLUMN == 0 to avoid bounds checks.
            जबतक (lanes_per_column > 1u) {
                स्थिर bool valid_lanes = (gdn_subgroup_आकार % lanes_per_column) == 0 &&
                                         (S_V % lanes_per_column) == 0;
                स्थिर uint32_t cols_per_wg = valid_lanes ? gdn_subgroup_आकार / lanes_per_column : 0;
                अगर (valid_lanes && cols_per_wg > 0 && (S_V % cols_per_wg) == 0) {
                    अवरोध;
                }
                lanes_per_column >>= 1u;
            }

            GGML_ASSERT((gdn_subgroup_आकार % lanes_per_column) == 0);
            GGML_ASSERT((S_V % lanes_per_column) == 0);
            GGML_ASSERT((S_V % (gdn_subgroup_आकार / lanes_per_column)) == 0);

            स्थिर bool need_partial_subgroup_reduce = lanes_per_column != 1u && lanes_per_column < gdn_subgroup_आकार;
            स्थिर bool use_clustered_reduce = device->subgroup_arithmetic && device->subgroup_clustered && need_partial_subgroup_reduce;
            स्थिर bool use_subgroup_reduce = device->subgroup_arithmetic && !need_partial_subgroup_reduce;
            स्थिर bool use_subgroup_ops = use_clustered_reduce || use_subgroup_reduce;
            माप_प्रकार gdn_len;
            स्थिर व्योम * gdn_तथ्य;
            अगर (use_clustered_reduce) {
                gdn_len = gated_delta_net_f32_len;
                gdn_तथ्य = (स्थिर व्योम *)gated_delta_net_f32_तथ्य;
            } अन्यथा अगर (use_subgroup_reduce) {
                gdn_len = gated_delta_net_f32_nocluster_len;
                gdn_तथ्य = (स्थिर व्योम *)gated_delta_net_f32_nocluster_तथ्य;
            } अन्यथा {
                gdn_len = gated_delta_net_f32_shmem_len;
                gdn_तथ्य = (स्थिर व्योम *)gated_delta_net_f32_shmem_तथ्य;
            }

            स्थिर uint32_t cols_per_wg = gdn_subgroup_आकार / lanes_per_column;
            स्थिर std::array<uint32_t, 3> wg_denoms = {1u, 1u, cols_per_wg};

            क्रम (uint32_t kda = 0; kda < 2; kda++) {
                ggml_vk_create_pipeline(device, device->pipeline_gated_delta_net[si][kda],
                    gdn_names[si][kda], gdn_len, gdn_तथ्य, "main", 7, माप(vk_op_gated_delta_net_धकेलो_constants),
                    wg_denoms, {S_V, kda, gdn_subgroup_आकार, lanes_per_column}, 1, true, use_subgroup_ops, gdn_subgroup_आकार);
            }
        }
    }

    अगर (device->subgroup_arithmetic && device->subgroup_require_full_support) {
        ggml_vk_create_pipeline(device, device->pipeline_ssm_scan_f32_d128, "ssm_scan_128_f32", ssm_scan_subgroup_f32_len, ssm_scan_subgroup_f32_तथ्य, "main", 8, माप(vk_op_ssm_scan_धकेलो_constants), {1, 1, 1}, {128, device->subgroup_आकार}, 1, true, true);
        ggml_vk_create_pipeline(device, device->pipeline_ssm_scan_f32_d256, "ssm_scan_256_f32", ssm_scan_subgroup_f32_len, ssm_scan_subgroup_f32_तथ्य, "main", 8, माप(vk_op_ssm_scan_धकेलो_constants), {1, 1, 1}, {256, device->subgroup_आकार}, 1, true, true);
    } अन्यथा {
        ggml_vk_create_pipeline(device, device->pipeline_ssm_scan_f32_d128, "ssm_scan_128_f32", ssm_scan_f32_len, ssm_scan_f32_तथ्य, "main", 8, माप(vk_op_ssm_scan_धकेलो_constants), {1, 1, 1}, {128, device->subgroup_आकार, 16}, 1, true, true);
        ggml_vk_create_pipeline(device, device->pipeline_ssm_scan_f32_d256, "ssm_scan_256_f32", ssm_scan_f32_len, ssm_scan_f32_तथ्य, "main", 8, माप(vk_op_ssm_scan_धकेलो_constants), {1, 1, 1}, {256, device->subgroup_आकार, 16}, 1, true, true);
    }

    ggml_vk_create_pipeline(device, device->pipeline_ssm_conv_f32,           "ssm_conv_f32",           ssm_conv_f32_len, ssm_conv_f32_तथ्य, "main", 4, माप(vk_op_ssm_conv_धकेलो_constants), {32, 16, 1}, {32, 16, 0, 0}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_ssm_conv_silu_f32,      "ssm_conv_silu_f32",      ssm_conv_f32_len, ssm_conv_f32_तथ्य, "main", 4, माप(vk_op_ssm_conv_धकेलो_constants), {32, 16, 1}, {32, 16, 0, 1}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_ssm_conv_bias_silu_f32, "ssm_conv_bias_silu_f32", ssm_conv_f32_len, ssm_conv_f32_तथ्य, "main", 4, माप(vk_op_ssm_conv_धकेलो_constants), {32, 16, 1}, {32, 16, 1, 1}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_opt_step_adamw_f32, "opt_step_adamw_f32", opt_step_adamw_f32_len, opt_step_adamw_f32_तथ्य, "main", 5, माप(vk_op_धकेलो_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_opt_step_sgd_f32, "opt_step_sgd_f32", opt_step_sgd_f32_len, opt_step_sgd_f32_तथ्य, "main", 3, माप(vk_op_धकेलो_constants), {512, 1, 1}, {}, 1);

    // conv2d, conv_transpose_2d, conv3d
    क्रम (uint32_t s = 0; s < CONV_SHAPE_गिनती; ++s) {
        // smaller WG for the small-tile fallback gives more concurrent WGs per SM
        uint32_t conv2d_WG_आकार  = (s == CONV_SHAPE_64x32) ? 128 : 256;
        uint32_t use_collectives = 0;  // Enables subgroup ops for preventing the re-calculation of indices.
        uint32_t conv2d_TS_K     = (s == CONV_SHAPE_64x32) ? 4 : 8;
        uint32_t conv2d_SHMEM_PAD = 4;
        vk_conv_block_आकार conv2d_BS = vk_conv_block_sizes[s];
        bool conv2d_UNROLL = true;

#अगर defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
        अगर (device->coopmat2) {
            conv2d_SHMEM_PAD = 8; // 8 float16_t
        }
#पूर्ण_अगर

        अगर (device->vendor_id == VK_VENDOR_ID_INTEL) {
            conv2d_SHMEM_PAD = 0;
            conv2d_UNROLL = false;
        } अन्यथा अगर (device->vendor_id == VK_VENDOR_ID_AMD) {
            conv2d_SHMEM_PAD = device->architecture == vk_device_architecture::AMD_GCN ? 1 : 4;
            अगर (s == CONV_SHAPE_128x128 && device->architecture != vk_device_architecture::AMD_GCN) {
                conv2d_UNROLL = false;
            }
        }

        // Use collectives on pre-Turing NVIDIA GPUs and GCN AMD cards, which had slower integer math.
        bool allow_collectives_nv = device->vendor_id != VK_VENDOR_ID_NVIDIA ||
                                    device->architecture == vk_device_architecture::NVIDIA_PRE_TURING;
        bool allow_collectives_amd = device->vendor_id != VK_VENDOR_ID_AMD ||
                                     device->architecture == vk_device_architecture::AMD_GCN;

        अगर (device->subgroup_shuffle &&
            device->vendor_id != VK_VENDOR_ID_INTEL &&   // Do not enable collectives on Intel, see PR 14316.
            allow_collectives_nv &&
            allow_collectives_amd) {
            use_collectives = 1;
            conv2d_BS.CRS   = std::min(
                device->subgroup_आकार,
                conv2d_BS.CRS);  // CRS block size should be capped at subgroup size for correctness when shuffle is used.
        }

        // cm1 is used only when cm2 is unavailable; capped at 64x128 (due to shared memory size).
        // Requires 16x16x16 f16-acc since that's the fragment shape hard-coded in the shader.
        // Subgroup size must be 32 or 64 (to keep WG_SIZE sane) and we need
        // subgroup_size_control to force the driver to actually use it.
        bool conv2d_use_cm1 = false;
#अगर defined(VK_KHR_cooperative_matrix) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
        conv2d_use_cm1 = !device->coopmat2 &&
                         device->coopmat_support && device->coopmat_support_16x16x16_f16acc &&
                         device->subgroup_आकार_control &&
                         (device->subgroup_आकार == 32 || device->subgroup_आकार == 64) &&
                         s != CONV_SHAPE_128x128;
#पूर्ण_अगर

        स्थिर uint32_t conv2d_cm1_shmem_pad = 8;

        स्वतः shmem_req = [&](uint32_t pad, bool csh_store, bool fp16_shmem) {
            स्थिर uint32_t elem_आकार = fp16_shmem ? (uint32_t)माप(uint16_t) : (uint32_t)माप(भग्न);
            स्थिर uint32_t csh_elems = csh_store ? conv2d_BS.K * conv2d_BS.NPQ : 0u;
            वापस (conv2d_BS.K * (conv2d_BS.CRS + pad) + conv2d_BS.CRS * (conv2d_BS.NPQ + pad) + csh_elems) * elem_आकार;
        };

        // 2D, transpose-2D, and 3D conv use the same KxCRS @ CRSxNPQ shmem
        // layout. cm1 needs Csh for output, so check before applying cm1 params.
        अगर (conv2d_use_cm1 && device->properties.limits.max_Compute_Shared_Memory_आकार < shmem_req(conv2d_cm1_shmem_pad, true, true)) {
            conv2d_use_cm1 = false;
        }

        uint32_t conv2d_WM = 16, conv2d_WN = 16;  // cm1 subgroup tile, ignored otherwise
        अगर (conv2d_use_cm1) {
            conv2d_SHMEM_PAD = conv2d_cm1_shmem_pad;
            // 16x16x16 fragments; pick WM/WN to keep WG_SIZE at 256
            // (i.e. 8 subgroups for sg=32, 4 subgroups for sg=64).
            स्थिर bool sg64 = (device->subgroup_आकार == 64);
            चयन (s) {
                हाल CONV_SHAPE_64x32:   conv2d_WM = sg64 ? 32 : 16; conv2d_WN = 16; अवरोध;
                हाल CONV_SHAPE_64x128:  conv2d_WM = 32; conv2d_WN = sg64 ? 64 : 32; अवरोध;
                हाल CONV_SHAPE_32x256:  conv2d_WM = sg64 ? 16 : 32; conv2d_WN = sg64 ? 128 : 32; अवरोध;
                शेष: अवरोध;
            }
            स्थिर uint32_t warps_M = conv2d_BS.K / conv2d_WM;
            स्थिर uint32_t warps_N = conv2d_BS.NPQ / conv2d_WN;
            conv2d_WG_आकार         = warps_M * warps_N * device->subgroup_आकार;
        }

        // stage cm2 accumulator through shmem for coalesced global stores;
        // skipped on 128x128 where the extra Csh footprint hurts occupancy.
        // cm1 always uses the staged path.
        uint32_t conv2d_csh_store = (device->coopmat2 && s != CONV_SHAPE_128x128) ? 1u : 0u;
        अगर (conv2d_use_cm1) {
            conv2d_csh_store = 1;
        }

        // shmem is fp16 on cm2/cm1 (matches Csh), fp32 on scalar
        स्थिर bool conv2d_use_fp16_shmem = device->coopmat2 || conv2d_use_cm1;

        // shrink CRS if the non-cm1 config still doesn't fit
        अगर (device->properties.limits.max_Compute_Shared_Memory_आकार < shmem_req(conv2d_SHMEM_PAD, conv2d_csh_store, conv2d_use_fp16_shmem)) {
            GGML_ASSERT(!conv2d_use_cm1);
            conv2d_BS.CRS = 8;
            अगर (use_collectives) {
                conv2d_BS.CRS = std::min(device->subgroup_आकार, conv2d_BS.CRS);
            }
            conv2d_csh_store = 0;
        }

        std::array<uint32_t, 3> wg_denoms = { conv2d_BS.K, 1, 1 };
        std::vector<uint32_t> spec_constants = { conv2d_WG_आकार, conv2d_BS.K, conv2d_BS.CRS, conv2d_BS.NPQ, conv2d_TS_K, use_collectives, conv2d_SHMEM_PAD };

        // cm1 needs a fixed subgroup width to match the WG_SIZE we computed
        स्थिर uint32_t conv2d_required_subgroup_आकार = conv2d_use_cm1 ? device->subgroup_आकार : 0;

#घोषणा CREATE_CONV(name, type_suffix, spv_suffix) \
        क्रम (स्वतः &c : device->pipeline_##name##type_suffix[s]) { \
            स्थिर vk_conv2d_pipeline_state &state = c.first;  \
            std::vector<uint32_t> spec_constants_cpy = spec_constants; \
            spec_constants_cpy.push_back(state.s0); \
            spec_constants_cpy.push_back(state.s1); \
            spec_constants_cpy.push_back(state.p0); \
            spec_constants_cpy.push_back(state.p1); \
            spec_constants_cpy.push_back(state.d0); \
            spec_constants_cpy.push_back(state.d1); \
            spec_constants_cpy.push_back(state.KW); \
            spec_constants_cpy.push_back(state.KH); \
            spec_constants_cpy.push_back(state.aligned); \
            spec_constants_cpy.push_back(conv2d_csh_store); \
            spec_constants_cpy.push_back(conv2d_WM); \
            spec_constants_cpy.push_back(conv2d_WN); \
            ggml_vk_create_pipeline( \
                device, c.second, #name #type_suffix, \
                name##type_suffix##spv_suffix##_len, name##type_suffix##spv_suffix##_data, "main", 3, \
                माप(vk_op_conv2d_धकेलो_constants), wg_denoms, spec_constants_cpy, 1, true, use_collectives || conv2d_required_subgroup_आकार, conv2d_required_subgroup_आकार);    \
        }
#घोषणा CREATE_CONVS(spv_suffix) \
        CREATE_CONV(conv2d, _f32, spv_suffix) \
        CREATE_CONV(conv2d, _f16_f32, spv_suffix) \
        CREATE_CONV(conv_transpose_2d, _f32, spv_suffix) \
        CREATE_CONV(conv_transpose_2d, _f16_f32, spv_suffix)
#अगर defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
        अगर (device->coopmat2) {
            CREATE_CONVS(_cm2)
        } अन्यथा
#पूर्ण_अगर
#अगर defined(VK_KHR_cooperative_matrix) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
        अगर (conv2d_use_cm1) {
            CREATE_CONVS(_cm1)
        } अन्यथा
#पूर्ण_अगर
        अगर (conv2d_UNROLL) {
            CREATE_CONVS(_unroll)
        } अन्यथा {
            CREATE_CONVS( )
        }
#अघोषित CREATE_CONV
#अघोषित CREATE_CONVS

        std::vector<uint32_t> conv3d_spec_constants = { conv2d_WG_आकार, conv2d_BS.K, conv2d_BS.CRS, conv2d_BS.NPQ, conv2d_TS_K, conv2d_SHMEM_PAD };
#घोषणा CREATE_CONV3D(type_suffix, spv_suffix) \
        क्रम (स्वतः &c : device->pipeline_conv3d##type_suffix[s]) { \
            स्थिर vk_conv3d_pipeline_state &state = c.first; \
            std::vector<uint32_t> spec_constants_cpy = conv3d_spec_constants; \
            spec_constants_cpy.push_back(state.s0); \
            spec_constants_cpy.push_back(state.s1); \
            spec_constants_cpy.push_back(state.s2); \
            spec_constants_cpy.push_back(state.p0); \
            spec_constants_cpy.push_back(state.p1); \
            spec_constants_cpy.push_back(state.p2); \
            spec_constants_cpy.push_back(state.d0); \
            spec_constants_cpy.push_back(state.d1); \
            spec_constants_cpy.push_back(state.d2); \
            spec_constants_cpy.push_back(state.KW); \
            spec_constants_cpy.push_back(state.KH); \
            spec_constants_cpy.push_back(state.KD); \
            spec_constants_cpy.push_back(state.aligned); \
            spec_constants_cpy.push_back(conv2d_csh_store); \
            spec_constants_cpy.push_back(conv2d_WM); \
            spec_constants_cpy.push_back(conv2d_WN); \
            ggml_vk_create_pipeline( \
                device, c.second, "conv3d" #type_suffix, \
                conv3d##type_suffix##spv_suffix##_len, conv3d##type_suffix##spv_suffix##_data, "main", 3, \
                माप(vk_op_conv3d_धकेलो_constants), wg_denoms, spec_constants_cpy, 1, true, conv2d_required_subgroup_आकार != 0, conv2d_required_subgroup_आकार); \
        }
#अगर defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
        अगर (device->coopmat2) {
            CREATE_CONV3D(_f32, _cm2)
            CREATE_CONV3D(_f16_f32, _cm2)
        } अन्यथा
#पूर्ण_अगर
#अगर defined(VK_KHR_cooperative_matrix) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
        अगर (conv2d_use_cm1) {
            CREATE_CONV3D(_f32, _cm1)
            CREATE_CONV3D(_f16_f32, _cm1)
        } अन्यथा
#पूर्ण_अगर
        अगर (conv2d_UNROLL) {
            CREATE_CONV3D(_f32, _unroll)
            CREATE_CONV3D(_f16_f32, _unroll)
        } अन्यथा {
            CREATE_CONV3D(_f32, )
            CREATE_CONV3D(_f16_f32, )
        }
#अघोषित CREATE_CONV3D
    }

    ggml_vk_create_pipeline(device, device->pipeline_conv2d_dw_whcn_f32, "conv2d_dw_whcn_f32", conv2d_dw_whcn_f32_len, conv2d_dw_whcn_f32_तथ्य, "main", 3, माप(vk_op_conv2d_dw_धकेलो_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_conv2d_dw_cwhn_f32, "conv2d_dw_cwhn_f32", conv2d_dw_cwhn_f32_len, conv2d_dw_cwhn_f32_तथ्य, "main", 3, माप(vk_op_conv2d_dw_धकेलो_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_conv2d_dw_whcn_f16_f32, "conv2d_dw_whcn_f16_f32", conv2d_dw_whcn_f16_f32_len, conv2d_dw_whcn_f16_f32_तथ्य, "main", 3, माप(vk_op_conv2d_dw_धकेलो_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_conv2d_dw_cwhn_f16_f32, "conv2d_dw_cwhn_f16_f32", conv2d_dw_cwhn_f16_f32_len, conv2d_dw_cwhn_f16_f32_तथ्य, "main", 3, माप(vk_op_conv2d_dw_धकेलो_constants), {512, 1, 1}, {}, 1);

    क्रम (uint32_t use_धकेलो = 0; use_धकेलो < 2; ++use_धकेलो) {
        क्रम (uint32_t i = 0; i < अंक_topk_moe_pipelines; ++i) {
            ggml_vk_create_pipeline2(device, device->pipeline_topk_moe[i][use_धकेलो], "topk_moe_f32_"+std::to_string(i), topk_moe_f32_len, topk_moe_f32_तथ्य, "main", 4, माप(vk_op_topk_moe_धकेलो_constants), {1, 1, 1}, {device->subgroup_आकार, 1u<<i, use_धकेलो}, 1, true, true, device->subgroup_आकार);
        }
    }

    // Drop compile_mutex so other threads can walk while we compile.
    compile_lock.unlock();

    // Compile what we claimed; create_pipeline_func reacquires compile_mutex
    // at the end to flip compile_pending/compiled and notify waiters.
    अगर (has_claimed_task) {
        स्वतः & task = claimed_task;
        ggml_vk_create_pipeline_func(device, task.pipeline, task.spv_आकार, task.spv_तथ्य,
                                     task.entrypoint, task.parameter_गिनती, task.wg_denoms,
                                     task.specialization_constants, task.disable_robustness,
                                     task.require_full_subgroups, task.required_subgroup_आकार);
    }

    // Another thread may be compiling the pipeline we need; block on it here.
    अगर (wait_pipeline) {
        std::unique_lock<std::mutex> wait_lock(device->compile_mutex);
        device->compile_cv.रुको(wait_lock, [&] {
            वापस wait_pipeline->compiled.load();
        });
    }
}

अटल bool ggml_vk_external_memory_host_usable(vk::DriverId driver_id) {
    // Disable external_memory_host until https://github.com/KhronosGroup/MoltenVK/pull/2622
    // is available in the Vulkan SDK.
    वापस driver_id != vk::DriverId::eMoltenvk;
}

अटल bool ggml_vk_host_बफर_import_usable(uint32_t vendor_id) {
    // amdgpu re-faults all pages of every imported host allocation on each submission
    वापस vendor_id != VK_VENDOR_ID_AMD;
}

vk_device ggml_vk_get_device(माप_प्रकार idx) {
    VK_LOG_DEBUG("ggml_vk_get_device(" << idx << ")");

    अटल std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    अगर (vk_instance.devices[idx] == nullptr) {
        VK_LOG_DEBUG("Initializing new vk_device");
        vk_device device = std::make_shared<vk_device_struct>();

        device->memory_logger = std::unique_ptr<vk_memory_logger>(नया vk_memory_logger());

        माप_प्रकार dev_अंक = vk_instance.device_indices[idx];

        std::vector<vk::PhysicalDevice> physical_devices = vk_instance.instance.enumeratePhysicalDevices();

        अगर (dev_अंक >= physical_devices.size()) {
            std::अ_त्रुटि << "ggml_vulkan: Device with index " << dev_अंक << " does not exist." << std::प_पूर्ण;
            फेंको std::runtime_error("Device not found");
        }

        device->physical_device = physical_devices[dev_अंक];
        स्थिर std::vector<vk::ExtensionProperties> ext_props = device->physical_device.enumerateDeviceExtensionProperties();

        device->architecture = get_device_architecture(device->physical_device);

        स्थिर अक्षर* GGML_VK_PREFER_HOST_MEMORY = दो_पर्या("GGML_VK_PREFER_HOST_MEMORY");
        device->prefer_host_memory = GGML_VK_PREFER_HOST_MEMORY != nullptr;

        स्थिर अक्षर* GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM = दो_पर्या("GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM");
        device->disable_host_visible_vidmem = GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM != nullptr;

        स्थिर अक्षर* GGML_VK_ALLOW_SYSMEM_FALLBACK = दो_पर्या("GGML_VK_ALLOW_SYSMEM_FALLBACK");
        device->allow_sysmem_fallback = GGML_VK_ALLOW_SYSMEM_FALLBACK != nullptr;

        स्थिर अक्षर* GGML_VK_DISABLE_GRAPH_OPTIMIZE = दो_पर्या("GGML_VK_DISABLE_GRAPH_OPTIMIZE");
        device->disable_graph_optimize = GGML_VK_DISABLE_GRAPH_OPTIMIZE != nullptr;

        bool fp16_storage = false;
        bool fp16_compute = false;
        bool maintenance4_support = false;
        bool sm_builtins = false;
        bool amd_shader_core_properties2 = false;
        bool pipeline_robustness = false;
        bool coopmat2_support = false;
        bool coopmat2_decode_vector_support = false;
        bool pipeline_executable_properties_support = false;
        bool internally_sync_support = false;
        device->coopmat_support = false;
        device->integer_dot_product = false;
        device->shader_64b_indexing = false;
        bool bfloat16_support = false;
        bool dot2_f16_support = false;
        bool ocp_microscaling_extension = false;
        bool shader_float8_extension = false;

        क्रम (स्थिर स्वतः& properties : ext_props) {
            अगर (म_भेद("VK_KHR_maintenance4", properties.extensionName) == 0) {
                maintenance4_support = true;
            } अन्यथा अगर (म_भेद("VK_KHR_16bit_storage", properties.extensionName) == 0) {
                fp16_storage = true;
            } अन्यथा अगर (म_भेद("VK_KHR_shader_float16_int8", properties.extensionName) == 0) {
                fp16_compute = true;
            } अन्यथा अगर (म_भेद("VK_NV_shader_sm_builtins", properties.extensionName) == 0) {
                sm_builtins = true;
            } अन्यथा अगर (म_भेद("VK_AMD_shader_core_properties2", properties.extensionName) == 0) {
                amd_shader_core_properties2 = true;
            } अन्यथा अगर (म_भेद("VK_EXT_pipeline_robustness", properties.extensionName) == 0) {
                pipeline_robustness = true;
            } अन्यथा अगर (म_भेद("VK_EXT_subgroup_size_control", properties.extensionName) == 0) {
                device->subgroup_आकार_control = true;
#अगर defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
            } अन्यथा अगर (म_भेद("VK_KHR_cooperative_matrix", properties.extensionName) == 0 &&
                       !दो_पर्या("GGML_VK_DISABLE_COOPMAT")) {
                device->coopmat_support = true;
                device->coopmat_m = 0;
                device->coopmat_n = 0;
                device->coopmat_k = 0;
#पूर्ण_अगर
#अगर defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
            } अन्यथा अगर (म_भेद("VK_NV_cooperative_matrix2", properties.extensionName) == 0 &&
                       !दो_पर्या("GGML_VK_DISABLE_COOPMAT2")) {
                coopmat2_support = true;
#पूर्ण_अगर
            } अन्यथा अगर (म_भेद(VK_NV_COOPERATIVE_MATRIX_DECODE_VECTOR_EXTENSION_NAME, properties.extensionName) == 0 &&
                       !दो_पर्या("GGML_VK_DISABLE_COOPMAT2_DECODE_VECTOR")) {
                coopmat2_decode_vector_support = true;
#अगर defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
            } अन्यथा अगर (म_भेद("VK_KHR_shader_integer_dot_product", properties.extensionName) == 0 &&
                       !दो_पर्या("GGML_VK_DISABLE_INTEGER_DOT_PRODUCT")) {
                device->integer_dot_product = true;
#पूर्ण_अगर
#अगर defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
            } अन्यथा अगर (म_भेद("VK_KHR_shader_bfloat16", properties.extensionName) == 0 &&
                       !दो_पर्या("GGML_VK_DISABLE_BFLOAT16")) {
                bfloat16_support = true;
#पूर्ण_अगर
#अगर defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT)
            } अन्यथा अगर (म_भेद(VK_EXT_SHADER_OCP_MICROSCALING_TYPES_EXTENSION_NAME, properties.extensionName) == 0) {
                ocp_microscaling_extension = true;
#पूर्ण_अगर
#अगर defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
            } अन्यथा अगर (म_भेद(VK_EXT_SHADER_FLOAT8_EXTENSION_NAME, properties.extensionName) == 0) {
                shader_float8_extension = true;
#पूर्ण_अगर
            } अन्यथा अगर (म_भेद("VK_VALVE_shader_mixed_float_dot_product", properties.extensionName) == 0 &&
                       !दो_पर्या("GGML_VK_DISABLE_DOT2")) {
                dot2_f16_support = true;
            } अन्यथा अगर (म_भेद("VK_KHR_pipeline_executable_properties", properties.extensionName) == 0) {
                pipeline_executable_properties_support = true;
            } अन्यथा अगर (म_भेद("VK_EXT_memory_priority", properties.extensionName) == 0 &&
                       दो_पर्या("GGML_VK_ENABLE_MEMORY_PRIORITY")) {
                device->memory_priority = true;
            } अन्यथा अगर (म_भेद("VK_EXT_external_memory_host", properties.extensionName) == 0) {
                device->external_memory_host = true;
#अगर defined(VK_EXT_shader_64bit_indexing)
            } अन्यथा अगर (म_भेद("VK_EXT_shader_64bit_indexing", properties.extensionName) == 0) {
                device->shader_64b_indexing = true;
#पूर्ण_अगर
            } अन्यथा अगर (म_भेद(VK_KHR_INTERNALLY_SYNCHRONIZED_QUEUES_EXTENSION_NAME, properties.extensionName) == 0) {
                internally_sync_support = true;
            } अन्यथा अगर (म_भेद("VK_EXT_device_fault", properties.extensionName) == 0) {
                device->device_fault = true;
            }
        }

        vk::PhysicalDeviceProperties2 props2;
        vk::PhysicalDeviceMaintenance3Properties props3;
        vk::PhysicalDeviceMaintenance4Properties props4;
        vk::PhysicalDeviceSubgroupProperties subgroup_props;
        vk::PhysicalDeviceDriverProperties driver_props;
        vk::PhysicalDeviceShaderSMBuiltinsPropertiesNV sm_props;
        vk::PhysicalDeviceShaderCoreProperties2AMD amd_shader_core_properties2_props;
        vk::PhysicalDeviceVulkan11Properties vk11_props;
        vk::PhysicalDeviceVulkan12Properties vk12_props;
        vk::Physical_Device_Subgroup_आकार_Control_Properties_EXT subgroup_आकार_control_props;
        vk::PhysicalDeviceShaderIntegerDotProductPropertiesKHR shader_integer_dot_product_props;
        vk::PhysicalDeviceExternalMemoryHostPropertiesEXT external_memory_host_props;

        props2.p_अग्र = &props3;
        props3.p_अग्र = &subgroup_props;
        subgroup_props.p_अग्र = &driver_props;
        driver_props.p_अग्र = &vk11_props;
        vk11_props.p_अग्र = &vk12_props;

        VkBaseOutStructure * last_struct = (VkBaseOutStructure *)&vk12_props;

        अगर (maintenance4_support) {
            last_struct->p_अग्र = (VkBaseOutStructure *)&props4;
            last_struct = (VkBaseOutStructure *)&props4;
        }
        अगर (sm_builtins) {
            last_struct->p_अग्र = (VkBaseOutStructure *)&sm_props;
            last_struct = (VkBaseOutStructure *)&sm_props;
        }
        अगर (amd_shader_core_properties2) {
            last_struct->p_अग्र = (VkBaseOutStructure *)&amd_shader_core_properties2_props;
            last_struct = (VkBaseOutStructure *)&amd_shader_core_properties2_props;
        }
        अगर (device->subgroup_आकार_control) {
            last_struct->p_अग्र = (VkBaseOutStructure *)&subgroup_आकार_control_props;
            last_struct = (VkBaseOutStructure *)&subgroup_आकार_control_props;
        }

#अगर defined(VK_NV_cooperative_matrix2)
        vk::PhysicalDeviceCooperativeMatrix2PropertiesNV coopmat2_props;
        अगर (coopmat2_support) {
            last_struct->p_अग्र = (VkBaseOutStructure *)&coopmat2_props;
            last_struct = (VkBaseOutStructure *)&coopmat2_props;
        }
#पूर्ण_अगर

        अगर (device->integer_dot_product) {
            last_struct->p_अग्र = (VkBaseOutStructure *)&shader_integer_dot_product_props;
            last_struct = (VkBaseOutStructure *)&shader_integer_dot_product_props;
        }

        अगर (device->external_memory_host) {
            last_struct->p_अग्र = (VkBaseOutStructure *)&external_memory_host_props;
            last_struct = (VkBaseOutStructure *)&external_memory_host_props;
        }

        device->physical_device.getProperties2(&props2);
        device->properties = props2.properties;
        device->vendor_id = device->properties.vendorID;
        device->driver_id = driver_props.driverID;

        device->external_memory_host = device->external_memory_host && ggml_vk_external_memory_host_usable(device->driver_id);

        // Implementing the async backend interfaces seems broken on older Intel HW,
        // see https://github.com/ggml-org/llama.cpp/issues/17302.
        device->support_async = (device->vendor_id != VK_VENDOR_ID_INTEL ||
                                 std::string(device->properties.deviceName.तथ्य()).find("(DG1)") == std::string::npos) &&
                                दो_पर्या("GGML_VK_DISABLE_ASYNC") == nullptr;

        अगर (!device->support_async) {
            GGML_LOG_DEBUG("ggml_vulkan: WARNING: Async execution disabled on certain Intel devices.\n");
        }

        स्थिर अक्षर* GGML_VK_FORCE_MAX_ALLOCATION_आकार = दो_पर्या("GGML_VK_FORCE_MAX_ALLOCATION_SIZE");

        अगर (GGML_VK_FORCE_MAX_ALLOCATION_आकार != nullptr) {
            device->max_memory_allocation_आकार = std::stoull(GGML_VK_FORCE_MAX_ALLOCATION_आकार);
        } अन्यथा अगर (maintenance4_support) {
            device->max_memory_allocation_आकार = std::min(props3.max_Memory_Allocation_आकार, props4.max_बफर_आकार_2);
        } अन्यथा {
            device->max_memory_allocation_आकार = props3.max_Memory_Allocation_आकार;
        }

        स्थिर अक्षर* GGML_VK_FORCE_MAX_बफर_आकार = दो_पर्या("GGML_VK_FORCE_MAX_BUFFER_SIZE");

        अगर (GGML_VK_FORCE_MAX_बफर_आकार != nullptr) {
            device->max_बफर_आकार = std::stoull(GGML_VK_FORCE_MAX_बफर_आकार);
        } अन्यथा अगर (maintenance4_support) {
            device->max_बफर_आकार = props4.max_बफर_आकार_2;
        } अन्यथा {
            device->max_बफर_आकार = device->max_memory_allocation_आकार;
        }

        स्थिर अक्षर* GGML_VK_SUBALLOCATION_BLOCK_आकार = दो_पर्या("GGML_VK_SUBALLOCATION_BLOCK_SIZE");

        अगर (GGML_VK_SUBALLOCATION_BLOCK_आकार != nullptr) {
            device->suballocation_block_आकार = std::stoull(GGML_VK_SUBALLOCATION_BLOCK_आकार);
        } अन्यथा {
            // Limit batching of allocations to 1GB by default to avoid fragmentation issues
            device->suballocation_block_आकार = 1024*1024*1024;
        }
        device->suballocation_block_आकार = std::min(device->suballocation_block_आकार, device->max_memory_allocation_आकार);

        device->subgroup_आकार = subgroup_props.subgroup_आकार_2;
        device->subgroup_आकार_log2 = uint32_t(log2f(भग्न(device->subgroup_आकार)));
        device->uma = device->properties.deviceType == vk::PhysicalDeviceType::eIntegratedGpu;
        अगर (sm_builtins) {
            device->shader_core_गिनती = sm_props.shaderSMCount;
        } अन्यथा अगर (amd_shader_core_properties2) {
            device->shader_core_गिनती = amd_shader_core_properties2_props.active_Compute_Unit_गिनती;
        } अन्यथा अगर (device->vendor_id == VK_VENDOR_ID_INTEL) {
            device->shader_core_गिनती = ggml_vk_intel_shader_core_गिनती(device->physical_device);
        } अन्यथा {
            device->shader_core_गिनती = 0;
        }
        device->float_controls_rte_fp16 = vk12_props.shaderRoundingModeRTEFloat16;
        device->float_controls_denorm_preserve_fp16 = vk12_props.shaderDenormPreserveFloat16;

        device->subgroup_basic = (vk11_props.subgroupSupportedStages & vk::ShaderStageFlagBits::eCompute) &&
                                 (vk11_props.subgroupSupportedOperations & vk::SubgroupFeatureFlagBits::eBasic);
        device->subgroup_arithmetic = (vk11_props.subgroupSupportedStages & vk::ShaderStageFlagBits::eCompute) &&
                                      (vk11_props.subgroupSupportedOperations & vk::SubgroupFeatureFlagBits::eArithmetic);
#अगर_घोषित __APPLE__
        // Workaround for subgroup arithmetic failing on MoltenVK with AMD GPUs (issue 15846)
        अगर (device->vendor_id == VK_VENDOR_ID_AMD) {
            device->subgroup_arithmetic = false;
        }
#पूर्ण_अगर
        device->subgroup_shuffle = (vk11_props.subgroupSupportedStages & vk::ShaderStageFlagBits::eCompute) &&
                                   (vk11_props.subgroupSupportedOperations & vk::SubgroupFeatureFlagBits::eShuffle);
#अगर_घोषित __APPLE__
        अगर (device->vendor_id == VK_VENDOR_ID_AMD) {
            device->subgroup_shuffle = false;
        }
#पूर्ण_अगर
        device->subgroup_clustered = (vk11_props.subgroupSupportedStages & vk::ShaderStageFlagBits::eCompute) &&
                                     (vk11_props.subgroupSupportedOperations & vk::SubgroupFeatureFlagBits::eClustered);

        device->subgroup_ballot = (vk11_props.subgroupSupportedStages & vk::ShaderStageFlagBits::eCompute) &&
                                  (vk11_props.subgroupSupportedOperations & vk::SubgroupFeatureFlagBits::eBallot);

        device->subgroup_vote = (vk11_props.subgroupSupportedStages & vk::ShaderStageFlagBits::eCompute) &&
                                (vk11_props.subgroupSupportedOperations & vk::SubgroupFeatureFlagBits::eVote);

        // Submit at least every 100 nodes, in case there are workloads without as much matmul.
        device->max_nodes_per_submit = 100;
        स्थिर अक्षर* GGML_VK_MAX_NODES_PER_SUBMIT = दो_पर्या("GGML_VK_MAX_NODES_PER_SUBMIT");
        अगर (GGML_VK_MAX_NODES_PER_SUBMIT != nullptr) {
            uint32_t max_nodes_per_submit = std::stoul(GGML_VK_MAX_NODES_PER_SUBMIT);
            device->max_nodes_per_submit = std::max(max_nodes_per_submit, 1u);
        }

        स्थिर bool force_disable_f16 = दो_पर्या("GGML_VK_DISABLE_F16") != nullptr;

        device->fp16 = !force_disable_f16 && fp16_storage && fp16_compute;

        अगर (!ggml_vk_khr_cooperative_matrix_support(device->properties, driver_props, device->architecture)) {
            device->coopmat_support = false;
        }

        device->integer_dot_product = device->integer_dot_product && shader_integer_dot_product_props.integerDotProduct4x8BitPackedSignedAccelerated;

        device->min_imported_host_pointer_alignment = external_memory_host_props.minImportedHostPointerAlignment;

        device->max_workgroup_आकार_log2 = uint32_t(log2f(भग्न(device->properties.limits.maxComputeWorkGroupInvocations)));

        std::vector<vk::QueueFamilyProperties> queue_family_props = device->physical_device.getQueueFamilyProperties();

        // Try to find a non-graphics compute queue and transfer-focused queues
        // Allow overriding avoiding the graphics queue because it can increase performance on RADV
        स्थिर bool allow_graphics_queue = (दो_पर्या("GGML_VK_ALLOW_GRAPHICS_QUEUE") != nullptr);
        स्थिर vk::QueueFlagBits graphics_flag = allow_graphics_queue ? (vk::QueueFlagBits)0 : vk::QueueFlagBits::eGraphics;
        स्थिर uint32_t compute_queue_family_सूचकांक = ggml_vk_find_queue_family_सूचकांक(queue_family_props, vk::QueueFlagBits::eCompute, graphics_flag, -1, 1);
        स्थिर uint32_t transfer_queue_family_सूचकांक = ggml_vk_find_queue_family_सूचकांक(queue_family_props, vk::QueueFlagBits::eTransfer, vk::QueueFlagBits::eCompute | graphics_flag, compute_queue_family_सूचकांक, 1);

        स्थिर भग्न priorities[] = { 1.0f, 1.0f };
        device->single_queue = compute_queue_family_सूचकांक == transfer_queue_family_सूचकांक && queue_family_props[compute_queue_family_सूचकांक].queue_गिनती == 1;

        std::vector<vk::DeviceQueueCreateInfo> device_queue_create_infos;
        vk::DeviceCreateInfo device_create_info{};
        std::vector<स्थिर अक्षर *> device_extensions;
        vk::PhysicalDeviceFeatures device_features = device->physical_device.getFeatures();

        VkPhysicalDeviceFeatures2 device_features2;
        device_features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        device_features2.p_अग्र = nullptr;
        device_features2.features = (VkPhysicalDeviceFeatures)device_features;

        VkPhysicalDeviceVulkan11Features vk11_features;
        vk11_features.p_अग्र = nullptr;
        vk11_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
        device_features2.p_अग्र = &vk11_features;

        VkPhysicalDeviceVulkan12Features vk12_features;
        vk12_features.p_अग्र = nullptr;
        vk12_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        vk11_features.p_अग्र = &vk12_features;

        last_struct = (VkBaseOutStructure *)&vk12_features;

        VkPhysicalDeviceInternallySynchronizedQueuesFeaturesKHR internally_synchronized_queues_features{};
        internally_synchronized_queues_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INTERNALLY_SYNCHRONIZED_QUEUES_FEATURES_KHR;
        internally_synchronized_queues_features.p_अग्र = nullptr;
        internally_synchronized_queues_features.internallySynchronizedQueues = VK_FALSE;

        अगर (internally_sync_support) {
            last_struct->p_अग्र = (VkBaseOutStructure *)&internally_synchronized_queues_features;
            last_struct = (VkBaseOutStructure *)&internally_synchronized_queues_features;
            device_extensions.push_back(VK_KHR_INTERNALLY_SYNCHRONIZED_QUEUES_EXTENSION_NAME);
        }

        VkPhysicalDevicePipelineRobustnessFeaturesEXT pl_robustness_features;
        pl_robustness_features.p_अग्र = nullptr;
        pl_robustness_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_ROBUSTNESS_FEATURES_EXT;
        pl_robustness_features.pipelineRobustness = VK_FALSE;

        अगर (pipeline_robustness) {
            last_struct->p_अग्र = (VkBaseOutStructure *)&pl_robustness_features;
            last_struct = (VkBaseOutStructure *)&pl_robustness_features;
            device_extensions.push_back("VK_EXT_pipeline_robustness");
        }

        VkPhysicalDeviceMemoryPriorityFeaturesEXT memory_priority_features;
        memory_priority_features.p_अग्र = nullptr;
        memory_priority_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PRIORITY_FEATURES_EXT;
        memory_priority_features.memoryPriority = VK_FALSE;
        अगर (device->memory_priority) {
            last_struct->p_अग्र = (VkBaseOutStructure *)&memory_priority_features;
            last_struct = (VkBaseOutStructure *)&memory_priority_features;
            device_extensions.push_back("VK_EXT_memory_priority");
        }

        Vk_Physical_Device_Subgroup_आकार_Control_Features_EXT subgroup_आकार_control_features;
        subgroup_आकार_control_features.p_अग्र = nullptr;
        subgroup_आकार_control_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_आकार_CONTROL_FEATURES_EXT;
        subgroup_आकार_control_features.computeFullSubgroups = false;
        subgroup_आकार_control_features.subgroup_आकार_Control = false;

        अगर (device->subgroup_आकार_control) {
            last_struct->p_अग्र = (VkBaseOutStructure *)&subgroup_आकार_control_features;
            last_struct = (VkBaseOutStructure *)&subgroup_आकार_control_features;
        }

#अगर defined(VK_KHR_cooperative_matrix)
        VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopmat_features;
        coopmat_features.p_अग्र = nullptr;
        coopmat_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
        coopmat_features.cooperativeMatrix = VK_FALSE;

        अगर (device->coopmat_support) {
            last_struct->p_अग्र = (VkBaseOutStructure *)&coopmat_features;
            last_struct = (VkBaseOutStructure *)&coopmat_features;
        }
#पूर्ण_अगर

#अगर defined(VK_NV_cooperative_matrix2)
        VkPhysicalDeviceCooperativeMatrix2FeaturesNV coopmat2_features {};
        coopmat2_features.p_अग्र = nullptr;
        coopmat2_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_FEATURES_NV;
        अगर (coopmat2_support) {
            last_struct->p_अग्र = (VkBaseOutStructure *)&coopmat2_features;
            last_struct = (VkBaseOutStructure *)&coopmat2_features;
            device_extensions.push_back("VK_NV_cooperative_matrix2");
        }
#पूर्ण_अगर

        VkPhysicalDeviceCooperativeMatrixDecodeVectorFeaturesNV coopmat2_decode_vector_features {};
        coopmat2_decode_vector_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_DECODE_VECTOR_FEATURES_NV;
        अगर (coopmat2_decode_vector_support) {
            last_struct->p_अग्र = (VkBaseOutStructure *)&coopmat2_decode_vector_features;
            last_struct = (VkBaseOutStructure *)&coopmat2_decode_vector_features;
            device_extensions.push_back(VK_NV_COOPERATIVE_MATRIX_DECODE_VECTOR_EXTENSION_NAME);
        }

#अगर defined(VK_KHR_shader_bfloat16)
        VkPhysicalDeviceShaderBfloat16FeaturesKHR bfloat16_features {};
        bfloat16_features.p_अग्र = nullptr;
        bfloat16_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR;
        अगर (bfloat16_support) {
            last_struct->p_अग्र = (VkBaseOutStructure *)&bfloat16_features;
            last_struct = (VkBaseOutStructure *)&bfloat16_features;
            device_extensions.push_back("VK_KHR_shader_bfloat16");
        }
#पूर्ण_अगर

        VkPhysicalDeviceShaderOCPMicroscalingTypesFeaturesEXT ocp_microscaling_features {};
        ocp_microscaling_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OCP_MICROSCALING_TYPES_FEATURES_EXT;
        अगर (ocp_microscaling_extension) {
            last_struct->p_अग्र = (VkBaseOutStructure *)&ocp_microscaling_features;
            last_struct = (VkBaseOutStructure *)&ocp_microscaling_features;
            device_extensions.push_back(VK_EXT_SHADER_OCP_MICROSCALING_TYPES_EXTENSION_NAME);
        }

        VkPhysicalDeviceShaderFloat8FeaturesEXT shader_float8_features {};
        shader_float8_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT;
        अगर (shader_float8_extension) {
            last_struct->p_अग्र = (VkBaseOutStructure *)&shader_float8_features;
            last_struct = (VkBaseOutStructure *)&shader_float8_features;
            device_extensions.push_back(VK_EXT_SHADER_FLOAT8_EXTENSION_NAME);
        }

        VkPhysicalDeviceMaintenance4Features maint4_features {};
        maint4_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES;
        अगर (maintenance4_support) {
            last_struct->p_अग्र = (VkBaseOutStructure *)&maint4_features;
            last_struct = (VkBaseOutStructure *)&maint4_features;
            device_extensions.push_back("VK_KHR_maintenance4");
        }

        VkPhysicalDeviceShaderIntegerDotProductFeaturesKHR shader_integer_dot_product_features {};
        shader_integer_dot_product_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES_KHR;
        अगर (device->integer_dot_product) {
            last_struct->p_अग्र = (VkBaseOutStructure *)&shader_integer_dot_product_features;
            last_struct = (VkBaseOutStructure *)&shader_integer_dot_product_features;
            device_extensions.push_back("VK_KHR_shader_integer_dot_product");
        }

        VkPhysicalDeviceShaderMixedFloatDotProductFeaturesVALVE dot2_features {};
        dot2_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_MIXED_FLOAT_DOT_PRODUCT_FEATURES_VALVE;
        अगर (dot2_f16_support) {
            last_struct->p_अग्र = (VkBaseOutStructure *)&dot2_features;
            last_struct = (VkBaseOutStructure *)&dot2_features;
            device_extensions.push_back("VK_VALVE_shader_mixed_float_dot_product");
        }

        VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR pep_features {};
        pep_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR;
        अगर (pipeline_executable_properties_support) {
            last_struct->p_अग्र = (VkBaseOutStructure *)&pep_features;
            last_struct = (VkBaseOutStructure *)&pep_features;
            device_extensions.push_back("VK_KHR_pipeline_executable_properties");
        }

        अगर (device->external_memory_host) {
            device_extensions.push_back("VK_EXT_external_memory_host");
        }

#अगर defined(VK_EXT_shader_64bit_indexing)
        VkPhysicalDeviceShader64BitIndexingFeaturesEXT shader_64bit_indexing_features {};
        shader_64bit_indexing_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_64_BIT_INDEXING_FEATURES_EXT;
        अगर (device->shader_64b_indexing) {
            last_struct->p_अग्र = (VkBaseOutStructure *)&shader_64bit_indexing_features;
            last_struct = (VkBaseOutStructure *)&shader_64bit_indexing_features;
            device_extensions.push_back("VK_EXT_shader_64bit_indexing");
        }
#पूर्ण_अगर

        VkPhysicalDeviceFaultFeaturesEXT fault_features {};
        fault_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT;
        अगर (device->device_fault) {
            last_struct->p_अग्र = (VkBaseOutStructure *)&fault_features;
            last_struct = (VkBaseOutStructure *)&fault_features;
            device_extensions.push_back("VK_EXT_device_fault");
        }

        vkGetPhysicalDeviceFeatures2(device->physical_device, &device_features2);

        device->device_fault = device->device_fault && fault_features.deviceFault;

        device->has_internally_synchronized_queues = internally_synchronized_queues_features.internallySynchronizedQueues;

        // Build queue create infos only after querying whether internally synchronized queues are enabled.
        // getQueue2() later uses the same flag, so creation/retrieval must stay consistent.
        vk::DeviceQueueCreateFlags queue_flags = device->has_internally_synchronized_queues ?
                                                eInternallySynchronizedKHR :
                                                vk::DeviceQueueCreateFlags();

        अगर (compute_queue_family_सूचकांक != transfer_queue_family_सूचकांक) {
            device_queue_create_infos.push_back({queue_flags, compute_queue_family_सूचकांक, 1, priorities});
            device_queue_create_infos.push_back({queue_flags, transfer_queue_family_सूचकांक, 1, priorities + 1});
        } अन्यथा अगर(!device->single_queue) {
            device_queue_create_infos.push_back({queue_flags, compute_queue_family_सूचकांक, 2, priorities});
        } अन्यथा {
            device_queue_create_infos.push_back({queue_flags, compute_queue_family_सूचकांक, 1, priorities});
        }

        device->pipeline_executable_properties_support = pipeline_executable_properties_support;

        device->fp16 = device->fp16 && vk12_features.shaderFloat16;

#अगर defined(VK_KHR_shader_bfloat16)
        device->bf16 = bfloat16_support && bfloat16_features.shaderBFloat16Type;
#अन्यथा
        device->bf16 = false;
#पूर्ण_अगर

        device->dot2_f16 = dot2_f16_support && dot2_features.shaderMixedFloatDotProductFloat16AccFloat32;
        device->ocp_fp4 = ocp_microscaling_extension && ocp_microscaling_features.shaderFloat4 &&
                          shader_float8_extension && shader_float8_features.shaderFloat8 &&
                          !दो_पर्या("GGML_VK_DISABLE_OCP_FP4");

        device->pipeline_robustness = pl_robustness_features.pipelineRobustness;

        device->multi_add = vk12_props.shaderRoundingModeRTEFloat16 &&
                            device->properties.limits.max_धकेलो_Constants_आकार >= माप(vk_op_multi_add_धकेलो_constants) &&
                            दो_पर्या("GGML_VK_DISABLE_MULTI_ADD") == nullptr;

        device->shader_int64 = device_features2.features.shaderInt64;
        device->बफर_device_address = vk12_features.बफर_Device_Address;
        device->vulkan_memory_model = vk12_features.vulkanMemoryModel;

        अगर (device->subgroup_आकार_control) {
            device->subgroup_min_आकार = subgroup_आकार_control_props.min_Subgroup_आकार;
            device->subgroup_max_आकार = subgroup_आकार_control_props.max_Subgroup_आकार;
            device_extensions.push_back("VK_EXT_subgroup_size_control");
        }

        device->subgroup_आकार_control = device->subgroup_आकार_control &&
                (subgroup_आकार_control_props.required_Subgroup_आकार_Stages & vk::ShaderStageFlagBits::eCompute) &&
                subgroup_आकार_control_features.subgroup_आकार_Control;

        device->subgroup_require_full_support = subgroup_आकार_control_features.computeFullSubgroups;

#अगर defined(VK_KHR_cooperative_matrix)
        device->coopmat_support = device->coopmat_support && coopmat_features.cooperativeMatrix;
        device->coopmat1_fa_support = device->coopmat_support && device->subgroup_require_full_support;
#पूर्ण_अगर

        अगर (coopmat2_support) {
#अगर defined(VK_NV_cooperative_matrix2) && defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
            अगर (coopmat2_features.cooperativeMatrixWorkgroupScope &&
                coopmat2_features.cooperativeMatrixFlexibleDimensions &&
                coopmat2_features.cooperativeMatrixReductions &&
                coopmat2_features.cooperativeMatrixConversions &&
                coopmat2_features.cooperativeMatrixPerElementOperations &&
                coopmat2_features.cooperativeMatrixTensorAddressing &&
                coopmat2_features.cooperativeMatrixBlockLoads &&
                vk12_features.बफर_Device_Address) {

                std::vector<VkCooperativeMatrixFlexibleDimensionsPropertiesNV> flexible_dimensions;
                uint32_t count = 0;

                PFN_vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV
                    _vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV =
                        (PFN_vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV)
                        vk_instance.instance.getProcAddr("vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV");

                _vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV(device->physical_device, &count, nullptr);

                VkCooperativeMatrixFlexibleDimensionsPropertiesNV empty_prop {};
                empty_prop.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_FLEXIBLE_DIMENSIONS_PROPERTIES_NV;
                flexible_dimensions.resize(count, empty_prop);

                _vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV(device->physical_device, &count, flexible_dimensions.तथ्य());

                bool found_fp16_128 = false,
                     found_fp16_256 = false,
                     found_fp32_128 = false,
                     found_fp32_256 = false;
                bool found_bf16_128 = false,
                     found_bf16_256 = false;
                // need to support fp16*fp16 with fp16/fp32 accumulator, for workgroupsize 128
                // with 32x16x16 and 256 with 32x32x16.
                क्रम (स्वतः &prop : flexible_dimensions) {
                    अगर (prop.saturatingAccumulation == VK_FALSE &&
                        prop.scope == VK_SCOPE_WORKGROUP_KHR) {

                        अगर (prop.AType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                            prop.BType == VK_COMPONENT_TYPE_FLOAT16_KHR) {

                            अगर (prop.workgroupInvocations == 128 &&
                                prop.MGranularity <= 32 &&
                                prop.NGranularity <= 16 &&
                                prop.KGranularity <= 16) {
                                अगर (prop.CType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                                    prop.परिणाम_Type == VK_COMPONENT_TYPE_FLOAT16_KHR) {
                                    found_fp16_128 = true;
                                }
                                अगर (prop.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                                    prop.परिणाम_Type == VK_COMPONENT_TYPE_FLOAT32_KHR) {
                                    found_fp32_128 = true;
                                }
                            }
                            अगर (prop.workgroupInvocations == 256 &&
                                prop.MGranularity <= 32 &&
                                prop.NGranularity <= 32 &&
                                prop.KGranularity <= 16) {
                                अगर (prop.CType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                                    prop.परिणाम_Type == VK_COMPONENT_TYPE_FLOAT16_KHR) {
                                    found_fp16_256 = true;
                                }
                                अगर (prop.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                                    prop.परिणाम_Type == VK_COMPONENT_TYPE_FLOAT32_KHR) {
                                    found_fp32_256 = true;
                                }
                            }
                        }

#अगर defined(VK_KHR_shader_bfloat16) && defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
                        अगर (bfloat16_support &&
                            prop.AType == VK_COMPONENT_TYPE_BFLOAT16_KHR &&
                            prop.BType == VK_COMPONENT_TYPE_BFLOAT16_KHR &&
                            prop.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                            prop.परिणाम_Type == VK_COMPONENT_TYPE_FLOAT32_KHR) {

                            अगर (prop.workgroupInvocations == 128 &&
                                prop.MGranularity <= 32 &&
                                prop.NGranularity <= 16 &&
                                prop.KGranularity <= 16) {
                                found_bf16_128 = true;
                            }
                            अगर (prop.workgroupInvocations == 256 &&
                                prop.MGranularity <= 32 &&
                                prop.NGranularity <= 32 &&
                                prop.KGranularity <= 16) {
                                found_bf16_256 = true;
                            }
                        }
#पूर्ण_अगर
                    }
                }
                अगर (found_fp16_128 && found_fp16_256 &&
                    found_fp32_128 && found_fp32_256 &&
                    coopmat2_props.cooperativeMatrixFlexibleDimensionsMaxDimension >= 512) {
                    device->coopmat2 = true;
                    device->coopmat2_bf16_support = found_bf16_128 && found_bf16_256;
                    device->coopmat2_decode_vector = coopmat2_decode_vector_support && coopmat2_decode_vector_features.cooperativeMatrixDecodeVector;
                }
            }
#पूर्ण_अगर
        }

        अगर (!vk11_features.storageBuffer16BitAccess) {
            std::अ_त्रुटि << "ggml_vulkan: device " << GGML_VK_NAME << idx << " does not support 16-bit storage." << std::प_पूर्ण;
            फेंको std::runtime_error("Unsupported device");
        }

        device_extensions.push_back("VK_KHR_16bit_storage");

#अगर_घोषित GGML_VULKAN_VALIDATE
        device_extensions.push_back("VK_KHR_shader_non_semantic_info");
#पूर्ण_अगर

        अगर (device->fp16) {
            device_extensions.push_back("VK_KHR_shader_float16_int8");
        }

#अगर defined(VK_KHR_cooperative_matrix)
        अगर (device->coopmat_support) {
            // Query supported shapes
            std::vector<VkCooperativeMatrixPropertiesKHR> cm_props;

            PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR pfn_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR =
                (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)vkGetInstanceProcAddr(vk_instance.instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");

            uint32_t cm_props_अंक;

            pfn_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(device->physical_device, &cm_props_अंक, nullptr);

            cm_props.resize(cm_props_अंक);

            क्रम (स्वतः& prop : cm_props) {
                prop.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
            }

            pfn_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(device->physical_device, &cm_props_अंक, cm_props.तथ्य());

            VK_LOG_DEBUG("ggml_vulkan: Cooperative Matrix Shapes: " << cm_props.size());

            क्रम (स्वतः& prop : cm_props) {
                VK_LOG_DEBUG("ggml_vulkan: M: " << prop.MSize << " N: " << prop.NSize << " K: " << prop.KSize << " A: " << vk::to_string((vk::ComponentTypeKHR)prop.AType) << " B: " << vk::to_string((vk::ComponentTypeKHR)prop.BType) << " C: " << vk::to_string((vk::ComponentTypeKHR)prop.CType) << " Result: " << vk::to_string((vk::ComponentTypeKHR)prop.परिणाम_Type) << " saturatingAccumulation: " << prop.saturatingAccumulation << " scope: " << vk::to_string((vk::ScopeKHR)prop.scope));

                अगर ((vk::ComponentTypeKHR)prop.AType == vk::ComponentTypeKHR::eFloat16 &&
                    (vk::ComponentTypeKHR)prop.BType == vk::ComponentTypeKHR::eFloat16 &&
                    (vk::ScopeKHR)prop.scope == vk::ScopeKHR::eSubgroup
                ) {
                    अगर ((vk::ComponentTypeKHR)prop.CType == vk::ComponentTypeKHR::eFloat32 &&
                        (vk::ComponentTypeKHR)prop.परिणाम_Type == vk::ComponentTypeKHR::eFloat32) {
                        // coopmat sizes not set yet
                        अगर (device->coopmat_m == 0) {
                            device->coopmat_acc_f32_support = true;
                            device->coopmat_m = prop.MSize;
                            device->coopmat_n = prop.NSize;
                            device->coopmat_k = prop.KSize;
                        } अन्यथा अगर (device->coopmat_m == prop.MSize && device->coopmat_n == prop.NSize && device->coopmat_k == prop.KSize) {
                            // Only enable if shape is identical
                            device->coopmat_acc_f32_support = true;
                        }
                        अगर (prop.MSize == 16 && prop.NSize == 16 && prop.KSize == 16) {
                            device->coopmat_support_16x16x16_f32acc = true;
                        }
                    } अन्यथा अगर ((vk::ComponentTypeKHR)prop.CType == vk::ComponentTypeKHR::eFloat16 &&
                               (vk::ComponentTypeKHR)prop.परिणाम_Type == vk::ComponentTypeKHR::eFloat16) {
                        // coopmat sizes not set yet
                        अगर (device->coopmat_m == 0) {
                            device->coopmat_acc_f16_support = true;
                            device->coopmat_m = prop.MSize;
                            device->coopmat_n = prop.NSize;
                            device->coopmat_k = prop.KSize;
                        } अन्यथा अगर (device->coopmat_m == prop.MSize && device->coopmat_n == prop.NSize && device->coopmat_k == prop.KSize) {
                            // Only enable if shape is identical
                            device->coopmat_acc_f16_support = true;
                        }
                        अगर (prop.MSize == 16 && prop.NSize == 16 && prop.KSize == 16) {
                            device->coopmat_support_16x16x16_f16acc = true;
                        }
                    }
                } अन्यथा अगर ((vk::ComponentTypeKHR)prop.AType      == vk::ComponentTypeKHR::eSint8 &&
                           (vk::ComponentTypeKHR)prop.BType      == vk::ComponentTypeKHR::eSint8 &&
                           (vk::ComponentTypeKHR)prop.CType      == vk::ComponentTypeKHR::eSint32 &&
                           (vk::ComponentTypeKHR)prop.परिणाम_Type == vk::ComponentTypeKHR::eSint32 &&
                           (vk::ScopeKHR)prop.scope == vk::ScopeKHR::eSubgroup &&
                           device->coopmat_int_m == 0
                ) {
                    device->coopmat_int_support = true;
                    device->coopmat_int_m = prop.MSize;
                    device->coopmat_int_n = prop.NSize;
                    device->coopmat_int_k = prop.KSize;
                }
#अगर defined(VK_KHR_shader_bfloat16) && defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
                अगर (bfloat16_support &&
                    prop.AType == VK_COMPONENT_TYPE_BFLOAT16_KHR &&
                    prop.BType == VK_COMPONENT_TYPE_BFLOAT16_KHR &&
                    prop.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                    prop.परिणाम_Type == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                    (vk::ScopeKHR)prop.scope == vk::ScopeKHR::eSubgroup
                ) {
                    // coopmat sizes not set yet
                    अगर (device->coopmat_m == 0) {
                        device->coopmat_bf16_support = true;
                        device->coopmat_m = prop.MSize;
                        device->coopmat_n = prop.NSize;
                        device->coopmat_k = prop.KSize;
                    } अन्यथा अगर (device->coopmat_m == prop.MSize && device->coopmat_n == prop.NSize && device->coopmat_k == prop.KSize) {
                        // Only enable if shape is identical
                        device->coopmat_bf16_support = true;
                    }
                }
#पूर्ण_अगर
            }

            अगर (device->coopmat_m == 0 || !device->coopmat_acc_f32_support) {
                // No suitable matmul mode found
                GGML_LOG_DEBUG("ggml_vulkan: WARNING: No suitable matrix core mode found. Disabling matrix cores.\n");
                device->coopmat_support = false;
            }
        }

        अगर (device->coopmat_support) {
            device_extensions.push_back("VK_KHR_cooperative_matrix");
        }
#पूर्ण_अगर
        device->name = GGML_VK_NAME + std::to_string(idx);

        device_create_info
            .setFlags(vk::DeviceCreateFlags())
            .setQueueCreateInfos(device_queue_create_infos)
            .setPEnabledExtensionNames(device_extensions);
        device_create_info.setPNext(&device_features2);
        device->device = device->physical_device.createDevice(device_create_info);

        अगर (device->device_fault) {
            device->pfn_vkGetDeviceFaultInfoEXT = (PFN_vkGetDeviceFaultInfoEXT)
                vkGetDeviceProcAddr(device->device, "vkGetDeviceFaultInfoEXT");
        }

        // Queues
        device->compute_queue = ggml_vk_create_queue(device, compute_queue_family_सूचकांक, 0, { vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eTransfer }, false);

        // Shaders
        // Disable matmul tile sizes early if performance low or not supported
        क्रम (uint32_t i = 0; i < GGML_TYPE_गिनती; ++i) {
            चयन (device->vendor_id) {
#अगर_अघोषित GGML_VULKAN_RUN_TESTS
            हाल VK_VENDOR_ID_AMD:
                device->mul_mat_l[i]    = device->coopmat_support && device->driver_id != vk::DriverId::eAmdProprietary;
                device->mul_mat_m[i]    = true;
                device->mul_mat_s[i]    = true;
                device->mul_mat_id_l[i] = false;
                device->mul_mat_id_m[i] = true;
                device->mul_mat_id_s[i] = true;
                अवरोध;
            हाल VK_VENDOR_ID_INTEL: {
                // Current Windows driver does not expose BF16 support.
                // We only want to use l_warptile if coopmat is available
                स्थिर bool use_l_warptile = (i == GGML_TYPE_BF16) ? (device->coopmat_bf16_support && device->coopmat_support) : device->coopmat_support;
                device->mul_mat_l[i] = use_l_warptile;
                device->mul_mat_id_l[i] = use_l_warptile;
                device->mul_mat_m[i] = true;
                device->mul_mat_s[i] = true;
                device->mul_mat_id_m[i] = true;
                device->mul_mat_id_s[i] = true;
                अवरोध;
            }
            हाल VK_VENDOR_ID_APPLE:
                device->mul_mat_l[i] = false;
                device->mul_mat_m[i] = true;
                device->mul_mat_s[i] = false;
                device->mul_mat_id_l[i] = false;
                device->mul_mat_id_m[i] = true;
                device->mul_mat_id_s[i] = false;
                अवरोध;
            हाल VK_VENDOR_ID_QUALCOMM:
                device->mul_mat_l[i] = false;
                device->mul_mat_m[i] = true;
                device->mul_mat_s[i] = !device->coopmat_support;
                device->mul_mat_id_l[i] = false;
                device->mul_mat_id_m[i] = true;
                device->mul_mat_id_s[i] = !device->coopmat_support;
                अवरोध;
            हाल VK_VENDOR_ID_SAMSUNG:
                device->mul_mat_l[i] = false;
                device->mul_mat_m[i] = true;
                device->mul_mat_s[i] = true;
                device->mul_mat_id_l[i] = false;
                device->mul_mat_id_m[i] = true;
                device->mul_mat_id_s[i] = true;
                अवरोध;
#पूर्ण_अगर
            शेष:
                device->mul_mat_l[i] = true;
                device->mul_mat_m[i] = true;
                device->mul_mat_s[i] = true;
                device->mul_mat_id_l[i] = true;
                device->mul_mat_id_m[i] = true;
                device->mul_mat_id_s[i] = true;
                अवरोध;
            }

#अगर VK_HEADER_VERSION >= 287
            // Honeykrisp driver for Asahi Linux doesn't report VK_VENDOR_ID_APPLE.
            // Check for Honeykrisp driver and force same configuration as the VK_VENDOR_ID_APPLE case.
            अगर (device->driver_id == vk::DriverId::eMesaHoneykrisp) {
                device->mul_mat_l[i] = false;
                device->mul_mat_m[i] = true;
                device->mul_mat_s[i] = false;
                device->mul_mat_id_l[i] = false;
                device->mul_mat_id_m[i] = true;
                device->mul_mat_id_s[i] = false;
            }
#पूर्ण_अगर

            device->mul_mat_l_int[i]    = device->mul_mat_l[i];
            device->mul_mat_m_int[i]    = device->mul_mat_m[i];
            device->mul_mat_s_int[i]    = device->mul_mat_s[i];
            device->mul_mat_id_l_int[i] = device->mul_mat_id_l[i];
            device->mul_mat_id_m_int[i] = device->mul_mat_id_m[i];
            device->mul_mat_id_s_int[i] = device->mul_mat_id_s[i];
        }


        std::vector<vk::DescriptorSetLayoutBinding> dsl_binding;
        std::vector<vk::DescriptorBindingFlags> dsl_binding_flags;
        क्रम (uint32_t i = 0; i < MAX_PARAMETER_गिनती; i++) {
            dsl_binding.push_back({i, vk::DescriptorType::e_Storage_बफर, 1, vk::ShaderStageFlagBits::eCompute});
            dsl_binding_flags.push_back({});
        }

        vk::DescriptorSetLayoutBindingFlagsCreateInfo dslbfci = { dsl_binding_flags };

        vk::DescriptorSetLayoutCreateInfo descriptor_set_layout_create_info(
            {},
            dsl_binding);
        descriptor_set_layout_create_info.setPNext(&dslbfci);
        device->dsl = device->device.createDescriptorSetLayout(descriptor_set_layout_create_info);

        ggml_vk_load_shaders(device);

        // Prefer a dedicated transfer queue on AMD dGPUs (non-GCN) when graphics queue use is disabled.
        स्थिर bool prefers_transfer_queue =
            device->vendor_id == VK_VENDOR_ID_AMD &&
            device->architecture != AMD_GCN &&
            !device->uma &&
            !allow_graphics_queue;

        अगर (!device->single_queue) {
            स्थिर uint32_t transfer_queue_सूचकांक = compute_queue_family_सूचकांक == transfer_queue_family_सूचकांक ? 1 : 0;
            device->transfer_queue = ggml_vk_create_queue(device, transfer_queue_family_सूचकांक, transfer_queue_सूचकांक, { vk::PipelineStageFlagBits::eTransfer }, true);

            device->async_use_transfer_queue = prefers_transfer_queue || (दो_पर्या("GGML_VK_ASYNC_USE_TRANSFER_QUEUE") != nullptr);
        } अन्यथा {
            device->transfer_queue = ggml_vk_create_aliased_queue(device, device->compute_queue);

            device->async_use_transfer_queue = false;
        }

        device->fence = device->device.createFence({});

        device->बफर_type = {
            /* .iface    = */ ggml_backend_vk_बफर_type_interface,
            /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_vk_reg(), idx),
            /* .context  = */ नया ggml_backend_vk_बफर_type_context{ device->name, device },
        };

        device->idx = idx;

        device->serialize_submissions = दो_पर्या("GGML_VK_SERIALIZE_SUBMISSIONS") != nullptr;

        device->disable_fusion = दो_पर्या("GGML_VK_DISABLE_FUSION") != nullptr;

        device->disable_descriptor_reuse = दो_पर्या("GGML_VK_DISABLE_DESCRIPTOR_REUSE") != nullptr;

        device->add_rms_fusion = !device->disable_fusion &&
                                 device->subgroup_arithmetic &&
                                 device->vendor_id != VK_VENDOR_ID_INTEL;
        device->partials_binding_alignment =
            std::max(4u, (uint32_t)device->properties.limits.min_Storage_बफर_Offset_Alignment);

        device->mmvq_mode = 0;
        अगर (दो_पर्या("GGML_VK_DISABLE_MMVQ")) {
            device->mmvq_mode = -1;
        } अन्यथा अगर (दो_पर्या("GGML_VK_FORCE_MMVQ")) {
            device->mmvq_mode = 1;
        }

        // publish the device only when it is fully built
        vk_instance.devices[idx] = device;

        वापस device;
    }

    वापस vk_instance.devices[idx];
}

अटल व्योम ggml_vk_print_gpu_info(माप_प्रकार idx) {
    GGML_ASSERT(idx < vk_instance.device_indices.size());
    माप_प्रकार dev_अंक = vk_instance.device_indices[idx];
    VK_LOG_DEBUG("ggml_vk_print_gpu_info(" << dev_अंक << ")");
    GGML_ASSERT(vk_instance_initialized);

    std::vector<vk::PhysicalDevice> devices = vk_instance.instance.enumeratePhysicalDevices();

    अगर (dev_अंक >= devices.size()) {
        std::अ_त्रुटि << "ggml_vulkan: Device with index " << dev_अंक << " does not exist." << std::प_पूर्ण;
        फेंको std::runtime_error("Device not found");
    }

    vk::PhysicalDevice physical_device = devices[dev_अंक];
    std::vector<vk::ExtensionProperties> ext_props = physical_device.enumerateDeviceExtensionProperties();

    bool fp16_storage = false;
    bool fp16_compute = false;
    bool coopmat_support = false;
    bool coopmat2_support = false;
    bool coopmat2_decode_vector_support = false;
    bool integer_dot_product = false;
    bool bfloat16_support = false;
    bool dot2_f16_support = false;
    bool ocp_microscaling_extension = false;
    bool shader_float8_extension = false;

    क्रम (स्वतः properties : ext_props) {
        अगर (म_भेद("VK_KHR_16bit_storage", properties.extensionName) == 0) {
            fp16_storage = true;
        } अन्यथा अगर (म_भेद("VK_KHR_shader_float16_int8", properties.extensionName) == 0) {
            fp16_compute = true;
#अगर defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
       } अन्यथा अगर (म_भेद("VK_KHR_cooperative_matrix", properties.extensionName) == 0 &&
                   !दो_पर्या("GGML_VK_DISABLE_COOPMAT")) {
            coopmat_support = true;
#पूर्ण_अगर
#अगर defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
        } अन्यथा अगर (म_भेद("VK_NV_cooperative_matrix2", properties.extensionName) == 0 &&
                   !दो_पर्या("GGML_VK_DISABLE_COOPMAT2")) {
            coopmat2_support = true;
#पूर्ण_अगर
        } अन्यथा अगर (म_भेद(VK_NV_COOPERATIVE_MATRIX_DECODE_VECTOR_EXTENSION_NAME, properties.extensionName) == 0 &&
                   !दो_पर्या("GGML_VK_DISABLE_COOPMAT2_DECODE_VECTOR")) {
            coopmat2_decode_vector_support = true;
#अगर defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
        } अन्यथा अगर (म_भेद("VK_KHR_shader_integer_dot_product", properties.extensionName) == 0 &&
                    !दो_पर्या("GGML_VK_DISABLE_INTEGER_DOT_PRODUCT")) {
            integer_dot_product = true;
#पूर्ण_अगर
#अगर defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        } अन्यथा अगर (म_भेद("VK_KHR_shader_bfloat16", properties.extensionName) == 0 &&
                    !दो_पर्या("GGML_VK_DISABLE_BFLOAT16")) {
            bfloat16_support = true;
#पूर्ण_अगर
#अगर defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT)
        } अन्यथा अगर (म_भेद(VK_EXT_SHADER_OCP_MICROSCALING_TYPES_EXTENSION_NAME, properties.extensionName) == 0) {
            ocp_microscaling_extension = true;
#पूर्ण_अगर
#अगर defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
        } अन्यथा अगर (म_भेद(VK_EXT_SHADER_FLOAT8_EXTENSION_NAME, properties.extensionName) == 0) {
            shader_float8_extension = true;
#पूर्ण_अगर
        } अन्यथा अगर (म_भेद("VK_VALVE_shader_mixed_float_dot_product", properties.extensionName) == 0 &&
                    !दो_पर्या("GGML_VK_DISABLE_DOT2")) {
            dot2_f16_support = true;
        }
    }

    स्थिर vk_device_architecture device_architecture = get_device_architecture(physical_device);

    स्थिर अक्षर* GGML_VK_DISABLE_F16 = दो_पर्या("GGML_VK_DISABLE_F16");
    bool force_disable_f16 = GGML_VK_DISABLE_F16 != nullptr;

    bool fp16 = !force_disable_f16 && fp16_storage && fp16_compute;

    vk::PhysicalDeviceProperties2 props2;
    vk::PhysicalDeviceMaintenance3Properties props3;
    vk::PhysicalDeviceSubgroupProperties subgroup_props;
    vk::PhysicalDeviceDriverProperties driver_props;
    vk::PhysicalDeviceShaderIntegerDotProductPropertiesKHR shader_integer_dot_product_props;
    props2.p_अग्र = &props3;
    props3.p_अग्र = &subgroup_props;
    subgroup_props.p_अग्र = &driver_props;

    // Pointer to the last chain element
    VkBaseOutStructure * last_struct = (VkBaseOutStructure *)&driver_props;

    अगर (integer_dot_product) {
        last_struct->p_अग्र = (VkBaseOutStructure *)&shader_integer_dot_product_props;
        last_struct = (VkBaseOutStructure *)&shader_integer_dot_product_props;
    }

    physical_device.getProperties2(&props2);

    VkPhysicalDeviceFeatures2 device_features2;
    device_features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    device_features2.p_अग्र = nullptr;

    VkPhysicalDeviceVulkan11Features vk11_features;
    vk11_features.p_अग्र = nullptr;
    vk11_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    device_features2.p_अग्र = &vk11_features;

    VkPhysicalDeviceVulkan12Features vk12_features;
    vk12_features.p_अग्र = nullptr;
    vk12_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vk11_features.p_अग्र = &vk12_features;

    // Pointer to the last chain element
    last_struct = (VkBaseOutStructure *)&vk12_features;

#अगर defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopmat_features;
    coopmat_features.p_अग्र = nullptr;
    coopmat_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
    coopmat_features.cooperativeMatrix = VK_FALSE;

    अगर (coopmat_support) {
        last_struct->p_अग्र = (VkBaseOutStructure *)&coopmat_features;
        last_struct = (VkBaseOutStructure *)&coopmat_features;
    }
#पूर्ण_अगर

    VkPhysicalDeviceShaderIntegerDotProductFeaturesKHR shader_integer_dot_product_features {};
    shader_integer_dot_product_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES_KHR;
    अगर (integer_dot_product) {
        last_struct->p_अग्र = (VkBaseOutStructure *)&shader_integer_dot_product_features;
        last_struct = (VkBaseOutStructure *)&shader_integer_dot_product_features;
    }

#अगर defined(VK_KHR_shader_bfloat16)
    VkPhysicalDeviceShaderBfloat16FeaturesKHR bfloat16_features {};
    bfloat16_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR;
    अगर (bfloat16_support) {
        last_struct->p_अग्र = (VkBaseOutStructure *)&bfloat16_features;
        last_struct = (VkBaseOutStructure *)&bfloat16_features;
    }
#पूर्ण_अगर

#अगर defined(VK_NV_cooperative_matrix2)
    VkPhysicalDeviceCooperativeMatrix2FeaturesNV coopmat2_features {};
    coopmat2_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_FEATURES_NV;
    अगर (coopmat2_support) {
        last_struct->p_अग्र = (VkBaseOutStructure *)&coopmat2_features;
        last_struct = (VkBaseOutStructure *)&coopmat2_features;
    }
#पूर्ण_अगर

    VkPhysicalDeviceCooperativeMatrixDecodeVectorFeaturesNV coopmat2_decode_vector_features {};
    coopmat2_decode_vector_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_DECODE_VECTOR_FEATURES_NV;
    अगर (coopmat2_decode_vector_support) {
        last_struct->p_अग्र = (VkBaseOutStructure *)&coopmat2_decode_vector_features;
        last_struct = (VkBaseOutStructure *)&coopmat2_decode_vector_features;
    }

    VkPhysicalDeviceShaderMixedFloatDotProductFeaturesVALVE dot2_features {};
    dot2_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_MIXED_FLOAT_DOT_PRODUCT_FEATURES_VALVE;
    अगर (dot2_f16_support) {
        last_struct->p_अग्र = (VkBaseOutStructure *)&dot2_features;
        last_struct = (VkBaseOutStructure *)&dot2_features;
    }

#अगर defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
    VkPhysicalDeviceShaderOCPMicroscalingTypesFeaturesEXT ocp_microscaling_features {};
    ocp_microscaling_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OCP_MICROSCALING_TYPES_FEATURES_EXT;
    VkPhysicalDeviceShaderFloat8FeaturesEXT shader_float8_features {};
    shader_float8_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT;
    अगर (ocp_microscaling_extension) {
        last_struct->p_अग्र = (VkBaseOutStructure *)&ocp_microscaling_features;
        last_struct = (VkBaseOutStructure *)&ocp_microscaling_features;
    }
    अगर (shader_float8_extension) {
        last_struct->p_अग्र = (VkBaseOutStructure *)&shader_float8_features;
        last_struct = (VkBaseOutStructure *)&shader_float8_features;
    }
#पूर्ण_अगर

    vkGetPhysicalDeviceFeatures2(physical_device, &device_features2);

    fp16 = fp16 && vk12_features.shaderFloat16;

#अगर defined(VK_KHR_shader_bfloat16)
    bool bf16 = bfloat16_support && bfloat16_features.shaderBFloat16Type;
#अन्यथा
    bool bf16 = false;
#पूर्ण_अगर

    uint32_t default_subgroup_आकार = get_subgroup_आकार("", device_architecture);
    स्थिर माप_प्रकार subgroup_आकार = (default_subgroup_आकार != 0) ? default_subgroup_आकार : subgroup_props.subgroup_आकार_2;
    स्थिर bool uma = props2.properties.deviceType == vk::PhysicalDeviceType::eIntegratedGpu;

    integer_dot_product = integer_dot_product
                       && shader_integer_dot_product_props.integerDotProduct4x8BitPackedSignedAccelerated
                       && shader_integer_dot_product_features.shaderIntegerDotProduct;

    coopmat_support = coopmat_support
#अगर defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
                   && coopmat_features.cooperativeMatrix
#पूर्ण_अगर
                   && ggml_vk_khr_cooperative_matrix_support(props2.properties, driver_props, device_architecture);

#अगर defined(VK_NV_cooperative_matrix2) && defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
    coopmat2_support = coopmat2_support &&
                       coopmat2_features.cooperativeMatrixWorkgroupScope &&
                       coopmat2_features.cooperativeMatrixFlexibleDimensions &&
                       coopmat2_features.cooperativeMatrixReductions &&
                       coopmat2_features.cooperativeMatrixConversions &&
                       coopmat2_features.cooperativeMatrixPerElementOperations &&
                       coopmat2_features.cooperativeMatrixTensorAddressing &&
                       coopmat2_features.cooperativeMatrixBlockLoads;
#अन्यथा
    coopmat2_support = false;
#पूर्ण_अगर

    coopmat2_decode_vector_support = coopmat2_decode_vector_support && coopmat2_decode_vector_features.cooperativeMatrixDecodeVector;
#अगर !defined(GGML_VULKAN_COOPMAT2_DECODE_VECTOR_GLSLC_SUPPORT)
    coopmat2_decode_vector_support = false;
#पूर्ण_अगर

    std::string matrix_cores = coopmat2_support ? (coopmat2_decode_vector_support ? "NV_coopmat2v" : "NV_coopmat2")
                             : coopmat_support  ? "KHR_coopmat"
                             : "none";

    bool dot2_f16 = dot2_f16_support && dot2_features.shaderMixedFloatDotProductFloat16AccFloat32;
    स्थिर अक्षर *fp16_str = fp16 ? (dot2_f16 ? "dot2" : "1") : "0";
#अगर defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
    स्थिर bool fp4 = ocp_microscaling_extension && ocp_microscaling_features.shaderFloat4 &&
                     shader_float8_extension && shader_float8_features.shaderFloat8 &&
                     !दो_पर्या("GGML_VK_DISABLE_OCP_FP4");
#अन्यथा
    GGML_UNUSED(ocp_microscaling_extension);
    GGML_UNUSED(shader_float8_extension);
    स्थिर bool fp4 = false;
#पूर्ण_अगर

    std::string device_name = props2.properties.deviceName.तथ्य();
    GGML_LOG_DEBUG("ggml_vulkan: %zu = %s (%s) | uma: %d | fp16: %s | bf16: %d | fp4: %d | warp size: %zu | shared memory: %d | int dot: %d | matrix cores: %s\n",
              idx, device_name.c_str(), driver_props.driverName.तथ्य(), uma, fp16_str, bf16, fp4, subgroup_आकार,
              props2.properties.limits.max_Compute_Shared_Memory_आकार, integer_dot_product, matrix_cores.c_str());

    अगर (props2.properties.deviceType == vk::PhysicalDeviceType::eCpu) {
        GGML_LOG_DEBUG("ggml_vulkan: Warning: Device type is CPU. This is probably not the device you want.\n");
    }
}

अटल DispatchLoaderDynamic ggml_vk_default_dispatcher_instance;

DispatchLoaderDynamic & ggml_vk_default_dispatcher() {
    वापस ggml_vk_default_dispatcher_instance;
}

व्योम ggml_vk_instance_init() {
    अगर (vk_instance_initialized) {
        वापस;
    }
    VK_LOG_DEBUG("ggml_vk_instance_init()");

    // See https://github.com/KhronosGroup/Vulkan-Hpp?tab=readme-ov-file#extensions--per-device-function-pointers-
    ggml_vk_default_dispatcher_instance.init(vkGetInstanceProcAddr);

    // vkEnumerateInstanceVersion is Vulkan 1.1. A null value indicated Vulkan 1.0.
    अगर (ggml_vk_default_dispatcher_instance.vkEnumerateInstanceVersion == nullptr) {
        GGML_LOG_ERROR("ggml_vulkan: Error: Vulkan 1.2 required.");
        फेंको vk::SystemError(vk::परिणाम_2::eErrorFeatureNotPresent, "Vulkan 1.2 required");
    }

    uint32_t api_version = vk::enumerateInstanceVersion();

    अगर (api_version < VK_API_VERSION_1_2) {
        GGML_LOG_ERROR("ggml_vulkan: Error: Vulkan 1.2 required.");
        फेंको vk::SystemError(vk::परिणाम_2::eErrorFeatureNotPresent, "Vulkan 1.2 required");
    }

    vk::ApplicationInfo app_info{ "ggml-vulkan", 1, nullptr, 0, api_version };

    स्थिर std::vector<vk::ExtensionProperties> instance_extensions = vk::enumerateInstanceExtensionProperties();
    स्थिर bool layer_settings = ggml_vk_instance_layer_settings_available();
#अगर_घोषित __APPLE__
    स्थिर bool portability_enumeration_ext = ggml_vk_instance_portability_enumeration_ext_available(instance_extensions);
#पूर्ण_अगर
    स्थिर bool debug_utils_ext = ggml_vk_instance_debug_utils_ext_available(instance_extensions) && दो_पर्या("GGML_VK_DEBUG_MARKERS") != nullptr;
    std::vector<स्थिर अक्षर*> layers;

    अगर (layer_settings) {
        layers.push_back("VK_LAYER_KHRONOS_validation");
    }
    std::vector<स्थिर अक्षर*> extensions;
    अगर (layer_settings) {
        extensions.push_back("VK_EXT_layer_settings");
    }
#अगर_घोषित __APPLE__
    अगर (portability_enumeration_ext) {
        extensions.push_back("VK_KHR_portability_enumeration");
    }
#पूर्ण_अगर
    अगर (debug_utils_ext) {
        extensions.push_back("VK_EXT_debug_utils");
    }
    VkBool32 enable_best_practice = layer_settings;
    std::vector<vk::LayerSettingEXT> settings = {
        {
            "VK_LAYER_KHRONOS_validation",
            "validate_best_practices",
            vk::LayerSettingTypeEXT::eBool32,
            1,
            &enable_best_practice
        },
    };
    vk::LayerSettingsCreateInfoEXT layer_setting_info(settings);
    vk::InstanceCreateInfo instance_create_info(vk::InstanceCreateFlags{}, &app_info, layers, extensions, &layer_setting_info);
#अगर_घोषित __APPLE__
    अगर (portability_enumeration_ext) {
        instance_create_info.flags |= vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
    }
#पूर्ण_अगर

    vk_instance.instance = vk::createInstance(instance_create_info);
    vk_instance_initialized = true;

    अगर (debug_utils_ext) {
        vk_instance.debug_utils_support              = true;
        vk_instance.pfn_vkSetDebugUtilsObjectNameEXT = (PFN_vkSetDebugUtilsObjectNameEXT) vkGetInstanceProcAddr(vk_instance.instance, "vkSetDebugUtilsObjectNameEXT");
        vk_instance.pfn_vkQueueBeginDebugUtilsLabelEXT = (PFN_vkQueueBeginDebugUtilsLabelEXT) vkGetInstanceProcAddr(vk_instance.instance, "vkQueueBeginDebugUtilsLabelEXT");
        vk_instance.pfn_vkQueueEndDebugUtilsLabelEXT = (PFN_vkQueueEndDebugUtilsLabelEXT) vkGetInstanceProcAddr(vk_instance.instance, "vkQueueEndDebugUtilsLabelEXT");
        vk_instance.pfn_vkCmdBeginDebugUtilsLabelEXT = (PFN_vkCmdBeginDebugUtilsLabelEXT) vkGetInstanceProcAddr(vk_instance.instance, "vkCmdBeginDebugUtilsLabelEXT");
        vk_instance.pfn_vkCmdEndDebugUtilsLabelEXT =   (PFN_vkCmdEndDebugUtilsLabelEXT) vkGetInstanceProcAddr(vk_instance.instance, "vkCmdEndDebugUtilsLabelEXT");
        vk_instance.pfn_vk_Cmd_जोड़ो_Debug_Utils_Label_EXT = (PFN_vk_Cmd_जोड़ो_Debug_Utils_Label_EXT) vkGetInstanceProcAddr(vk_instance.instance, "vkCmdInsertDebugUtilsLabelEXT");
    }

    vk_perf_logger_enabled = दो_पर्या("GGML_VK_PERF_LOGGER") != nullptr;
    vk_perf_logger_concurrent = दो_पर्या("GGML_VK_PERF_LOGGER_CONCURRENT") != nullptr;
    vk_enable_sync_logger = दो_पर्या("GGML_VK_SYNC_LOGGER") != nullptr;
    vk_memory_logger_enabled = दो_पर्या("GGML_VK_MEMORY_LOGGER") != nullptr;
    स्थिर अक्षर* GGML_VK_PIPELINE_STATS = दो_पर्या("GGML_VK_PIPELINE_STATS");
    अगर (GGML_VK_PIPELINE_STATS != nullptr) {
        vk_pipeline_stats_filter = GGML_VK_PIPELINE_STATS;
    }
    स्थिर अक्षर* GGML_VK_PERF_LOGGER_FREQUENCY = दो_पर्या("GGML_VK_PERF_LOGGER_FREQUENCY");

    अगर (GGML_VK_PERF_LOGGER_FREQUENCY != nullptr) {
        vk_perf_logger_frequency = std::stoul(GGML_VK_PERF_LOGGER_FREQUENCY);
    }

    // See https://github.com/KhronosGroup/Vulkan-Hpp?tab=readme-ov-file#extensions--per-device-function-pointers-
    VULKAN_HPP_DEFAULT_DISPATCHER.init(vk_instance.instance);

    std::vector<vk::PhysicalDevice> devices = vk_instance.instance.enumeratePhysicalDevices();

    // Emulate behavior of CUDA_VISIBLE_DEVICES for Vulkan
    अक्षर * devices_env = दो_पर्या("GGML_VK_VISIBLE_DEVICES");
    अगर (devices_env != nullptr) {
        माप_प्रकार अंक_available_devices = devices.size();

        std::string devices(devices_env);
        std::replace(devices.begin(), devices.end(), ',', ' ');

        std::stringstream ss(devices);
        माप_प्रकार tmp;
        जबतक (ss >> tmp) {
            अगर(tmp >= अंक_available_devices) {
                std::अ_त्रुटि << "ggml_vulkan: Invalid device index " << tmp << " in GGML_VK_VISIBLE_DEVICES." << std::प_पूर्ण;
                फेंको std::runtime_error("Invalid Vulkan device index");
            }
            vk_instance.device_indices.push_back(tmp);
        }
    } अन्यथा {
        // If no vulkan devices are found, return early
        अगर (devices.empty()) {
            GGML_LOG_INFO("ggml_vulkan: No devices found.\n");
            वापस;
        }

        // Default to using all dedicated GPUs
        क्रम (माप_प्रकार i = 0; i < devices.size(); i++) {
            vk::PhysicalDeviceProperties2 new_props;
            vk::PhysicalDeviceDriverProperties new_driver;
            vk::PhysicalDeviceIDProperties new_id;
            new_props.p_अग्र = &new_driver;
            new_driver.p_अग्र = &new_id;
            devices[i].getProperties2(&new_props);

            अगर ((new_props.properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu || new_props.properties.deviceType == vk::PhysicalDeviceType::eIntegratedGpu) && ggml_vk_device_is_supported(devices[i])) {
                // Check if there are two physical devices corresponding to the same GPU
                // This handles the case where the same GPU appears with different drivers (e.g., RADV + AMDVLK on Linux),
                // see https://github.com/ggml-org/llama.cpp/pull/7582 for original deduplication.
                // MoltenVK on macOS may report the same UUID for distinct GPUs on multi-GPU cards,
                // see https://github.com/KhronosGroup/MoltenVK/issues/2683. Skip when both old/new
                // driver is MoltenVK
                स्वतः old_device = std::find_if(
                    vk_instance.device_indices.begin(),
                    vk_instance.device_indices.end(),
                    [&devices, &new_id, &new_driver](स्थिर माप_प्रकार k){
                        vk::PhysicalDeviceProperties2 old_props;
                        vk::PhysicalDeviceDriverProperties old_driver;
                        vk::PhysicalDeviceIDProperties old_id;
                        old_props.p_अग्र = &old_driver;
                        old_driver.p_अग्र = &old_id;
                        devices[k].getProperties2(&old_props);

                        bool same_uuid = std::equal(std::begin(old_id.deviceUUID), std::end(old_id.deviceUUID), std::begin(new_id.deviceUUID));
                        same_uuid = same_uuid || (
                            old_id.deviceLUIDValid && new_id.deviceLUIDValid &&
                            std::equal(std::begin(old_id.deviceLUID), std::end(old_id.deviceLUID), std::begin(new_id.deviceLUID))
                        );
                        bool both_molten_vk = (new_driver.driverID == vk::DriverId::eMoltenvk && old_driver.driverID == vk::DriverId::eMoltenvk);

                        वापस same_uuid && !both_molten_vk;
                    }
                );
                अगर (old_device == vk_instance.device_indices.end()) {
                    vk_instance.device_indices.push_back(i);
                } अन्यथा {
                    // There can be two physical devices corresponding to the same GPU if there are 2 different drivers
                    // This can cause error when splitting layers aross the devices, need to keep only 1
                    VK_LOG_DEBUG("Device " << i << " and device " << *old_device << " have the same deviceUUID");

                    vk::PhysicalDeviceProperties2 old_props;
                    vk::PhysicalDeviceDriverProperties old_driver;
                    old_props.p_अग्र = &old_driver;
                    devices[*old_device].getProperties2(&old_props);

                    std::map<vk::DriverId, पूर्णांक> driver_priorities {};
                    पूर्णांक old_priority = std::numeric_limits<पूर्णांक>::max();
                    पूर्णांक new_priority = std::numeric_limits<पूर्णांक>::max();

                    // Check https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkDriverId.html for the list of driver id
                    // Smaller number -> higher priority
                    चयन (old_props.properties.vendorID) {
                        हाल VK_VENDOR_ID_AMD:
                            driver_priorities[vk::DriverId::eMesaRadv] = 1;
                            driver_priorities[vk::DriverId::eAmdOpenSource] = 2;
                            driver_priorities[vk::DriverId::eAmdProprietary] = 3;
                            अवरोध;
                        हाल VK_VENDOR_ID_INTEL:
                            driver_priorities[vk::DriverId::eIntelOpenSourceMESA] = 1;
                            driver_priorities[vk::DriverId::eIntelProprietaryWindows] = 2;
                            अवरोध;
                        हाल VK_VENDOR_ID_NVIDIA:
                            driver_priorities[vk::DriverId::eNvidiaProprietary] = 1;
#अगर defined(VK_API_VERSION_1_3) && VK_HEADER_VERSION >= 235
                            driver_priorities[vk::DriverId::eMesaNvk] = 2;
#पूर्ण_अगर
                            अवरोध;
                        हाल VK_VENDOR_ID_QUALCOMM:
                            driver_priorities[vk::DriverId::eQualcommProprietary] = 1;
                            driver_priorities[vk::DriverId::eMesaTurnip] = 2;
                            अवरोध;
                    }
                    driver_priorities[vk::DriverId::eMesaDozen] = 100;

                    अगर (driver_priorities.count(old_driver.driverID)) {
                        old_priority = driver_priorities[old_driver.driverID];
                    }
                    अगर (driver_priorities.count(new_driver.driverID)) {
                        new_priority = driver_priorities[new_driver.driverID];
                    }

                    अगर (new_priority < old_priority) {
                        स्वतः r = std::हटाओ(vk_instance.device_indices.begin(), vk_instance.device_indices.end(), *old_device);
                        vk_instance.device_indices.erase(r, vk_instance.device_indices.end());
                        vk_instance.device_indices.push_back(i);

                        VK_LOG_DEBUG("Prioritize device " << i << " driver " << new_driver.driverName << " over device " << *old_device << " driver " << old_driver.driverName);
                    }
                    अन्यथा {
                        VK_LOG_DEBUG("Prioritize device " << *old_device << " driver " << old_driver.driverName << " over device " << i << " driver " << new_driver.driverName << std::प_पूर्ण);
                    }
                }
            }
        }

        // If no GPUs found, fall back to the first non-CPU device.
        // If only CPU devices are available, return without devices.
        अगर (vk_instance.device_indices.empty()) {
            क्रम (माप_प्रकार i = 0; i < devices.size(); i++) {
                अगर (devices[i].getProperties().deviceType != vk::PhysicalDeviceType::eCpu) {
                    vk_instance.device_indices.push_back(i);
                    अवरोध;
                }
            }
        }

        अगर (vk_instance.device_indices.empty()) {
            GGML_LOG_INFO("ggml_vulkan: No devices found.\n");
            वापस;
        }
    }
    GGML_LOG_DEBUG("ggml_vulkan: Found %zu Vulkan devices:\n", vk_instance.device_indices.size());

    माप_प्रकार host_बफर_import_गिनती = 0;
    क्रम (माप_प्रकार i = 0; i < vk_instance.device_indices.size(); i++) {
        vk::PhysicalDevice vkdev = devices[vk_instance.device_indices[i]];
        std::vector<vk::ExtensionProperties> extensionprops = vkdev.enumerateDeviceExtensionProperties();

        bool membudget_supported = false;
        bool external_memory_host_supported = false;
        क्रम (स्थिर स्वतः & ext : extensionprops) {
            अगर (म_भेद(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME, ext.extensionName) == 0) {
                membudget_supported = true;
            } अन्यथा अगर (म_भेद("VK_EXT_external_memory_host", ext.extensionName) == 0) {
                external_memory_host_supported = true;
            }
        }

        vk_instance.device_supports_membudget.push_back(membudget_supported);

        bool host_बफर_import = false;
        अगर (external_memory_host_supported) {
            vk::PhysicalDeviceProperties2 props2;
            vk::PhysicalDeviceDriverProperties driver_props;
            vk::PhysicalDeviceExternalMemoryHostPropertiesEXT ext_props;
            props2.p_अग्र = &driver_props;
            driver_props.p_अग्र = &ext_props;
            vkdev.getProperties2(&props2);
            host_बफर_import = ggml_vk_external_memory_host_usable(driver_props.driverID) &&
                                 ggml_vk_host_बफर_import_usable(props2.properties.vendorID);
            अगर (host_बफर_import) {
                vk_instance.host_import_alignment = std::max<माप_प्रकार>(vk_instance.host_import_alignment, ext_props.minImportedHostPointerAlignment);
                host_बफर_import_गिनती++;
            }
        }
        vk_instance.device_host_बफर_import.push_back(host_बफर_import);

        ggml_vk_print_gpu_info(i);
    }

    vk_instance.multi_device = host_बफर_import_गिनती > 1 && vk_instance.device_host_बफर_import[0];
}

व्योम ggml_vk_init(ggml_backend_vk_context * ctx, माप_प्रकार idx) {
    VK_LOG_DEBUG("ggml_vk_init(" << ctx->name << ", " << idx << ")");
    ggml_vk_instance_init();
    GGML_ASSERT(idx < vk_instance.device_indices.size());

    ctx->name = GGML_VK_NAME + std::to_string(idx);

    ctx->device = ggml_vk_get_device(idx);

    ctx->semaphore_idx = 0;
    ctx->event_idx = 0;

    ctx->prealloc_आकार_x = 0;
    ctx->prealloc_आकार_y = 0;
    ctx->prealloc_आकार_split_k = 0;
    // Fixed size of 1KB, for deterministic behavior
    ctx->prealloc_आकार_add_rms_partials = 1024;

    ctx->fence = ctx->device->device.createFence({});
    ctx->almost_ready_fence = ctx->device->device.createFence({});

    ctx->compute_cmd_pool.init(ctx->device, ctx->device->compute_queue.लो());
    अगर (ctx->device->async_use_transfer_queue) {
        vk::SemaphoreTypeCreateInfo tci{ vk::SemaphoreType::eTimeline, 0 };
        vk::SemaphoreCreateInfo ci{};
        ci.setPNext(&tci);
        ctx->transfer_semaphore.s = ctx->device->device.createSemaphore(ci);
        ctx->transfer_semaphore.value = 0;

        ctx->transfer_cmd_pool.init(ctx->device, ctx->device->transfer_queue.लो());
    }

    अगर (vk_perf_logger_enabled) {
        ctx->perf_logger = std::unique_ptr<vk_perf_logger>(नया vk_perf_logger());
    }

#अगर_घोषित GGML_VULKAN_CHECK_RESULTS
    स्थिर अक्षर* skip_checks = दो_पर्या("GGML_VULKAN_SKIP_CHECKS");
    vk_skip_checks = (skip_checks == शून्य ? 0 : म_से_प(skip_checks));
    स्थिर अक्षर* output_tensor = दो_पर्या("GGML_VULKAN_OUTPUT_TENSOR");
    vk_output_tensor = (output_tensor == शून्य ? 0 : म_से_प(output_tensor));
#पूर्ण_अगर
}

vk_pipeline ggml_vk_get_to_fp16(ggml_backend_vk_context * ctx, ggml_type type) {
    VK_LOG_DEBUG("ggml_vk_get_to_fp16()");
    चयन (type) {
        हाल GGML_TYPE_F32:
        हाल GGML_TYPE_Q1_0:
        हाल GGML_TYPE_Q2_0:
        हाल GGML_TYPE_Q4_0:
        हाल GGML_TYPE_Q4_1:
        हाल GGML_TYPE_Q5_0:
        हाल GGML_TYPE_Q5_1:
        हाल GGML_TYPE_Q8_0:
        हाल GGML_TYPE_Q2_K:
        हाल GGML_TYPE_Q3_K:
        हाल GGML_TYPE_Q4_K:
        हाल GGML_TYPE_Q5_K:
        हाल GGML_TYPE_Q6_K:
        हाल GGML_TYPE_IQ1_S:
        हाल GGML_TYPE_IQ1_M:
        हाल GGML_TYPE_IQ2_XXS:
        हाल GGML_TYPE_IQ2_XS:
        हाल GGML_TYPE_IQ2_S:
        हाल GGML_TYPE_IQ3_XXS:
        हाल GGML_TYPE_IQ3_S:
        हाल GGML_TYPE_IQ4_XS:
        हाल GGML_TYPE_IQ4_NL:
        हाल GGML_TYPE_MXFP4:
        हाल GGML_TYPE_NVFP4:
        हाल GGML_TYPE_TQ1_0:
        हाल GGML_TYPE_TQ2_0:
            अवरोध;
        शेष:
            वापस nullptr;
    }

    वापस ctx->device->pipeline_dequant[type];
}

अटल vk_pipeline ggml_vk_get_dequantize_mul_mat_vec(ggml_backend_vk_context * ctx, ggml_type a_type, ggml_type b_type, uint32_t अंक_cols, uint32_t m, uint32_t k) {
    VK_LOG_DEBUG("ggml_vk_get_dequantize_mul_mat_vec()");
    GGML_ASSERT(b_type == GGML_TYPE_F32 || b_type == GGML_TYPE_F16 || b_type == GGML_TYPE_Q8_1);
    GGML_ASSERT(अंक_cols >= 1 && अंक_cols <= mul_mat_vec_max_cols);

    अगर (b_type == GGML_TYPE_Q8_1) {
        चयन (a_type) {
            हाल GGML_TYPE_Q2_0:
            हाल GGML_TYPE_Q4_0:
            हाल GGML_TYPE_Q4_1:
            हाल GGML_TYPE_Q5_0:
            हाल GGML_TYPE_Q5_1:
            हाल GGML_TYPE_Q8_0:
            हाल GGML_TYPE_MXFP4:
            हाल GGML_TYPE_Q2_K:
            हाल GGML_TYPE_Q3_K:
            हाल GGML_TYPE_Q4_K:
            हाल GGML_TYPE_Q5_K:
            हाल GGML_TYPE_Q6_K:
            हाल GGML_TYPE_IQ1_S:
            हाल GGML_TYPE_IQ1_M:
            हाल GGML_TYPE_IQ4_XS:
                अवरोध;
            शेष:
                वापस nullptr;
        }
    }

    चयन (a_type) {
        हाल GGML_TYPE_F32:
        हाल GGML_TYPE_F16:
        हाल GGML_TYPE_BF16:
        हाल GGML_TYPE_Q1_0:
        हाल GGML_TYPE_Q2_0:
        हाल GGML_TYPE_Q4_0:
        हाल GGML_TYPE_Q4_1:
        हाल GGML_TYPE_Q5_0:
        हाल GGML_TYPE_Q5_1:
        हाल GGML_TYPE_Q8_0:
        हाल GGML_TYPE_Q2_K:
        हाल GGML_TYPE_Q3_K:
        हाल GGML_TYPE_Q4_K:
        हाल GGML_TYPE_Q5_K:
        हाल GGML_TYPE_Q6_K:
        हाल GGML_TYPE_IQ1_S:
        हाल GGML_TYPE_IQ1_M:
        हाल GGML_TYPE_IQ2_XXS:
        हाल GGML_TYPE_IQ2_XS:
        हाल GGML_TYPE_IQ2_S:
        हाल GGML_TYPE_IQ3_XXS:
        हाल GGML_TYPE_IQ3_S:
        हाल GGML_TYPE_IQ4_XS:
        हाल GGML_TYPE_IQ4_NL:
        हाल GGML_TYPE_MXFP4:
        हाल GGML_TYPE_NVFP4:
        हाल GGML_TYPE_TQ1_0:
        हाल GGML_TYPE_TQ2_0:
            अवरोध;
        शेष:
            वापस nullptr;
    }

    // heuristic to choose workgroup size
    uint32_t dmmv_wg = DMMV_WG_आकार_SUBGROUP;
    अगर ((ctx->device->vendor_id == VK_VENDOR_ID_NVIDIA && ctx->device->architecture != vk_device_architecture::NVIDIA_PRE_TURING) || ctx->device->vendor_id == VK_VENDOR_ID_INTEL) {
        // Prefer larger workgroups when M is small, to spread the work out more
        // and keep more SMs busy.
        // q6_k seems to prefer small workgroup size even for "medium" values of M.
        अगर (a_type == GGML_TYPE_Q6_K) {
            अगर (m < 4096 && k >= 1024) {
                dmmv_wg = DMMV_WG_आकार_LARGE;
            }
        } अन्यथा {
            अगर (m <= 8192 && k >= 1024) {
                dmmv_wg = DMMV_WG_आकार_LARGE;
            }
        }
    }

    अगर (b_type == GGML_TYPE_Q8_1) {
        अगर (ctx->device->vendor_id == VK_VENDOR_ID_INTEL) {
            dmmv_wg = DMMV_WG_आकार_SUBGROUP;
        }
        वापस ctx->device->pipeline_dequant_mul_mat_vec_q8_1_f32[dmmv_wg][a_type][अंक_cols-1];
    }

    वापस b_type == GGML_TYPE_F32 ? ctx->device->pipeline_dequant_mul_mat_vec_f32_f32[dmmv_wg][a_type][अंक_cols-1] : ctx->device->pipeline_dequant_mul_mat_vec_f16_f32[dmmv_wg][a_type][अंक_cols-1];
}

अटल vk_pipeline ggml_vk_get_dequantize_mul_mat_vec_id(ggml_backend_vk_context * ctx, ggml_type a_type, ggml_type b_type, uint32_t m, uint32_t k) {
    VK_LOG_DEBUG("ggml_vk_get_dequantize_mul_mat_vec_id()");
    GGML_ASSERT(b_type == GGML_TYPE_F32 || b_type == GGML_TYPE_Q8_1);

    अगर (b_type == GGML_TYPE_Q8_1) {
        चयन (a_type) {
            हाल GGML_TYPE_Q2_0:
            हाल GGML_TYPE_Q4_0:
            हाल GGML_TYPE_Q4_1:
            हाल GGML_TYPE_Q5_0:
            हाल GGML_TYPE_Q5_1:
            हाल GGML_TYPE_Q8_0:
            हाल GGML_TYPE_MXFP4:
            हाल GGML_TYPE_Q2_K:
            हाल GGML_TYPE_Q3_K:
            हाल GGML_TYPE_Q4_K:
            हाल GGML_TYPE_Q5_K:
            हाल GGML_TYPE_Q6_K:
            हाल GGML_TYPE_IQ1_S:
            हाल GGML_TYPE_IQ1_M:
            हाल GGML_TYPE_IQ4_XS:
                अवरोध;
            शेष:
                वापस nullptr;
        }
    }

    चयन (a_type) {
        हाल GGML_TYPE_F32:
        हाल GGML_TYPE_F16:
        हाल GGML_TYPE_BF16:
        हाल GGML_TYPE_Q1_0:
        हाल GGML_TYPE_Q2_0:
        हाल GGML_TYPE_Q4_0:
        हाल GGML_TYPE_Q4_1:
        हाल GGML_TYPE_Q5_0:
        हाल GGML_TYPE_Q5_1:
        हाल GGML_TYPE_Q8_0:
        हाल GGML_TYPE_Q2_K:
        हाल GGML_TYPE_Q3_K:
        हाल GGML_TYPE_Q4_K:
        हाल GGML_TYPE_Q5_K:
        हाल GGML_TYPE_Q6_K:
        हाल GGML_TYPE_IQ1_S:
        हाल GGML_TYPE_IQ1_M:
        हाल GGML_TYPE_IQ2_XXS:
        हाल GGML_TYPE_IQ2_XS:
        हाल GGML_TYPE_IQ2_S:
        हाल GGML_TYPE_IQ3_XXS:
        हाल GGML_TYPE_IQ3_S:
        हाल GGML_TYPE_IQ4_XS:
        हाल GGML_TYPE_IQ4_NL:
        हाल GGML_TYPE_MXFP4:
        हाल GGML_TYPE_NVFP4:
        हाल GGML_TYPE_TQ1_0:
        हाल GGML_TYPE_TQ2_0:
            अवरोध;
        शेष:
            वापस nullptr;
    }

    // heuristic to choose workgroup size
    uint32_t dmmv_wg = DMMV_WG_आकार_SUBGROUP;
    अगर ((ctx->device->vendor_id == VK_VENDOR_ID_NVIDIA && ctx->device->architecture != vk_device_architecture::NVIDIA_PRE_TURING) || ctx->device->vendor_id == VK_VENDOR_ID_INTEL) {
        // Prefer larger workgroups when M is small, to spread the work out more
        // and keep more SMs busy.
        // q6_k seems to prefer small workgroup size even for "medium" values of M.
        अगर (a_type == GGML_TYPE_Q6_K) {
            अगर (m < 4096 && k >= 1024) {
                dmmv_wg = DMMV_WG_आकार_LARGE;
            }
        } अन्यथा {
            अगर (m <= 8192 && k >= 1024) {
                dmmv_wg = DMMV_WG_आकार_LARGE;
            }
        }
    }

    अगर (b_type == GGML_TYPE_Q8_1) {
        अगर (ctx->device->vendor_id == VK_VENDOR_ID_INTEL) {
            dmmv_wg = DMMV_WG_आकार_SUBGROUP;
        }
        वापस ctx->device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[dmmv_wg][a_type];
    }

    वापस ctx->device->pipeline_dequant_mul_mat_vec_id_f32[dmmv_wg][a_type];
}

vk_subbuffer ggml_vk_tensor_subbuffer(
    स्थिर ggml_backend_vk_context * ctx, स्थिर ggml_tensor * tensor, bool allow_misalign) {

    vk_बफर बफर = nullptr;
    माप_प्रकार offset = 0;
    अगर (ctx->device->uma) {
        ggml_vk_host_get(ctx->device, tensor->तथ्य, बफर, offset);
    }
    अगर (!बफर) {
        स्वतः buf_ctx = (ggml_backend_vk_बफर_context *)tensor->बफर->context;
        बफर = buf_ctx->dev_बफर;
        offset = vk_tensor_offset(tensor) + tensor->view_offs;
    }
    GGML_ASSERT(बफर != nullptr);

    माप_प्रकार size = ggml_nbytes(tensor);

    स्थिर माप_प्रकार descriptor_offset = ggml_vk_descriptor_offset(
        offset, ctx->device->properties.limits.min_Storage_बफर_Offset_Alignment, ggml_type_आकार(tensor->type));
    स्थिर माप_प्रकार misalign_bytes = offset - descriptor_offset;
    // The shader must support misaligned offsets when indexing into the buffer
    GGML_ASSERT(allow_misalign || misalign_bytes == 0);
    offset = descriptor_offset;
    size += misalign_bytes;

    वापस vk_subbuffer{बफर, offset, size};
}

अटल vk_command_बफर* ggml_vk_get_or_create_cmd_बफर(vk_device& device, vk_command_pool& pool) {
    क्रम (स्वतः& cmd_बफर : pool.cmd_buffers) {
        अगर (!cmd_बफर.in_use) {
            cmd_बफर.use_गणक++;
            cmd_बफर.in_use = true;
            वापस &cmd_बफर;
        }
    }
    वापस ggml_vk_create_cmd_बफर(device, pool);
}

अटल vk_submission ggml_vk_begin_submission(vk_device& device, vk_command_pool& p, bool one_time = true) {
    vk_submission s;
    s.बफर = ggml_vk_get_or_create_cmd_बफर(device, p);
    अगर (one_time) {
        s.बफर->buf.begin({ vk::Command_बफर_Usage_Flag_Bits::eOneTimeSubmit });
    } अन्यथा {
        s.बफर->buf.begin({ vk::Command_बफर_Usage_Flags{} });
    }

    वापस s;
}

व्योम ggml_vk_cmd_label_begin(vk::Command_बफर buf, स्थिर अक्षर * name) {
    vk::DebugUtilsLabelEXT label = {};
    label.pLabelName = name;
    label.color = std::array<भग्न, 4>{1.0f, 1.0f, 1.0f, 1.0f};
    vk_instance.pfn_vkCmdBeginDebugUtilsLabelEXT(buf, reinterpret_cast<VkDebugUtilsLabelEXT *>(&label));
}

व्योम ggml_vk_ctx_end(vk_context& ctx) {
    VK_LOG_DEBUG("ggml_vk_ctx_end(" << ctx << ", " << ctx->seqs.size() << ")");
    अगर (ctx->s == nullptr) {
        वापस;
    }

    // close open labels so this buffer is balanced; reopened in ggml_vk_ctx_begin
    अगर (vk_instance.debug_utils_support) {
        क्रम (माप_प्रकार i = 0; i < ctx->debug_labels.size(); i++) {
            vk_instance.pfn_vkCmdEndDebugUtilsLabelEXT(ctx->s->बफर->buf);
        }
        // the enclosing per-command-buffer region
        vk_instance.pfn_vkCmdEndDebugUtilsLabelEXT(ctx->s->बफर->buf);
    }

    ctx->s->बफर->buf.end();
    ctx->s = nullptr;
}

व्योम ggml_vk_ctx_begin(vk_device& device, vk_context& subctx) {
    VK_LOG_DEBUG("ggml_vk_ctx_begin(" << device->name << ")");
    अगर (subctx->s != nullptr) {
        ggml_vk_ctx_end(subctx);
    }

    subctx->seqs.push_back({ ggml_vk_begin_submission(device, *subctx->p) });
    subctx->s = subctx->seqs[subctx->seqs.size() - 1].तथ्य();

    अगर (vk_instance.debug_utils_support) {
        // outermost region, one per command buffer, so the gaps between submits stand out
        स्थिर std::string name = "submit " + std::to_string(device->debug_cmdbuf_idx++);
        ggml_vk_cmd_label_begin(subctx->s->बफर->buf, name.c_str());

        // reopen labels left open when the previous command buffer was submitted
        क्रम (स्थिर std::string & label : subctx->debug_labels) {
            ggml_vk_cmd_label_begin(subctx->s->बफर->buf, label.c_str());
        }
    }
}

vk_context ggml_vk_get_compute_ctx(ggml_backend_vk_context * ctx) {
    vk_context परिणाम;
    अगर (!ctx->compute_ctx.expired()) {
        परिणाम = ctx->compute_ctx.lock();
    } अन्यथा {
        परिणाम = ggml_vk_create_context(ctx, ctx->compute_cmd_pool);

        ctx->compute_ctx = परिणाम;
        ggml_vk_ctx_begin(ctx->device, परिणाम);
    }

    अगर (ctx->device->async_use_transfer_queue && ctx->transfer_semaphore_last_submitted < ctx->transfer_semaphore.value) {
        परिणाम->s->wait_semaphores.push_back(ctx->transfer_semaphore);
        ctx->transfer_semaphore_last_submitted = ctx->transfer_semaphore.value;
    }

    वापस परिणाम;
}

vk_context ggml_vk_get_transfer_ctx(ggml_backend_vk_context * ctx) {
    vk_context परिणाम;
    अगर (!ctx->transfer_ctx.expired()) {
        परिणाम = ctx->transfer_ctx.lock();
    } अन्यथा {
        परिणाम = ggml_vk_create_context(ctx, ctx->transfer_cmd_pool);

        ctx->transfer_ctx = परिणाम;
        ggml_vk_ctx_begin(ctx->device, परिणाम);
    }

    वापस परिणाम;
}

bool ggml_vk_submit_transfer_ctx(ggml_backend_vk_context * ctx) {
    अगर (!ctx->device->async_use_transfer_queue || ctx->transfer_ctx.expired()) {
        वापस false;
    }

    vk_context cpy_ctx = ctx->transfer_ctx.lock();
    ggml_vk_ctx_end(cpy_ctx);

    क्रम (स्वतः& cpy : cpy_ctx->in_memcpys) {
        स_नकल(cpy.dst, cpy.src, cpy.n);
    }

    ctx->transfer_semaphore.value++;
    cpy_ctx->seqs.back().back().signal_semaphores.push_back(ctx->transfer_semaphore);

    ggml_vk_submit(cpy_ctx, {});
    ctx->transfer_ctx.reset();
    वापस true;
}

माप_प्रकार ggml_vk_align_आकार(माप_प्रकार width, माप_प्रकार align) {
    VK_LOG_DEBUG("ggml_vk_align_size(" << width << ", " << align << ")");
    वापस CEIL_DIV(width, align) * align;
}

व्योम deferred_memcpy(व्योम * dst, स्थिर व्योम * src, माप_प्रकार size, std::vector<vk_staging_memcpy>* memcpys) {
    अगर (memcpys == nullptr) {
        स_नकल(dst, src, size);
    } अन्यथा {
        memcpys->emplace_back(dst, src, size);
    }
}

व्योम deferred_memset(व्योम * dst, uint32_t val, माप_प्रकार size, std::vector<vk_staging_memset>* memsets) {
    अगर (memsets == nullptr) {
        स_रखो(dst, val, size);
    } अन्यथा {
        memsets->emplace_back(dst, val, size);
    }
}

अटल uint32_t ggml_vk_guess_split_k(ggml_backend_vk_context * ctx, uint32_t m, uint32_t n, uint32_t k, bool disable_split_k, स्थिर vk_pipeline& pipeline) {
    VK_LOG_DEBUG("ggml_vk_guess_split_k(" << m << ", " << n << ", " << k << ", " << disable_split_k << ")");

    अगर (disable_split_k) {
        वापस 1;
    }

    uint32_t split_k = 1;
    अगर (ctx->device->shader_core_गिनती != 0 && n >= pipeline->wg_denoms[1]) {
        // If k is 'large' and the SMs will fill less than halfway, use split_k.
        uint32_t m_tiles = CEIL_DIV(m, pipeline->wg_denoms[0]);
        uint32_t n_tiles = CEIL_DIV(n, pipeline->wg_denoms[1]);

        अगर (k >= 2048) {
            अगर (m_tiles * n_tiles <= ctx->device->shader_core_गिनती / 2) {
                split_k = ctx->device->shader_core_गिनती / (m_tiles * n_tiles);
            } अन्यथा अगर (m_tiles * n_tiles <= ctx->device->shader_core_गिनती * 2 / 3) {
                split_k = 3;
            }
            // Cap the split at 8x. Unless k is huge this is a lot of overhead.
            split_k = std::min(split_k, 8u);

            // ggml_vk_matmul will align the splits to be a multiple of 256.
            // If this rounded up size would cause the last split to be empty,
            // then reduce the split count.
            जबतक (true) {
                अगर (split_k == 1) {
                    अवरोध;
                }
                uint32_t k_split = CEIL_DIV(k, split_k);
                k_split = ROUNDUP_POW2(k_split, 256);
                अगर (k_split * (split_k - 1) < k) {
                    अवरोध;
                }
                split_k--;
            }
        }
    }

    वापस split_k;
}

व्योम ggml_vk_matmul(
        ggml_backend_vk_context * ctx, vk_context& subctx, vk_pipeline& pipeline,
        vk_subbuffer&& a, vk_subbuffer&& b, vk_subbuffer&& d, vk_subbuffer&& split_k_बफर,
        uint32_t m, uint32_t n, uint32_t k, uint32_t stride_a, uint32_t stride_b, uint32_t stride_d,
        uint32_t batch_stride_a, uint32_t batch_stride_b, uint32_t batch_stride_d,
        uint32_t split_k, uint32_t batch, uint32_t ne02, uint32_t ne12, uint32_t broadcast2, uint32_t broadcast3,
        uint32_t padded_n, uint64_t b_col_bytes, uint64_t d_col_bytes) {
        VK_LOG_DEBUG("ggml_vk_matmul(a: (" << a.बफर->बफर << ", " << a.offset << ", " << a.size << "), b: (" << b.बफर->बफर << ", " << b.offset << ", " << b.size << "), d: (" << d.बफर->बफर << ", " << d.offset << ", " << d.size << "), split_k: (" << (split_k_बफर.बफर != nullptr ? split_k_बफर.बफर->बफर : VK_NULL_HANDLE) << ", " << split_k_बफर.offset << ", " << split_k_बफर.size << "), m: " << m << ", n: " << n << ", k: " << k << ", stride_a: " << stride_a << ", stride_b: " << stride_b << ", stride_d: " << stride_d << ", batch_stride_a: " << batch_stride_a << ", batch_stride_b: " << batch_stride_b << ", batch_stride_d: " << batch_stride_d << ", split_k: " << split_k << ", batch: " << batch << ", ne02: " << ne02 << ", ne12: " << ne12 << ", broadcast2: " << broadcast2 << ", broadcast3: " << broadcast3 << ", padded_n: " << padded_n << ")");
    अगर (split_k == 1) {
        // a small tile can still produce more workgroups than the device allows on the n axis:
        // split the dispatch into n-chunks, shifting the B and D column windows
        स्थिर uint64_t max_n = (uint64_t)ctx->device->properties.limits.max_Compute_Work_Group_गिनती[1] * pipeline->wg_denoms[1];
        स्थिर uint64_t n_chunks = CEIL_DIV(n, max_n);
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, n_chunks * CEIL_DIV(batch, ctx->device->properties.limits.max_Compute_Work_Group_गिनती[2]));

        क्रम (uint64_t n_base = 0; n_base < n; n_base += max_n) {
            स्थिर uint32_t n_chunk = (uint32_t) std::min(n - n_base, max_n);
            GGML_ASSERT(n_base * b_col_bytes < b.size && n_base * d_col_bytes < d.size);
            GGML_ASSERT(b.size != VK_WHOLE_आकार && d.size != VK_WHOLE_आकार);
            vk_subbuffer b_chunk = b;
            vk_subbuffer d_chunk = d;
            b_chunk.offset += n_base * b_col_bytes;
            b_chunk.size   -= n_base * b_col_bytes;
            d_chunk.offset += n_base * d_col_bytes;
            d_chunk.size   -= n_base * d_col_bytes;

            uint32_t base_work_group_z = 0;
            जबतक (base_work_group_z < batch) {
                uint32_t groups_z = std::min(batch - base_work_group_z, ctx->device->properties.limits.max_Compute_Work_Group_गिनती[2]);

                स्थिर vk_mat_mat_धकेलो_constants pc = { m, n_chunk, k, stride_a, stride_b, stride_d, batch_stride_a, batch_stride_b, batch_stride_d, base_work_group_z, batch, k, ne02, ne12, broadcast2, broadcast3, uint32_t(padded_n - n_base) };
                ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { a, b_chunk, d_chunk }, pc, { m, n_chunk, groups_z });
                base_work_group_z += groups_z;
            }
        }
        वापस;
    }

    अगर (ctx->prealloc_split_k_need_sync) {
        ggml_vk_sync_buffers(ctx, subctx);
    }

    GGML_ASSERT(batch_stride_d == m * n);

    // Round the split size up to a multiple of 256 (k-quant alignment)
    uint32_t k_split = CEIL_DIV(k, split_k);
    k_split = ROUNDUP_POW2(k_split, 256);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, CEIL_DIV(batch, ctx->device->properties.limits.max_Compute_Work_Group_गिनती[2]));

    uint32_t base_work_group_z = 0;
    जबतक (base_work_group_z < batch) {
        uint32_t groups_z = std::min(batch - base_work_group_z, ctx->device->properties.limits.max_Compute_Work_Group_गिनती[2]);

        स्थिर vk_mat_mat_धकेलो_constants pc1 = { m, n, k, stride_a, stride_b, stride_d, batch_stride_a, batch_stride_b, batch_stride_d, base_work_group_z, batch, k_split, ne02, ne12, broadcast2, broadcast3, padded_n };
        // Make sure enough workgroups get assigned for split k to work
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { a, b, split_k_बफर }, pc1, { (CEIL_DIV(m, pipeline->wg_denoms[0]) * pipeline->wg_denoms[0]) * split_k, n, groups_z });
        base_work_group_z += groups_z;
    }
    ggml_vk_sync_buffers(ctx, subctx);
    स्थिर std::array<uint32_t, 2> pc2 = { (uint32_t)(m * n * batch), split_k };
    ggml_vk_dispatch_pipeline(ctx, subctx, ctx->device->pipeline_matmul_split_k_reduce, { split_k_बफर, d }, pc2, { m * n * batch, 1, 1 });
    ctx->prealloc_split_k_need_sync = true;
}

अटल bool ggml_vk_get_mul_mat_mat_f16acc(ggml_backend_vk_context * ctx, ggml_type src0_type, ggml_type src1_type, ggml_prec prec) {
    अगर (src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_BF16) वापस false;
    अगर (src1_type == GGML_TYPE_Q8_1) वापस false;
    अगर (src0_type == GGML_TYPE_F16) {
        वापस prec == GGML_PREC_DEFAULT && ctx->device->fp16 && !(ctx->device->coopmat_support && !ctx->device->coopmat_acc_f16_support);
    }
    // quant types
    अगर (ctx->device->coopmat2) {
        वापस prec == GGML_PREC_DEFAULT;
    }
    अगर (ctx->device->coopmat_support) {
        वापस ctx->device->fp16 && ctx->device->coopmat_acc_f16_support && prec == GGML_PREC_DEFAULT;
    }
    वापस ctx->device->fp16 && prec == GGML_PREC_DEFAULT;
}

अटल स्थिर std::vector<vk_matmul_pipeline_pair>* ggml_vk_get_mul_mat_mat_pipeline_map(
        ggml_backend_vk_context * ctx, ggml_type src0_type, ggml_type src1_type, ggml_prec prec, bool mul_mat_id = false) {
    bool f16acc = ggml_vk_get_mul_mat_mat_f16acc(ctx, src0_type, src1_type, prec);
    vk_matmul_pipeline_कुंजी कुंजी{src0_type, src1_type, mul_mat_id, f16acc};
    स्वतः it = ctx->device->pipeline_matmul.find(कुंजी);
    अगर (it == ctx->device->pipeline_matmul.end() || it->second.empty()) {
        // Try without f16acc
        अगर (f16acc) {
            कुंजी.f16acc = false;
            it = ctx->device->pipeline_matmul.find(कुंजी);
            अगर (it != ctx->device->pipeline_matmul.end() && !it->second.empty()) वापस &it->second;
        }
        वापस nullptr;
    }
    वापस &it->second;
}

अटल vk_pipeline ggml_vk_guess_matmul_pipeline_map(ggml_backend_vk_context * ctx,
        स्थिर std::vector<vk_matmul_pipeline_pair>& configs,
        uint32_t m, uint32_t n, bool aligned, bool mul_mat_id) {
    स्वतः& selector = mul_mat_id ? ctx->device->matmul_id_tile_selector : ctx->device->matmul_tile_selector;
    uint32_t idx = selector(m, n, 0, ctx->device->shader_core_गिनती, configs);
    अगर (idx >= configs.size()) idx = (uint32_t)configs.size() - 1;
    वापस (aligned && configs[idx].aligned) ? configs[idx].aligned : configs[idx].unaligned;
}

अटल uint32_t ggml_vk_guess_matmul_pipeline_align_map(ggml_backend_vk_context * ctx,
        स्थिर std::vector<vk_matmul_pipeline_pair>& configs,
        uint32_t m, uint32_t n, bool mul_mat_id) {
    स्वतः& selector = mul_mat_id ? ctx->device->matmul_id_tile_selector : ctx->device->matmul_tile_selector;
    uint32_t idx = selector(m, n, 0, ctx->device->shader_core_गिनती, configs);
    अगर (idx >= configs.size()) idx = (uint32_t)configs.size() - 1;
    वापस configs[idx].align;
}

अटल uint64_t ggml_vk_mul_mat_id_max_tiles(uint64_t n_experts, uint64_t n_rows, uint32_t bn) {
    // Each nonempty expert can add at most one partial tile to the combined grid.
    वापस std::min(n_rows, CEIL_DIV(n_rows, bn) + std::min(n_experts, n_rows) - 1);
}

अटल व्योम ggml_vk_matmul_id(
        ggml_backend_vk_context * ctx, vk_context& subctx, vk_pipeline& pipeline,
        vk_subbuffer&& a, vk_subbuffer&& b, vk_subbuffer&& d, vk_subbuffer&& ids, स्थिर vk_subbuffer & expert_गिनती_buf,
        uint32_t m, uint32_t n, uint32_t k, uint32_t stride_a, uint32_t stride_b, uint32_t stride_d,
        uint32_t batch_stride_a, uint32_t batch_stride_b, uint32_t batch_stride_d,
        uint32_t n_as, uint32_t nei0, uint32_t nei1, uint32_t nbi1, uint32_t ne11,
        uint32_t row_ids_offset, uint32_t max_tiles) {
    VK_LOG_DEBUG("ggml_vk_matmul_id(a: (" << a.बफर->बफर << ", " << a.offset << ", " << a.size << "), b: (" << b.बफर->बफर << ", " << b.offset << ", " << b.size << "), d: (" << d.बफर->बफर << ", " << d.offset << ", " << d.size << "), ids: (" << ids.बफर->बफर << ", " << ids.offset << ", " << ids.size << "), expert_count: (" << expert_गिनती_buf.बफर->बफर << ", " << expert_गिनती_buf.offset << ", " << expert_गिनती_buf.size << "), " <<
        "m: " << m << ", n: " << n << ", k: " << k << ", stride_a: " << stride_a << ", stride_b: " << stride_b << ", stride_d: " << stride_d << ", " <<
        "batch_stride_a: " << batch_stride_a << ", batch_stride_b: " << batch_stride_b << ", batch_stride_d: " << batch_stride_d << ", " <<
        "n_as: " << n_as << ", nei0: " << nei0 << ", nei1: " << nei1 << ", nbi1: " << nbi1 << ", ne11: " << ne11 << ")");
    स्थिर vk_mat_mat_id_धकेलो_constants pc = { m, n, k, stride_a, stride_b, stride_d, batch_stride_a, batch_stride_b, batch_stride_d,
                                              nei0, nei1, nbi1, ne11, n_as, row_ids_offset };
    स्थिर uint32_t tiles_y = std::min(max_tiles, ctx->device->properties.limits.max_Compute_Work_Group_गिनती[1]);
    स्थिर std::array<uint32_t, 3> elements = { m, tiles_y * pipeline->wg_denoms[1], uint32_t(CEIL_DIV(max_tiles, tiles_y)) };
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { a, b, d, ids, expert_गिनती_buf }, pc, elements);
}

bool ggml_vk_dim01_contiguous(स्थिर ggml_tensor * tensor) {
    वापस
        tensor->nb[0] == ggml_type_आकार(tensor->type) &&
        tensor->nb[1] == (tensor->nb[0]*tensor->ne[0])/ggml_blck_आकार(tensor->type) &&
        (tensor->ne[3] == 1 || tensor->nb[3] == tensor->nb[2]*tensor->ne[2]);
}

// Batch stride in elements of a tensor read in place.
अटल uint32_t ggml_vk_batch_stride(स्थिर ggml_tensor * tensor) {
    वापस (uint32_t)(tensor->nb[2] / ggml_type_आकार(tensor->type) * ggml_blck_आकार(tensor->type));
}

vk_pipeline ggml_vk_get_cpy_pipeline(ggml_backend_vk_context * ctx, स्थिर ggml_tensor * src, स्थिर ggml_tensor * dst, ggml_type to) {

    // Choose "contiguous copy" shader if src/dst are contiguous
    bool contig = ggml_is_contiguous(src) && (!dst || ggml_is_contiguous(dst));

    // Use optimized "transpose" shader if src dim1 is the innermost dimension.
    bool transpose = dst && src->nb[1] == ggml_type_आकार(to) && ggml_are_same_shape(dst, src);

    अगर (transpose && src->type == to) {
        अगर (ggml_type_आकार(to) == 4) {
            वापस ctx->device->pipeline_cpy_transpose_32;
        } अन्यथा अगर (ggml_type_आकार(to) == 2) {
            वापस ctx->device->pipeline_cpy_transpose_16;
        }
    }

    // Same, for a 0<->2 swap: src dim2 is the innermost dimension.
    bool transpose02 = dst && !contig && src->nb[2] == ggml_type_आकार(to) &&
                       ggml_is_contiguous(dst) && ggml_are_same_shape(dst, src);

    अगर (transpose02 && src->type == to) {
        अगर (ggml_type_आकार(to) == 4) {
            वापस ctx->device->pipeline_cpy_transpose_02_32;
        } अन्यथा अगर (ggml_type_आकार(to) == 2) {
            वापस ctx->device->pipeline_cpy_transpose_02_16;
        }
    }

    अगर (src->type == GGML_TYPE_F32 && to == GGML_TYPE_F16) {
        अगर (contig) {
            वापस ctx->device->pipeline_contig_cpy_f32_f16;
        } अन्यथा {
            वापस ctx->device->pipeline_cpy_f32_f16;
        }
    }
    अगर (src->type == GGML_TYPE_F16 && to == GGML_TYPE_F32) {
        अगर (contig) {
            वापस ctx->device->pipeline_contig_cpy_f16_f32;
        } अन्यथा {
            वापस ctx->device->pipeline_cpy_f16_f32;
        }
    }
    अगर (src->type == GGML_TYPE_F32 && to == GGML_TYPE_BF16) {
        अगर (contig) {
            वापस ctx->device->pipeline_contig_cpy_f32_bf16;
        } अन्यथा {
            वापस ctx->device->pipeline_cpy_f32_bf16;
        }
    }
    अगर (src->type == GGML_TYPE_BF16 && to == GGML_TYPE_F32) {
        अगर (contig) {
            वापस ctx->device->pipeline_contig_cpy_bf16_f32;
        } अन्यथा {
            वापस ctx->device->pipeline_cpy_bf16_f32;
        }
    }
    अगर (src->type == GGML_TYPE_BF16 && to == GGML_TYPE_F16) {
        अगर (contig) {
            वापस ctx->device->pipeline_contig_cpy_bf16_f16;
        } अन्यथा {
            वापस ctx->device->pipeline_cpy_bf16_f16;
        }
    }
    अगर (src->type == GGML_TYPE_F32 && to == GGML_TYPE_I32) {
        अगर (contig) {
            वापस ctx->device->pipeline_contig_cpy_f32_i32;
        } अन्यथा {
            वापस ctx->device->pipeline_cpy_f32_i32;
        }
    }
    अगर (src->type == GGML_TYPE_I32 && to == GGML_TYPE_F32) {
        अगर (contig) {
            वापस ctx->device->pipeline_contig_cpy_i32_f32;
        } अन्यथा {
            वापस ctx->device->pipeline_cpy_i32_f32;
        }
    }
    अगर (src->type == GGML_TYPE_F32) {
        चयन (to) {
        हाल GGML_TYPE_Q1_0:
        हाल GGML_TYPE_Q2_0:
        हाल GGML_TYPE_Q4_0:
        हाल GGML_TYPE_Q4_1:
        हाल GGML_TYPE_Q5_0:
        हाल GGML_TYPE_Q5_1:
        हाल GGML_TYPE_Q8_0:
        हाल GGML_TYPE_IQ4_NL:
            वापस ctx->device->pipeline_cpy_f32_quant[to];
        शेष:
            अवरोध;
        }
    }

    अगर (to == GGML_TYPE_F32) {
        चयन (src->type) {
        हाल GGML_TYPE_Q1_0:
        हाल GGML_TYPE_Q2_0:
        हाल GGML_TYPE_Q4_0:
        हाल GGML_TYPE_Q4_1:
        हाल GGML_TYPE_Q5_0:
        हाल GGML_TYPE_Q5_1:
        हाल GGML_TYPE_Q8_0:
        हाल GGML_TYPE_IQ4_NL:
            वापस ctx->device->pipeline_cpy_quant_f32[src->type];
        शेष:
            अवरोध;
        }
    }

    अगर (src->type == to) {
        // Copy two or four bytes at a time, depending on block size.
        // For quantized types, we scale by block size/type size. But
        // this path is also used for bf16->bf16 for example, where the
        // type size must be exactly 2 or 4.
        GGML_ASSERT(ggml_is_quantized(to) || ggml_type_आकार(src->type) == 2 || ggml_type_आकार(src->type) == 4);
        अगर ((ggml_type_आकार(src->type) % 4) == 0) {
            अगर (contig) {
                वापस ctx->device->pipeline_contig_cpy_32;
            } अन्यथा {
                वापस ctx->device->pipeline_cpy_32;
            }
        } अन्यथा {
            अगर (contig) {
                वापस ctx->device->pipeline_contig_cpy_16;
            } अन्यथा {
                वापस ctx->device->pipeline_cpy_16;
            }
        }
    }

    std::अ_त्रुटि << "Missing CPY op for types: " << ggml_type_name(src->type) << " " << ggml_type_name(to) << std::प_पूर्ण;
    GGML_ABORT("fatal error");
}

व्योम ggml_vk_cpy_to_contiguous(ggml_backend_vk_context * ctx, vk_context& subctx, vk_pipeline pipeline, स्थिर ggml_tensor * tensor, स्थिर vk_subbuffer & in, स्थिर vk_subbuffer & out) {
    VK_LOG_DEBUG("ggml_vk_cpy_to_contiguous((" << tensor << ", type=" << tensor->type << ", ne0=" << tensor->ne[0] << ", ne1=" << tensor->ne[1] << ", ne2=" << tensor->ne[2] << ", ne3=" << tensor->ne[3] << ", nb0=" << tensor->nb[0] << ", nb1=" << tensor->nb[1] << ", nb2=" << tensor->nb[2] << ", nb3=" << tensor->nb[3] << "), ";
    std::अ_त्रुटि << "buffer in size=" << in.बफर->size << ", buffer out size=" << out.बफर->size << ")");

    स्थिर uint32_t ne = ggml_nelements(tensor);
    std::array<uint32_t, 3> elements;

    अगर (ne > 262144) {
        elements = { 512, 512, CEIL_DIV(ne, 262144) };
    } अन्यथा अगर (ne > 512) {
        elements = { 512, CEIL_DIV(ne, 512), 1 };
    } अन्यथा {
        elements = { ne, 1, 1 };
    }

    vk_op_unary_धकेलो_constants pc = vk_op_unary_धकेलो_constants_init(tensor, tensor, ne);
    pc.nb10 = 1;
    pc.nb11 = (uint32_t)tensor->ne[0];
    pc.nb12 = (uint32_t)(tensor->ne[0] * tensor->ne[1]);
    pc.nb13 = (uint32_t)(tensor->ne[0] * tensor->ne[1] * tensor->ne[2]);
    init_pushconst_fastdiv(pc);
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { in, out }, pc, elements);
    ggml_vk_sync_buffers(ctx, subctx);
}

अटल व्योम ggml_vk_cpy_to_strided(
        ggml_backend_vk_context * ctx, vk_context& subctx, vk_pipeline pipeline, स्थिर ggml_tensor * tensor,
        स्थिर vk_subbuffer & in, स्थिर vk_subbuffer & out,
        uint32_t nb10, uint32_t nb11, uint32_t nb12, uint32_t nb13) {
    VK_LOG_DEBUG("ggml_vk_cpy_to_strided((" << tensor << ", type=" << tensor->type << ", ne0=" << tensor->ne[0] << ", ne1=" << tensor->ne[1] << ", ne2=" << tensor->ne[2] << ", ne3=" << tensor->ne[3] << ", nb0=" << tensor->nb[0] << ", nb1=" << tensor->nb[1] << ", nb2=" << tensor->nb[2] << ", nb3=" << tensor->nb[3] << "), ";
    std::अ_त्रुटि << "dst_nb=(" << nb10 << ", " << nb11 << ", " << nb12 << ", " << nb13 << "), buffer in size=" << in.बफर->size << ", buffer out size=" << out.बफर->size << ")");

    स्थिर uint32_t ne = ggml_nelements(tensor);
    std::array<uint32_t, 3> elements;

    अगर (ne > 262144) {
        elements = { 512, 512, CEIL_DIV(ne, 262144) };
    } अन्यथा अगर (ne > 512) {
        elements = { 512, CEIL_DIV(ne, 512), 1 };
    } अन्यथा {
        elements = { ne, 1, 1 };
    }

    vk_op_unary_धकेलो_constants pc = vk_op_unary_धकेलो_constants_init(tensor, tensor, ne);
    pc.nb10 = nb10;
    pc.nb11 = nb11;
    pc.nb12 = nb12;
    pc.nb13 = nb13;
    init_pushconst_fastdiv(pc);
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { in, out }, pc, elements);
    ggml_vk_sync_buffers(ctx, subctx);
}

vk_pipeline ggml_vk_get_quantize_pipeline(ggml_backend_vk_context * ctx, ggml_type type) {
    चयन(type) {
        हाल GGML_TYPE_Q8_1:
            वापस ctx->device->pipeline_quantize_q8_1_x4;
        शेष:
            std::अ_त्रुटि << "Missing quantize pipeline for type: " << ggml_type_name(type) << std::प_पूर्ण;
            GGML_ABORT("fatal error");
    }
}

व्योम ggml_vk_quantize_q8_1(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर vk_subbuffer & in, स्थिर vk_subbuffer & out, uint32_t ne) {
    VK_LOG_DEBUG("ggml_vk_quantize_q8_1(" << "buffer in size=" << in.बफर->size << ", buffer out size=" << out.बफर->size << ", " << ne << ")");

    vk_pipeline pipeline = ggml_vk_get_quantize_pipeline(ctx, GGML_TYPE_Q8_1);

    स्थिर uint32_t अंक_blocks = CEIL_DIV(ne, pipeline->wg_denoms[0]);
    // clamp the number of elements to the max workgroup count. The shader will iterate over the total number of blocks.
    स्थिर uint64_t max_elements = std::min<uint64_t>(uint64_t{ctx->device->properties.limits.max_Compute_Work_Group_गिनती[0]} * pipeline->wg_denoms[0], std::numeric_limits<uint32_t>::max());
    स्थिर uint32_t elements = std::min(ne, static_cast<uint32_t>(max_elements));

    स्थिर vk_quantize_q8_1_धकेलो_constants pc = {
        ne,
        अंक_blocks,
    };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { in, out }, pc, { elements, 1, 1 });
    ggml_vk_sync_buffers(ctx, subctx);
}

अटल vk_pipeline ggml_vk_get_64b_indexing_pipeline(ggml_backend_vk_context * ctx, vk_pipeline &pipeline) {
    GGML_UNUSED(ctx);
#अगर defined(VK_EXT_shader_64bit_indexing)
    vk_pipeline *ptr = &pipeline;
    जबतक (*ptr) {
        अगर ((*ptr)->is_64b_indexing) {
            वापस *ptr;
        }
        ptr = &(*ptr)->अग्र;
    }
#पूर्ण_अगर
    वापस pipeline;
}

// perf-heuristic for choosing mul_mat b type, based on shader availability
अटल ggml_type ggml_vk_mul_mat_b_type(
        ggml_backend_vk_context * ctx, ggml_type src0_type, स्थिर ggml_tensor * src1,
        ggml_type f16_type, ggml_prec prec, bool mul_mat_id, bool can_quantize_y, bool prefer_f16_b) {
    स्वतः available = [&](ggml_type b) {
        वापस ggml_vk_get_mul_mat_mat_pipeline_map(ctx, src0_type, b, prec, mul_mat_id) != nullptr;
    };
    अगर (can_quantize_y && available(GGML_TYPE_Q8_1))             वापस GGML_TYPE_Q8_1;
    अगर (prefer_f16_b   && available(f16_type))                   वापस f16_type;
    अगर (ggml_vk_dim01_contiguous(src1) && available(src1->type)) वापस src1->type;
    अगर (src1->type == GGML_TYPE_BF16 && available(GGML_TYPE_F32)) वापस GGML_TYPE_F32;
    वापस f16_type;
}

अटल व्योम ggml_vk_mul_mat_q_f16(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, स्थिर ggml_tensor * src1, ggml_tensor * dst, bool disable_split_k) {
    VK_LOG_DEBUG("ggml_vk_mul_mat_q_f16((" << src0 << ", name=" << src0->name << ", type=" << ggml_type_name(src0->type) << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::अ_त्रुटि << "), (" << src1 << ", name=" << src1->name << ", type=" << ggml_type_name(src1->type) << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::अ_त्रुटि << "), (" << dst << ", name=" << dst->name << ", type=" << ggml_type_name(dst->type) << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::अ_त्रुटि << "))");
    GGML_ASSERT(ggml_vk_dim01_contiguous(src0) || src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16);  // NOLINT
    GGML_ASSERT(ggml_vk_dim01_contiguous(src1) || src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);  // NOLINT

    स्थिर uint64_t ne00 = src0->ne[0];
    स्थिर uint64_t ne01 = src0->ne[1];
    स्थिर uint64_t ne02 = src0->ne[2];
    स्थिर uint64_t ne03 = src0->ne[3];

    स्थिर uint64_t ne10 = src1->ne[0];
    स्थिर uint64_t ne11 = src1->ne[1];
    स्थिर uint64_t ne12 = src1->ne[2];
    स्थिर uint64_t ne13 = src1->ne[3];

    स्थिर uint64_t ne21 = dst->ne[1];
    स्थिर uint32_t stride_d = dst->nb[1] / ggml_type_आकार(dst->type);
    स्थिर uint32_t stride_batch_d = stride_d*ne21;

    स्थिर uint64_t r2 = ne12 / ne02;
    स्थिर uint64_t r3 = ne13 / ne03;

    ggml_backend_vk_बफर_context * dst_buf_ctx = (ggml_backend_vk_बफर_context *)dst->बफर->context;
    ggml_backend_vk_बफर_context * src0_buf_ctx = (ggml_backend_vk_बफर_context *)src0->बफर->context;
    ggml_backend_vk_बफर_context * src1_buf_ctx = (ggml_backend_vk_बफर_context *)src1->बफर->context;

    vk_बफर d_Qx = nullptr;
    माप_प्रकार qx_buf_offset = 0;
    vk_बफर d_Qy = nullptr;
    माप_प्रकार qy_buf_offset = 0;

    bool src0_uma = false;
    bool src1_uma = false;

    अगर (ctx->device->uma) {
        ggml_vk_host_get(ctx->device, src0->तथ्य, d_Qx, qx_buf_offset);
        ggml_vk_host_get(ctx->device, src1->तथ्य, d_Qy, qy_buf_offset);
        src0_uma = d_Qx != nullptr;
        src1_uma = d_Qy != nullptr;
    }

    स्थिर bool x_non_contig = (ctx->device->coopmat2 && src0->type == GGML_TYPE_F32) ||
                              !ggml_vk_dim01_contiguous(src0);
    // If src0 is BF16, try to use a BF16 x BF16 multiply
    ggml_type f16_type = src0->type == GGML_TYPE_BF16 ? GGML_TYPE_BF16 : GGML_TYPE_F16;

    स्थिर bool is_coopmat = ctx->device->coopmat_support || ctx->device->coopmat2;
    स्थिर bool can_quantize_y = (ctx->device->integer_dot_product || ctx->device->coopmat_int_support) &&
                                src1->type == GGML_TYPE_F32 && ggml_is_contiguous(src1) && (ne11 * ne10) % 4 == 0;
    // coopmat shaders are fast enough that memory bandwidth reduction from f16 activations helps
    स्थिर bool prefer_f16_b = is_coopmat && ggml_is_quantized(src0->type) && src1->type == GGML_TYPE_F32;

    स्थिर ggml_type y_kernel_type = ggml_vk_mul_mat_b_type(ctx, src0->type, src1, f16_type,
                                        (ggml_prec)dst->op_params[0], false, can_quantize_y, prefer_f16_b);

    स्थिर bool quantize_y       = y_kernel_type == GGML_TYPE_Q8_1;
    स्थिर bool y_f32_kernel     = y_kernel_type == GGML_TYPE_F32;
    स्थिर bool y_needs_reformat = !quantize_y && (!ggml_vk_dim01_contiguous(src1) || y_kernel_type != src1->type);

    स्थिर std::vector<vk_matmul_pipeline_pair>* mmp_map =
        ggml_vk_get_mul_mat_mat_pipeline_map(ctx, src0->type, y_kernel_type, (ggml_prec)dst->op_params[0]);

    स्थिर bool qx_needs_dequant = mmp_map == nullptr || x_non_contig;
    स्थिर bool qy_needs_dequant = y_needs_reformat;

    अगर (qx_needs_dequant) {
        // dequant src0 to f16
        mmp_map = ggml_vk_get_mul_mat_mat_pipeline_map(ctx, f16_type, y_kernel_type, (ggml_prec)dst->op_params[0]);
    }

    GGML_ASSERT(mmp_map != nullptr);

    स्थिर uint32_t kpad = quantize_y ? 0 : ggml_vk_align_आकार(ne10, ggml_vk_guess_matmul_pipeline_align_map(ctx, *mmp_map, ne01, ne11, false));
    स्थिर bool aligned = !quantize_y && ne10 == kpad && ne01 > 8 && ne11 > 8;

    vk_pipeline pipeline = ggml_vk_guess_matmul_pipeline_map(ctx, *mmp_map, ne01, ne11, aligned, false);

    अगर (ggml_nbytes(src0) > ctx->device->properties.limits.max_Storage_बफर_Range) {
        pipeline = ggml_vk_get_64b_indexing_pipeline(ctx, pipeline);
    }

    // Reserve extra storage in the N dimension for the Y matrix, so we can avoid bounds-checking
    uint32_t padded_n = qy_needs_dequant ? ROUNDUP_POW2(ne11, pipeline->wg_denoms[1]) : ne11;
    स्थिर uint64_t x_ne = ggml_nelements(src0);
    // 128 elements per Q8_1 x4 block
    स्थिर uint64_t y_ne = padded_n * ne10 * ne12 * ne13;
    स्थिर uint64_t d_ne = ggml_nelements(dst);

    स्थिर uint32_t split_k = ggml_vk_guess_split_k(ctx, ne01, ne11, ne10, disable_split_k, pipeline);

    स्थिर uint64_t qx_sz = ggml_type_आकार(src0->type) * x_ne / ggml_blck_आकार(src0->type);
    स्थिर uint64_t qy_sz = ggml_type_आकार(src1->type) * y_ne / ggml_blck_आकार(src1->type);
    स्थिर uint64_t x_sz = !qx_needs_dequant ? qx_sz : माप(ggml_fp16_t) * x_ne;
    स्थिर uint64_t y_sz = quantize_y ? (ggml_vk_align_आकार(y_ne, 128) * ggml_type_आकार(GGML_TYPE_Q8_1) / ggml_blck_आकार(GGML_TYPE_Q8_1)) : (y_f32_kernel ? माप(भग्न) * y_ne : माप(ggml_fp16_t) * y_ne);
    स्थिर uint64_t d_sz = माप(भग्न) * d_ne;

    vk_pipeline to_fp16_vk_0 = nullptr;
    vk_pipeline to_fp16_vk_1 = nullptr;
    vk_pipeline to_q8_1 = nullptr;

    अगर (x_non_contig) {
        to_fp16_vk_0 = ggml_vk_get_cpy_pipeline(ctx, src0, nullptr, f16_type);
    } अन्यथा {
        to_fp16_vk_0 = ggml_vk_get_to_fp16(ctx, src0->type);
    }
    अगर (y_needs_reformat) {
        to_fp16_vk_1 = ggml_vk_get_cpy_pipeline(ctx, src1, nullptr, y_kernel_type);
    } अन्यथा {
        to_fp16_vk_1 = ggml_vk_get_to_fp16(ctx, src1->type);
    }
    GGML_ASSERT(!qx_needs_dequant || to_fp16_vk_0 != nullptr);  // NOLINT
    GGML_ASSERT(!qy_needs_dequant || to_fp16_vk_1 != nullptr);  // NOLINT

    अगर (quantize_y) {
        to_q8_1 = ggml_vk_get_quantize_pipeline(ctx, GGML_TYPE_Q8_1);
    }

    {
        स्थिर uint64_t split_k_आकार = split_k > 1 ? d_sz * split_k : 0;
        अगर (
                (qx_needs_dequant && x_sz > ctx->device->properties.limits.max_Storage_बफर_Range) ||
                (qy_needs_dequant && y_sz > ctx->device->properties.limits.max_Storage_बफर_Range) ||
                (split_k > 1 && split_k_आकार > ctx->device->properties.limits.max_Storage_बफर_Range)) {
            GGML_ABORT("Requested preallocation size is too large");
        }
        अगर (qx_needs_dequant && ctx->prealloc_आकार_x < x_sz) {
            ctx->prealloc_आकार_x = x_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        अगर ((qy_needs_dequant || quantize_y) && ctx->prealloc_आकार_y < y_sz) {
            ctx->prealloc_आकार_y = y_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        अगर (split_k > 1 && ctx->prealloc_आकार_split_k < split_k_आकार) {
            ctx->prealloc_आकार_split_k = split_k_आकार;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }

        // Request descriptor sets
        अगर (qx_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, to_fp16_vk_0, 1);
        }
        अगर (qy_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, to_fp16_vk_1, 1);
        }
        अगर (quantize_y) {
            ggml_pipeline_request_descriptor_sets(ctx, to_q8_1, 1);
        }
        अगर (split_k > 1) {
            ggml_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_matmul_split_k_reduce, 1);
        }
    }

    vk_बफर d_D = dst_buf_ctx->dev_बफर;
    स्थिर uint64_t d_buf_offset = vk_tensor_offset(dst) + dst->view_offs;
    GGML_ASSERT(d_D != nullptr);
    GGML_ASSERT(d_D->size >= d_buf_offset + d_sz);
    vk_बफर d_X;
    uint64_t x_buf_offset = 0;
    vk_बफर d_Y;
    uint64_t y_buf_offset = 0;
    अगर (!src0_uma) {
        d_Qx = src0_buf_ctx->dev_बफर;
        qx_buf_offset = vk_tensor_offset(src0) + src0->view_offs;
        GGML_ASSERT(d_Qx != nullptr);
    }
    अगर (!src1_uma) {
        d_Qy = src1_buf_ctx->dev_बफर;
        qy_buf_offset = vk_tensor_offset(src1) + src1->view_offs;
        GGML_ASSERT(d_Qy != nullptr);
    }
    अगर (qx_needs_dequant) {
        d_X = ctx->prealloc_x;
        GGML_ASSERT(d_X->size >= x_sz);
    } अन्यथा {
        d_X = d_Qx;
        x_buf_offset = qx_buf_offset;
        GGML_ASSERT(qx_sz == x_sz);
    }
    अगर (qy_needs_dequant) {
        d_Y = ctx->prealloc_y;
        GGML_ASSERT(d_Y->size >= y_sz);
    } अन्यथा अगर (quantize_y) {
        d_Y = ctx->prealloc_y;
        GGML_ASSERT(d_Y->size >= CEIL_DIV(y_sz, 144) * 144);
    } अन्यथा {
        d_Y = d_Qy;
        y_buf_offset = qy_buf_offset;
        GGML_ASSERT(qy_sz == y_sz);
    }

    अगर (x_non_contig || qx_needs_dequant) {
        अगर (ctx->prealloc_x_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }
    }

    अगर (x_non_contig) {
        ggml_vk_cpy_to_contiguous(ctx, subctx, to_fp16_vk_0, src0, ggml_vk_subbuffer(ctx, d_Qx, qx_buf_offset), ggml_vk_subbuffer(ctx, d_X, 0));
    } अन्यथा अगर (qx_needs_dequant) {
        स्थिर std::vector<uint32_t> pc = { (uint32_t)ne01, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)(ggml_nelements(src0)) };
        ggml_vk_dispatch_pipeline(ctx, subctx, to_fp16_vk_0, { vk_subbuffer{ d_Qx, qx_buf_offset, qx_sz }, vk_subbuffer{ d_X, 0, x_sz } }, pc, { (uint32_t)(x_ne), 1, 1});
        ggml_vk_sync_buffers(ctx, subctx);
    }
    अगर (y_needs_reformat) {
        अगर (ctx->prealloc_y_last_pipeline_used != to_fp16_vk_1.लो() ||
            ctx->prealloc_y_last_tensor_used != src1 ||
            ctx->prealloc_y_last_k_padded) {
            अगर (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ggml_vk_cpy_to_contiguous(ctx, subctx, to_fp16_vk_1, src1, ggml_vk_subbuffer(ctx, d_Qy, qy_buf_offset), ggml_vk_subbuffer(ctx, d_Y, 0));
            ctx->prealloc_y_last_pipeline_used = to_fp16_vk_1.लो();
            ctx->prealloc_y_last_tensor_used = src1;
            ctx->prealloc_y_last_k_padded = false;
        }
    }
    अगर (quantize_y) {
        अगर (ctx->prealloc_y_last_pipeline_used != to_q8_1.लो() ||
            ctx->prealloc_y_last_tensor_used != src1 ||
            ctx->prealloc_y_last_k_padded) {
            अगर (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ggml_vk_quantize_q8_1(ctx, subctx, ggml_vk_subbuffer(ctx, d_Qy, qy_buf_offset), ggml_vk_subbuffer(ctx, d_Y, 0), y_ne);
            ctx->prealloc_y_last_pipeline_used = to_q8_1.लो();
            ctx->prealloc_y_last_tensor_used = src1;
            ctx->prealloc_y_last_k_padded = false;
        }
    }

    // The partial tiles of a quantized A rely on the bound range to read zeros past the last row,
    // so the range stays exact: the strided extent of a tensor read in place, the staged size otherwise.
    स्थिर uint64_t x_range = qx_needs_dequant ? x_sz : ggml_nbytes(src0);
    स्थिर uint64_t y_range = (qy_needs_dequant || quantize_y) ? y_sz : ggml_nbytes(src1);

    uint32_t stride_batch_x = qx_needs_dequant ? ne00*ne01 : ggml_vk_batch_stride(src0);
    uint32_t stride_batch_y = (qy_needs_dequant || quantize_y) ? ne10*ne11 : ggml_vk_batch_stride(src1);

    अगर (!ggml_vk_dim01_contiguous(src0) && !qx_needs_dequant) {
        stride_batch_x = src0->nb[0] / ggml_type_आकार(src0->type);
    }

    अगर (!ggml_vk_dim01_contiguous(src1) && !qy_needs_dequant && !quantize_y) {
        stride_batch_y = src1->nb[0] / ggml_type_आकार(src1->type);
    }

    // column stride in bytes of the B buffer: raw src1 keeps its own stride, staging buffers are tightly packed
    स्थिर uint64_t b_col_bytes = (y_needs_reformat || qy_needs_dequant || quantize_y)
        ? (quantize_y ? (uint64_t)(ne10 / ggml_blck_आकार(GGML_TYPE_Q8_1)) * ggml_type_आकार(GGML_TYPE_Q8_1) : (uint64_t)ne10 * माप(ggml_fp16_t))
        : (uint64_t)src1->nb[1];

    // compute
    ggml_vk_matmul(
        ctx, subctx, pipeline,
        { d_X, x_buf_offset, x_range }, { d_Y, y_buf_offset, y_range },
        ggml_vk_subbuffer(ctx, d_D, d_buf_offset), { ctx->prealloc_split_k, 0, d_sz * split_k },
        ne01, ne11, ne10,
        ne10, ne10, stride_d, stride_batch_x, stride_batch_y, stride_batch_d,
        split_k, ne12*ne13, ne02, ne12, r2, r3, padded_n,
        b_col_bytes, (uint64_t)dst->nb[1]
    );  // NOLINT

    अगर (x_non_contig || qx_needs_dequant) {
        ctx->prealloc_x_need_sync = true;
    }
    अगर (y_needs_reformat || quantize_y) {
        ctx->prealloc_y_need_sync = true;
    }
}

अटल bool ggml_vk_should_use_mmvq(स्थिर vk_device& device, uint32_t m, uint32_t n, uint32_t k, ggml_type src0_type) {
    अगर (device->mmvq_mode == 1) {
        वापस true;
    } अन्यथा अगर (device->mmvq_mode == -1) {
        वापस false;
    }

    // q6_k only has 2-byte alignment which makes it somewhat problematic,
    // using MMVQ is only a win on Intel.
    bool mmvq_q6 = device->vendor_id == VK_VENDOR_ID_INTEL;
    अगर (src0_type == GGML_TYPE_Q6_K && !mmvq_q6) {
        वापस false;
    }

    // MMVQ is generally good for batches
    अगर (n > 1) {
        वापस true;
    }

    // Quantization overhead is not worth it for small k
    चयन (device->vendor_id) {
    हाल VK_VENDOR_ID_NVIDIA:
        अगर (src0_type == GGML_TYPE_Q2_0 || src0_type == GGML_TYPE_Q2_K || src0_type == GGML_TYPE_Q3_K || src0_type == GGML_TYPE_IQ1_S || src0_type == GGML_TYPE_IQ1_M) {
            वापस true;
        }

        अगर (k <= 4096) {
            वापस false;
        }

        चयन (src0_type) {
        हाल GGML_TYPE_MXFP4:
        हाल GGML_TYPE_Q8_0:
            वापस device->architecture == vk_device_architecture::NVIDIA_PRE_TURING;
        शेष:
            वापस true;
        }
    हाल VK_VENDOR_ID_AMD:
        अगर (k < 2048) {
            वापस false;
        }

        चयन (src0_type) {
        हाल GGML_TYPE_Q8_0:
            वापस device->architecture == vk_device_architecture::AMD_GCN;
        शेष:
            वापस true;
        }
    हाल VK_VENDOR_ID_INTEL:
        अगर (device->architecture == vk_device_architecture::INTEL_XE2) {
            अगर (src0_type == GGML_TYPE_Q2_0 || src0_type == GGML_TYPE_Q2_K || src0_type == GGML_TYPE_Q3_K || src0_type == GGML_TYPE_Q6_K) {
                वापस true;
            }
        }

        अगर (device->driver_id == vk::DriverId::eIntelProprietaryWindows) {
            // Intel Windows proprietary driver MMVQ performance for !Q2/Q3/Q6 is worse than fp16,
            // see https://github.com/ggml-org/llama.cpp/issues/17628 and
            // https://github.com/ggml-org/llama.cpp/pull/23056
            वापस false;
        }

        अगर (k < 2048) {
            वापस false;
        }

        चयन (src0_type) {
        // From tests on A770 Linux, may need more tuning
        हाल GGML_TYPE_Q4_0:
        हाल GGML_TYPE_Q5_1:
        हाल GGML_TYPE_IQ4_XS:
            वापस false;
        शेष:
            वापस true;
        }
    हाल VK_VENDOR_ID_QUALCOMM:
        वापस false;
    शेष:
        वापस true;
    }

    GGML_UNUSED(m);
}

अटल व्योम ggml_vk_mul_mat_vec_q_f16(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर काष्ठा ggml_cgraph * cgraph, पूर्णांक गाँठ_idx, bool swap_inputs = false) {
    ggml_tensor * dst = cgraph->nodes[गाँठ_idx];
    स्थिर ggml_tensor * src0 = dst->src[swap_inputs ? 1 : 0];
    स्थिर ggml_tensor * src1 = dst->src[swap_inputs ? 0 : 1];

    VK_LOG_DEBUG("ggml_vk_mul_mat_vec_q_f16((" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::अ_त्रुटि << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::अ_त्रुटि << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::अ_त्रुटि << ")),)");
    GGML_ASSERT(ggml_vk_dim01_contiguous(src0) || src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16);  // NOLINT
    GGML_ASSERT(ggml_vk_dim01_contiguous(src1) || src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);  // NOLINT

    स्थिर uint64_t ne00 = src0->ne[0];
    स्थिर uint64_t ne01 = src0->ne[1];
    स्थिर uint64_t ne02 = src0->ne[2];
    स्थिर uint64_t ne03 = src0->ne[3];

    स्थिर uint64_t ne10 = src1->ne[0];
    स्थिर uint64_t ne11 = src1->ne[1];
    स्थिर uint64_t ne12 = src1->ne[2];
    स्थिर uint64_t ne13 = src1->ne[3];

    स्थिर uint64_t ne20 = dst->ne[swap_inputs ? 1 : 0];
    स्थिर uint64_t ne21 = dst->ne[swap_inputs ? 0 : 1];
    // const uint64_t ne22 = dst->ne[2];
    // const uint64_t ne23 = dst->ne[3];

    स्थिर uint64_t r2 = ne12 / ne02;
    स्थिर uint64_t r3 = ne13 / ne03;

    // batch_n indicates that we need to compute a few vector results, and this assumes
    // ne12 and ne13 are 1. It overloads the batch_strides to hold the row strides.
    GGML_ASSERT(ne11 == 1 || ne12 * ne13 == 1);
    bool batch_n = ne11 > 1;

    स्थिर bool x_non_contig = !ggml_vk_dim01_contiguous(src0);

    // BF16 src1 has no mul_mat_vec B-type; widen to F32
    स्थिर bool widen_bf16 = src1->type == GGML_TYPE_BF16;
    स्थिर bool y_non_contig = !ggml_vk_dim01_contiguous(src1) || widen_bf16;

    स्थिर ggml_type y_kernel_type = widen_bf16 ? GGML_TYPE_F32 : src1->type;
    स्थिर bool f16_f32_kernel = y_kernel_type == GGML_TYPE_F32;
    bool quantize_y = ctx->device->integer_dot_product && src1->type == GGML_TYPE_F32 && ggml_is_contiguous(src1) && !y_non_contig && (ne11 * ne10) % 4 == 0 && ggml_vk_should_use_mmvq(ctx->device, ne01, ne11, ne10, src0->type);

    vk_pipeline to_fp16_vk_0 = nullptr;
    vk_pipeline to_fp16_vk_1 = nullptr;
    अगर (x_non_contig) {
        to_fp16_vk_0 = ggml_vk_get_cpy_pipeline(ctx, src0, nullptr, src0->type);
    }
    अगर (y_non_contig) {
        to_fp16_vk_1 = ggml_vk_get_cpy_pipeline(ctx, src1, nullptr, y_kernel_type);
    } अन्यथा {
        to_fp16_vk_1 = ggml_vk_get_to_fp16(ctx, src1->type);
    }

    // Check for mmq first
    vk_pipeline dmmv = quantize_y ? ggml_vk_get_dequantize_mul_mat_vec(ctx, src0->type, GGML_TYPE_Q8_1, ne11, ne20, ne00) : nullptr;
    vk_pipeline to_q8_1 = nullptr;

    अगर (dmmv == nullptr) {
        // Fall back to f16 dequant mul mat
        dmmv = ggml_vk_get_dequantize_mul_mat_vec(ctx, src0->type, y_kernel_type, ne11, ne20, ne00);
        quantize_y = false;
    }

    अगर (quantize_y) {
        to_q8_1 = ggml_vk_get_quantize_pipeline(ctx, GGML_TYPE_Q8_1);
    }

    अगर (ggml_nbytes(src0) > ctx->device->properties.limits.max_Storage_बफर_Range) {
        dmmv = ggml_vk_get_64b_indexing_pipeline(ctx, dmmv);
    }

    स्थिर bool qx_needs_dequant = x_non_contig;
    स्थिर bool qy_needs_dequant = !quantize_y && ((src1->type != GGML_TYPE_F16 && !f16_f32_kernel) || y_non_contig);

    // Not implemented
    GGML_ASSERT(y_non_contig || !qy_needs_dequant);  // NOLINT

    GGML_ASSERT(!qx_needs_dequant || to_fp16_vk_0 != nullptr);  // NOLINT
    GGML_ASSERT(!qy_needs_dequant || to_fp16_vk_1 != nullptr);  // NOLINT
    GGML_ASSERT(dmmv != nullptr);

    स्थिर uint64_t x_ne = ggml_nelements(src0);
    स्थिर uint64_t y_ne = ggml_nelements(src1);

    स्थिर uint64_t qx_sz = ggml_vk_align_आकार(ggml_type_आकार(src0->type) * x_ne / ggml_blck_आकार(src0->type), ctx->device->properties.limits.min_Storage_बफर_Offset_Alignment);
    स्थिर uint64_t x_sz = x_non_contig ? ggml_vk_align_आकार(ggml_type_आकार(src0->type) * x_ne, ctx->device->properties.limits.min_Storage_बफर_Offset_Alignment) : qx_sz;
    स्थिर uint64_t y_sz = quantize_y ? (ggml_vk_align_आकार(y_ne, 128) * ggml_type_आकार(GGML_TYPE_Q8_1) / ggml_blck_आकार(GGML_TYPE_Q8_1)) :
                         (f16_f32_kernel ? माप(भग्न) * y_ne : माप(ggml_fp16_t) * y_ne);

    {
        अगर (
                (qx_needs_dequant && x_sz > ctx->device->properties.limits.max_Storage_बफर_Range) ||
                (qy_needs_dequant && y_sz > ctx->device->properties.limits.max_Storage_बफर_Range)) {
            GGML_ABORT("Requested preallocation size is too large");
        }
        अगर (qx_needs_dequant && ctx->prealloc_आकार_x < x_sz) {
            ctx->prealloc_आकार_x = x_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        अगर ((qy_needs_dequant || quantize_y) && ctx->prealloc_आकार_y < y_sz) {
            ctx->prealloc_आकार_y = y_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }

        // Request descriptor sets
        अगर (qx_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, to_fp16_vk_0, 1);
        }
        अगर (qy_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, to_fp16_vk_1, 1);
        }
        अगर (quantize_y) {
            ggml_pipeline_request_descriptor_sets(ctx, to_q8_1, 1);
        }
    }

    vk_subbuffer d_D = ggml_vk_tensor_subbuffer(ctx, cgraph->nodes[गाँठ_idx + ctx->अंक_additional_fused_ops]);
    vk_subbuffer d_Qx = ggml_vk_tensor_subbuffer(ctx, src0);
    vk_subbuffer d_Qy = ggml_vk_tensor_subbuffer(ctx, src1);
    vk_subbuffer d_X, d_Y;

    अगर (qx_needs_dequant) {
        d_X = { ctx->prealloc_x, 0, ctx->prealloc_x->size };
    } अन्यथा {
        d_X = d_Qx;
        GGML_ASSERT(qx_sz == x_sz);
    }
    अगर (qy_needs_dequant || quantize_y) {
        d_Y = { ctx->prealloc_y, 0, ctx->prealloc_y->size };
    } अन्यथा {
        d_Y = d_Qy;
    }

    अगर (x_non_contig) {
        अगर (ctx->prealloc_x_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }

        GGML_ASSERT(x_sz == ggml_vk_align_आकार(ggml_type_आकार(src0->type) * x_ne, ctx->device->properties.limits.min_Storage_बफर_Offset_Alignment));
        ggml_vk_cpy_to_contiguous(ctx, subctx, to_fp16_vk_0, src0, d_Qx, d_X);
    }
    अगर (y_non_contig) {
        GGML_ASSERT(y_sz == ggml_type_आकार(y_kernel_type) * y_ne);
        अगर (ctx->prealloc_y_last_pipeline_used != to_fp16_vk_1.लो() ||
            ctx->prealloc_y_last_tensor_used != src1 ||
            ctx->prealloc_y_last_k_padded) {
            अगर (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ggml_vk_cpy_to_contiguous(ctx, subctx, to_fp16_vk_1, src1, d_Qy, d_Y);
            ctx->prealloc_y_last_pipeline_used = to_fp16_vk_1.लो();
            ctx->prealloc_y_last_tensor_used = src1;
            ctx->prealloc_y_last_k_padded = false;
        }
    }
    अगर (quantize_y) {
        अगर (ctx->prealloc_y_last_pipeline_used != to_q8_1.लो() ||
            ctx->prealloc_y_last_tensor_used != src1 ||
            ctx->prealloc_y_last_k_padded) {
            अगर (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ggml_vk_quantize_q8_1(ctx, subctx, d_Qy, d_Y, y_ne);
            ctx->prealloc_y_last_pipeline_used = to_q8_1.लो();
            ctx->prealloc_y_last_tensor_used = src1;
            ctx->prealloc_y_last_k_padded = false;
        }
    }

    // For batch_n, the A matrix is the same for each batch, and B/D use the row stride as the batch stride
    uint32_t stride_batch_x = batch_n ? 0 : (qx_needs_dequant ? ne00*ne01 : ggml_vk_batch_stride(src0));
    uint32_t stride_batch_y = batch_n ? ne10 : ((qy_needs_dequant || quantize_y) ? ne10*ne11 : ggml_vk_batch_stride(src1));
    uint32_t stride_batch_d = batch_n ? ne20 : (ne20*ne21);

    अगर (!ggml_vk_dim01_contiguous(src0) && !qx_needs_dequant) {
        stride_batch_x = src0->nb[0] / ggml_type_आकार(src0->type);
    }

    अगर (!ggml_vk_dim01_contiguous(src1) && !qy_needs_dequant) {
        stride_batch_y = src1->nb[0] / ggml_type_आकार(src1->type);
    }

    स्थिर uint32_t max_groups_x = ctx->device->properties.limits.max_Compute_Work_Group_गिनती[0];

    uint32_t groups_x = ne01;
    uint32_t groups_z = 1;

    अगर (ne01 > max_groups_x) {
        groups_z = 64;
        groups_x = CEIL_DIV(groups_x, groups_z);
    }

    uint32_t fusion_flags = 0;

    vk_subbuffer d_F0 = d_D;
    अगर (ctx->अंक_additional_fused_ops > 0) {
        स्थिर ggml_tensor * add = cgraph->nodes[गाँठ_idx + 1];
        स्थिर ggml_tensor * bias = add->src[0] == dst ? add->src[1] : add->src[0];

        d_F0 = ggml_vk_tensor_subbuffer(ctx, bias);
        fusion_flags |= MAT_VEC_FUSION_FLAGS_BIAS0;
    }

    vk_subbuffer d_F1 = d_D;
    अगर (ctx->अंक_additional_fused_ops == 2) {
        स्थिर ggml_tensor * add = cgraph->nodes[गाँठ_idx + 2];
        स्थिर ggml_tensor * bias = add->src[0] == cgraph->nodes[गाँठ_idx + 1] ? add->src[1] : add->src[0];

        d_F1 = ggml_vk_tensor_subbuffer(ctx, bias);
        fusion_flags |= MAT_VEC_FUSION_FLAGS_BIAS1;
    }

    ggml_pipeline_request_descriptor_sets(ctx, dmmv, CEIL_DIV(ne12 * ne13, ctx->device->properties.limits.max_Compute_Work_Group_गिनती[1]));

    uint32_t base_work_group_y = 0;
    जबतक (base_work_group_y < ne12 * ne13) {

        uint32_t groups_y = std::min((uint32_t)(ne12 * ne13) - base_work_group_y, ctx->device->properties.limits.max_Compute_Work_Group_गिनती[1]);
        स्थिर vk_mat_vec_धकेलो_constants pc = {
            (uint32_t)ne00, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)ne01,
            stride_batch_x, stride_batch_y, stride_batch_d,
            fusion_flags, base_work_group_y,
            (uint32_t)ne02, (uint32_t)ne12, (uint32_t)r2, (uint32_t)r3,
        };
        ggml_vk_dispatch_pipeline(ctx, subctx, dmmv,
                                  {
                                    d_X,
                                    d_Y,
                                    d_D,
                                    d_F0,
                                    d_F1,
                                  },
                                  pc, { groups_x, groups_y, groups_z });
        base_work_group_y += groups_y;
    }

    अगर (x_non_contig) {
        ctx->prealloc_x_need_sync = true;
    }
    अगर (y_non_contig || quantize_y) {
        ctx->prealloc_y_need_sync = true;
    }
}

अटल व्योम ggml_vk_mul_mat_vec_p021_f16_f32(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर काष्ठा ggml_cgraph * cgraph, पूर्णांक गाँठ_idx) {
    ggml_tensor * dst = cgraph->nodes[गाँठ_idx];
    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];
    VK_LOG_DEBUG("ggml_vk_mul_mat_p021_f16_f32(" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::अ_त्रुटि << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::अ_त्रुटि << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::अ_त्रुटि << "))");
    GGML_ASSERT(ggml_is_permuted(src0) && ggml_is_permuted(src1));
    GGML_ASSERT(src0->nb[0] <= src0->nb[1] && src0->nb[2] <= src0->nb[3]);  // NOLINT
    GGML_ASSERT(src1->nb[0] <= src1->nb[1] && src1->nb[2] <= src1->nb[3]);  // NOLINT
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    स्थिर uint64_t ne00 = src0->ne[0];
    स्थिर uint64_t ne01 = src0->ne[1];
    स्थिर uint64_t ne02 = src0->ne[2];
    // const uint64_t ne03 = src0->ne[3];

    //const uint64_t ne10 = src1->ne[0];
    स्थिर uint64_t ne11 = src1->ne[1];
    स्थिर uint64_t ne12 = src1->ne[2];
    // const uint64_t ne13 = src1->ne[3];

    GGML_ASSERT(ne11 == 1);

    // With grouped query attention there are > 1 Q matrices per K, V matrix.
    uint32_t gqa_ratio = (uint32_t)ne12 / (uint32_t)ne02;
    अगर (gqa_ratio > 8 || gqa_ratio == 0 || ne12 != ne02 * gqa_ratio) {
        gqa_ratio = 1;
    }

    vk_pipeline pipeline = ctx->device->pipeline_mul_mat_vec_p021_f16_f32[gqa_ratio - 1];

    अगर (ggml_nbytes(src0) > ctx->device->properties.limits.max_Storage_बफर_Range) {
        pipeline = ggml_vk_get_64b_indexing_pipeline(ctx, pipeline);
    }

    {
        // Request descriptor sets
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    }

    vk_subbuffer d_D = ggml_vk_tensor_subbuffer(ctx, cgraph->nodes[गाँठ_idx + ctx->अंक_additional_fused_ops], true);
    vk_subbuffer d_Qx = ggml_vk_tensor_subbuffer(ctx, src0);
    vk_subbuffer d_Qy = ggml_vk_tensor_subbuffer(ctx, src1, true);

    vk_subbuffer d_F0 = d_D;

    uint32_t fusion_flags = 0;

    अगर (ctx->अंक_additional_fused_ops > 0) {
        स्थिर ggml_tensor * add = cgraph->nodes[गाँठ_idx + 1];
        स्थिर ggml_tensor * bias = add->src[0] == dst ? add->src[1] : add->src[0];

        d_F0 = ggml_vk_tensor_subbuffer(ctx, bias);
        fusion_flags |= MAT_VEC_FUSION_FLAGS_BIAS0;
    }

    vk_subbuffer d_F1 = d_D;
    अगर (ctx->अंक_additional_fused_ops > 1) {
        स्थिर ggml_tensor * bias = cgraph->nodes[गाँठ_idx + 2]->src[1];

        d_F1 = ggml_vk_tensor_subbuffer(ctx, bias);
        fusion_flags |= MAT_VEC_FUSION_FLAGS_BIAS1;
    }

    // compute

    vk_mat_vec_p021_धकेलो_constants pc = {
        (uint32_t)ne00, (uint32_t)ne01, (uint32_t)ne02, (uint32_t)ne12,
        0, 0, fusion_flags
    };

    init_pushconst_tensor_offsets(ctx, pc, src0, src1, nullptr, nullptr, cgraph->nodes[गाँठ_idx + ctx->अंक_additional_fused_ops]);

    uint32_t workgroups_z = (uint32_t)ne12;
    // When gqa_ratio > 1, each invocation does multiple rows and we can launch fewer workgroups
    अगर (gqa_ratio > 1) {
        workgroups_z /= gqa_ratio;
    }

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        {
            d_Qx,
            d_Qy,
            d_D,
            d_F0,
            d_F1,
        }, pc, { 1, (uint32_t)ne01, workgroups_z });
}

अटल व्योम ggml_vk_mul_mat_vec_nc_f16_f32(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर काष्ठा ggml_cgraph * cgraph, पूर्णांक गाँठ_idx) {
    ggml_tensor * dst = cgraph->nodes[गाँठ_idx];
    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];
    VK_LOG_DEBUG("ggml_vk_mul_mat_nc_f16_f32((" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::अ_त्रुटि << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::अ_त्रुटि << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::अ_त्रुटि << "))");
    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));
    GGML_ASSERT(!ggml_is_permuted(src0));
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    स्थिर uint64_t ne00 = src0->ne[0];
    स्थिर uint64_t ne01 = src0->ne[1];
    स्थिर uint64_t ne02 = src0->ne[2];
    स्थिर uint64_t ne03 = src0->ne[3];

    स्थिर uint64_t nb01 = src0->nb[1];
    स्थिर uint64_t nb02 = src0->nb[2];

    स्थिर uint64_t nb12 = src1->nb[2];

    // const uint64_t ne10 = src1->ne[0];
    स्थिर uint64_t ne11 = src1->ne[1];
    स्थिर uint64_t ne12 = src1->ne[2];
    // const uint64_t ne13 = src1->ne[3];

    स्थिर uint32_t nb03 = (uint32_t)(src0->nb[3] / माप(ggml_fp16_t));
    स्थिर uint32_t nb13 = (uint32_t)(src1->nb[3] / माप(भग्न));
    स्थिर uint32_t nb23 = (uint32_t)(dst->nb[3] / माप(भग्न));

    GGML_ASSERT(ne11 == 1);
    GGML_ASSERT(src0->ne[3] == src1->ne[3]); // checked in supports_op

    स्थिर uint32_t row_stride_x = nb01 / माप(ggml_fp16_t);
    स्थिर uint32_t channel_stride_x = nb02 / माप(ggml_fp16_t);
    स्थिर uint32_t channel_stride_y = nb12 / माप(भग्न);

    vk_pipeline pipeline = ctx->device->pipeline_mul_mat_vec_nc_f16_f32;
    अगर (ggml_nbytes(src0) > ctx->device->properties.limits.max_Storage_बफर_Range) {
        pipeline = ggml_vk_get_64b_indexing_pipeline(ctx, pipeline);
    }

    {
        // Request descriptor sets
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    }

    vk_subbuffer d_D = ggml_vk_tensor_subbuffer(ctx, cgraph->nodes[गाँठ_idx + ctx->अंक_additional_fused_ops], true);
    vk_subbuffer d_Qx = ggml_vk_tensor_subbuffer(ctx, src0);
    vk_subbuffer d_Qy = ggml_vk_tensor_subbuffer(ctx, src1, true);
    vk_subbuffer d_F0 = d_D;

    uint32_t fusion_flags = 0;

    अगर (ctx->अंक_additional_fused_ops > 0) {
        स्थिर ggml_tensor * add = cgraph->nodes[गाँठ_idx + 1];
        स्थिर ggml_tensor * bias = add->src[0] == dst ? add->src[1] : add->src[0];

        d_F0 = ggml_vk_tensor_subbuffer(ctx, bias);
        fusion_flags |= MAT_VEC_FUSION_FLAGS_BIAS0;
    }

    vk_subbuffer d_F1 = d_D;
    अगर (ctx->अंक_additional_fused_ops > 1) {
        स्थिर ggml_tensor * bias = cgraph->nodes[गाँठ_idx + 2]->src[1];

        d_F1 = ggml_vk_tensor_subbuffer(ctx, bias);
        fusion_flags |= MAT_VEC_FUSION_FLAGS_BIAS1;
    }

    // compute
    vk_mat_vec_nc_धकेलो_constants pc = {
        (uint32_t)ne00, (uint32_t)ne01,
        row_stride_x, channel_stride_x, channel_stride_y,
        (uint32_t)(ne12 / ne02), (uint32_t)ne12,
        0, 0,
        nb03, nb13, nb23, fusion_flags
    };

    init_pushconst_tensor_offsets(ctx, pc, src0, src1, nullptr, nullptr, cgraph->nodes[गाँठ_idx + ctx->अंक_additional_fused_ops]);

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        {
            d_Qx,
            d_Qy,
            d_D,
            d_F0,
            d_F1,
        }, pc, { (uint32_t)ne03, (uint32_t)ne01, (uint32_t)ne12 });
}

अटल पूर्णांक ggml_vk_fwht_pipeline_idx(int64_t n) {
    चयन (n) {
        हाल 64:  वापस 0;
        हाल 128: वापस 1;
        हाल 256: वापस 2;
        हाल 512: वापस 3;
        शेष:  वापस -1;
    }
}

bool ggml_vk_can_use_fwht(स्थिर ggml_backend_vk_context * ctx, स्थिर ggml_tensor * src1, स्थिर ggml_tensor * dst) {
    अगर (ctx->अंक_additional_fused_ops != 0) {
        वापस false;
    }

    अगर (ggml_get_op_params_i32(dst, 1) != GGML_HINT_SRC0_IS_HADAMARD) {
        वापस false;
    }

    स्थिर पूर्णांक idx = ggml_vk_fwht_pipeline_idx(src1->ne[0]);
    अगर (idx < 0 || ctx->device->pipeline_fwht_f32[idx] == nullptr) {
        वापस false;
    }

    अगर (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        वापस false;
    }

    अगर (!ggml_is_contiguous(src1)) {
        वापस false;
    }
    GGML_ASSERT(ggml_is_contiguous(dst));

    वापस true;
}

व्योम ggml_vk_fwht(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src, ggml_tensor * dst) {
    स्थिर पूर्णांक idx = ggml_vk_fwht_pipeline_idx(src->ne[0]);
    vk_pipeline pipeline = ctx->device->pipeline_fwht_f32[idx];

    स्थिर uint32_t rows_per_workgroup = 4;
    स्थिर uint32_t n_rows = (uint32_t)ggml_nrows(src);
    स्थिर uint32_t max_workgroups_x = ctx->device->properties.limits.max_Compute_Work_Group_गिनती[0];

    स्थिर uint32_t योग_workgroups = CEIL_DIV(n_rows, rows_per_workgroup);
    स्थिर uint32_t workgroups_x = std::min(योग_workgroups, max_workgroups_x);
    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    स्थिर vk_subbuffer src_buf = ggml_vk_tensor_subbuffer(ctx, src, true);
    स्थिर vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst, true);

    vk_op_fwht_धकेलो_constants pc = {
        n_rows,
        0,
        0,
        1.0f / std::वर्ग_मूल((भग्न)src->ne[0]),
    };
    init_pushconst_tensor_offsets(ctx, pc, src, nullptr, nullptr, nullptr, dst);

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src_buf, dst_buf }, pc, { workgroups_x, 1, 1 });
}

अटल uint32_t ggml_vk_nb_elem(स्थिर ggml_tensor * t, पूर्णांक i) {
    वापस (uint32_t)(t->nb[i] / ggml_type_आकार(t->type));
}

व्योम ggml_vk_dsv4_hc_comb(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * mixes, स्थिर ggml_tensor * scale, स्थिर ggml_tensor * base, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_vk_dsv4_hc_comb(" << mixes << ", " << scale << ", " << base << ", " << dst << ")");

    vk_pipeline pipeline = ctx->device->pipeline_dsv4_hc_comb_f32;
    GGML_ASSERT(pipeline != nullptr);

    स्थिर uint32_t n_tokens = (uint32_t)mixes->ne[1];

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    स्थिर vk_subbuffer mixes_buf = ggml_vk_tensor_subbuffer(ctx, mixes, true);
    स्थिर vk_subbuffer scale_buf = ggml_vk_tensor_subbuffer(ctx, scale, true);
    स्थिर vk_subbuffer base_buf  = ggml_vk_tensor_subbuffer(ctx, base,  true);
    स्थिर vk_subbuffer dst_buf   = ggml_vk_tensor_subbuffer(ctx, dst,   true);

    vk_op_dsv4_hc_comb_धकेलो_constants pc = {
        n_tokens,
        ggml_vk_nb_elem(mixes, 0), ggml_vk_nb_elem(mixes, 1),
        ggml_vk_nb_elem(scale, 0),
        ggml_vk_nb_elem(base,  0),
        ggml_vk_nb_elem(dst,   0), ggml_vk_nb_elem(dst, 1), ggml_vk_nb_elem(dst, 2),
        0, 0, 0, 0,
        ggml_get_op_params_f32(dst, 0),
        (uint32_t)ggml_get_op_params_i32(dst, 1),
    };
    init_pushconst_tensor_offsets(ctx, pc, mixes, scale, base, nullptr, dst);

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { mixes_buf, scale_buf, base_buf, dst_buf }, pc, { n_tokens, 1, 1 });
}

व्योम ggml_vk_dsv4_hc_pre(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * x, स्थिर ggml_tensor * weights, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_vk_dsv4_hc_pre(" << x << ", " << weights << ", " << dst << ")");

    स्थिर भग्न scale = ggml_get_op_params_f32(dst, 0);
    स्थिर bool  gated = ggml_get_op_params_i32(dst, 1) != 0;

    vk_pipeline pipeline = gated ? ctx->device->pipeline_dsv4_hc_pre_gated_f32 : ctx->device->pipeline_dsv4_hc_pre_f32;
    GGML_ASSERT(pipeline != nullptr);

    स्थिर uint32_t n_embd   = (uint32_t)x->ne[0];
    स्थिर uint32_t n_tokens = (uint32_t)x->ne[2];

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    स्थिर vk_subbuffer x_buf = ggml_vk_tensor_subbuffer(ctx, x,       true);
    स्थिर vk_subbuffer w_buf = ggml_vk_tensor_subbuffer(ctx, weights, true);
    स्थिर vk_subbuffer d_buf = ggml_vk_tensor_subbuffer(ctx, dst,     true);

    vk_op_dsv4_hc_pre_धकेलो_constants pc = {
        n_embd, n_tokens,
        ggml_vk_nb_elem(x, 0), ggml_vk_nb_elem(x, 1), ggml_vk_nb_elem(x, 2),
        ggml_vk_nb_elem(weights, 0), ggml_vk_nb_elem(weights, 1), ggml_vk_nb_elem(weights, 2),
        ggml_vk_nb_elem(dst, 0), ggml_vk_nb_elem(dst, 1),
        0, 0, 0,
        scale,
    };
    init_pushconst_tensor_offsets(ctx, pc, x, weights, nullptr, nullptr, dst);

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { x_buf, w_buf, d_buf }, pc, { n_embd, n_tokens, 1 });
}

व्योम ggml_vk_dsv4_hc_post(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * x, स्थिर ggml_tensor * residual, स्थिर ggml_tensor * post, स्थिर ggml_tensor * comb, ggml_tensor * dst, स्थिर ggml_tensor * gate_scale_in) {
    VK_LOG_DEBUG("ggml_vk_dsv4_hc_post(" << x << ", " << residual << ", " << post << ", " << comb << ", " << dst << ")");

    vk_pipeline pipeline = comb ? ctx->device->pipeline_dsv4_hc_post_f32 : ctx->device->pipeline_dsv4_hc_post_nocomb_f32;
    GGML_ASSERT(pipeline != nullptr);

    स्थिर uint32_t n_embd   = (uint32_t)x->ne[0];
    स्थिर uint32_t n_tokens = (uint32_t)x->ne[1];

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    स्थिर vk_subbuffer x_buf = ggml_vk_tensor_subbuffer(ctx, x,        true);
    स्थिर vk_subbuffer r_buf = ggml_vk_tensor_subbuffer(ctx, residual, true);
    // with a fused gate, post is scale(sigmoid(scale(p_src))) and the shader applies it to p_src
    स्थिर ggml_tensor * p_src = gate_scale_in ? gate_scale_in->src[0] : post;
    स्थिर vk_subbuffer p_buf = ggml_vk_tensor_subbuffer(ctx, p_src,    true);
    स्थिर vk_subbuffer c_buf = comb ? ggml_vk_tensor_subbuffer(ctx, comb, true) : x_buf;
    स्थिर vk_subbuffer d_buf = ggml_vk_tensor_subbuffer(ctx, dst,      true);

    vk_op_dsv4_hc_post_धकेलो_constants pc = {
        n_embd, n_tokens,
        ggml_vk_nb_elem(x, 0), ggml_vk_nb_elem(x, 1),
        ggml_vk_nb_elem(residual, 0), ggml_vk_nb_elem(residual, 1), ggml_vk_nb_elem(residual, 2),
        ggml_vk_nb_elem(p_src, 0), ggml_vk_nb_elem(p_src, 1),
        comb ? ggml_vk_nb_elem(comb, 0) : 0, comb ? ggml_vk_nb_elem(comb, 1) : 0, comb ? ggml_vk_nb_elem(comb, 2) : 0,
        ggml_vk_nb_elem(dst,  0), ggml_vk_nb_elem(dst,  1), ggml_vk_nb_elem(dst,  2),
        0, 0, 0, 0, 0,
        gate_scale_in ? 1u : 0u,
        gate_scale_in ? ggml_get_op_params_f32(gate_scale_in, 0) : 1.0f,
        gate_scale_in ? ggml_get_op_params_f32(post, 0) : 1.0f,
    };
    init_pushconst_tensor_offsets(ctx, pc, x, residual, p_src, comb, dst);

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { x_buf, r_buf, p_buf, c_buf, d_buf }, pc, { n_embd, n_tokens, 1 });
}

व्योम ggml_vk_mul_mat(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर काष्ठा ggml_cgraph * cgraph, पूर्णांक गाँठ_idx) {
    ggml_tensor * dst = cgraph->nodes[गाँठ_idx];
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];
    VK_LOG_DEBUG("ggml_vk_mul_mat(" << src0 << ", " << src1 << ", " << dst << ")");

    // Handle huge A matrix by splitting the M dimensions. This works well for convolution use cases
    // where the M dimension is very large.
    // Split_k doesn't work with M splitting.
    // This only supports batchsize == 1.
    स्थिर माप_प्रकार nbytes = ggml_nbytes(src0);
    स्थिर bool needs_split = dst->ne[2] == 1 && dst->ne[3] == 1 && nbytes > ctx->device->properties.limits.max_Storage_बफर_Range;
    अगर (needs_split) {
        // Choose the number of rows that can fit (and divide by two, to allow for any additional offsets)
        स्थिर uint32_t M_split = ctx->device->properties.limits.max_Storage_बफर_Range / (2 * src0->nb[1]);
        uint32_t m_offset = 0;
        जबतक (m_offset < dst->ne[0]) {
            स्थिर uint32_t cur_M_आकार = std::min(M_split, (uint32_t)(dst->ne[0] - m_offset));
            ggml_tensor dst2 = *dst;
            ggml_tensor src02 = *src0;

            dst2.view_src = dst->view_src ? dst->view_src : dst;
            src02.view_src = src0->view_src ? src0->view_src : src0;

            dst2.view_offs += m_offset * dst->nb[0];
            src02.view_offs += m_offset * src0->nb[1];
            dst2.ne[0] = cur_M_आकार;
            src02.ne[1] = cur_M_आकार;

            ggml_vk_mul_mat_q_f16(ctx, subctx, &src02, src1, &dst2, true);

            m_offset += cur_M_आकार;
        }
    } अन्यथा अगर (ggml_vk_can_use_fwht(ctx, src1, dst)) {
        ggml_vk_fwht(ctx, subctx, src1, dst);
    } अन्यथा अगर (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && ggml_is_permuted(src0) && ggml_is_permuted(src1) && dst->ne[1] == 1 &&
        // detect 0213 permutation, and batch size of 1
        src0->nb[0] <= src0->nb[2] &&
        src0->nb[2] <= src0->nb[1] &&
        src0->nb[1] <= src0->nb[3] &&
        src1->nb[0] <= src1->nb[2] &&
        src1->nb[2] <= src1->nb[1] &&
        src1->nb[1] <= src1->nb[3] &&
        src0->ne[3] == 1 &&
        src1->ne[3] == 1 &&
        src0->ne[1] <= ctx->device->properties.limits.max_Compute_Work_Group_गिनती[1] &&
        src1->ne[2] <= ctx->device->properties.limits.max_Compute_Work_Group_गिनती[2]) {
        ggml_vk_mul_mat_vec_p021_f16_f32(ctx, subctx, cgraph, गाँठ_idx);
    } अन्यथा अगर (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && !ggml_is_contiguous(src0) && !ggml_is_transposed(src1) && dst->ne[1] == 1 &&
               !ggml_is_permuted(src0) && !ggml_is_permuted(src1) &&
               src0->ne[3] <= ctx->device->properties.limits.max_Compute_Work_Group_गिनती[0] &&
               src0->ne[1] <= ctx->device->properties.limits.max_Compute_Work_Group_गिनती[1] &&
               src1->ne[2] <= ctx->device->properties.limits.max_Compute_Work_Group_गिनती[2]) {
        ggml_vk_mul_mat_vec_nc_f16_f32(ctx, subctx, cgraph, गाँठ_idx);
    // With one output row, B^T*A has the same flat output as A^T*B.
    } अन्यथा अगर (ctx->अंक_additional_fused_ops == 0 &&
               (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
               (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_BF16 || ggml_is_quantized(src1->type)) &&
               dst->ne[0] == 1 && dst->ne[1] > mul_mat_vec_max_cols &&
               src0->ne[2] == 1 && src0->ne[3] == 1 &&
               src1->ne[2] == 1 && src1->ne[3] == 1 &&
               ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst) &&
               get_misalign_bytes(ctx, src0) == 0 && get_misalign_bytes(ctx, src1) == 0 && get_misalign_bytes(ctx, dst) == 0) {
        ggml_vk_mul_mat_vec_q_f16(ctx, subctx, cgraph, गाँठ_idx, true);
    // mul_mat_vec supports batching ne12*ne13 when ne11==1, or treating ne11 as the batch size (up to four)
    // when ne12 and ne13 are one.
    } अन्यथा अगर ((dst->ne[1] == 1 || (dst->ne[1] <= mul_mat_vec_max_cols && src1->ne[2] * src1->ne[3] == 1)) &&
               (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16 || ggml_is_quantized(src0->type))) {
        ggml_vk_mul_mat_vec_q_f16(ctx, subctx, cgraph, गाँठ_idx);
    } अन्यथा {
        ggml_vk_mul_mat_q_f16(ctx, subctx, src0, src1, dst, false);
    }
}

अटल व्योम ggml_vk_mul_mat_id_q_f16(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, स्थिर ggml_tensor * src1, स्थिर ggml_tensor * ids, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_vk_mul_mat_id_q_f16((" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::अ_त्रुटि << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::अ_त्रुटि << "), (" << ids << ", name=" << ids->name << ", type=" << ids->type << ", ne0=" << ids->ne[0] << ", ne1=" << ids->ne[1] << ", ne2=" << ids->ne[2] << ", ne3=" << ids->ne[3] << ", nb0=" << ids->nb[0] << ", nb1=" << ids->nb[1] << ", nb2=" << ids->nb[2] << ", nb3=" << ids->nb[3];
    std::अ_त्रुटि << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3] << "),)");
    GGML_ASSERT(ggml_vk_dim01_contiguous(src1) || src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);  // NOLINT
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    स्थिर uint64_t ne00 = src0->ne[0];
    स्थिर uint64_t ne01 = src0->ne[1];
    स्थिर uint64_t ne02 = src0->ne[2];
    // const uint64_t ne03 = src0->ne[3];

    स्थिर uint64_t ne10 = src1->ne[0];
    स्थिर uint64_t ne11 = src1->ne[1];
    स्थिर uint64_t ne12 = src1->ne[2];
    स्थिर uint64_t ne13 = src1->ne[3];

    स्थिर uint64_t nei0 = ids->ne[0];
    स्थिर uint64_t nei1 = ids->ne[1];

    स्थिर uint32_t nbi0 = ids->nb[0];
    स्थिर uint32_t nbi1 = ids->nb[1];

    स्थिर uint64_t ne20 = dst->ne[0];
    स्थिर uint64_t ne21 = dst->ne[1];
    // const uint64_t ne22 = dst->ne[2];
    // const uint64_t ne23 = dst->ne[3];

    स्थिर uint64_t n_as = ne02;

    ggml_backend_vk_बफर_context * dst_buf_ctx = (ggml_backend_vk_बफर_context *)dst->बफर->context;
    ggml_backend_vk_बफर_context * src0_buf_ctx = (ggml_backend_vk_बफर_context *)src0->बफर->context;
    ggml_backend_vk_बफर_context * src1_buf_ctx = (ggml_backend_vk_बफर_context *)src1->बफर->context;

    vk_बफर d_Qx = nullptr;
    माप_प्रकार qx_buf_offset = 0;
    vk_बफर d_Qy = nullptr;
    माप_प्रकार qy_buf_offset = 0;

    bool src0_uma = false;
    bool src1_uma = false;

    अगर (ctx->device->uma) {
        ggml_vk_host_get(ctx->device, src0->तथ्य, d_Qx, qx_buf_offset);
        ggml_vk_host_get(ctx->device, src1->तथ्य, d_Qy, qy_buf_offset);
        src0_uma = d_Qx != nullptr;
        src1_uma = d_Qy != nullptr;
    }

    // Reformat and convert to fp16 if non-contiguous, or for coopmat2 for better perf
    स्थिर bool x_non_contig = (ctx->device->coopmat2 && src0->type == GGML_TYPE_F32) ||
                              !ggml_vk_dim01_contiguous(src0);
    // If src0 is BF16, try to use a BF16 x BF16 multiply
    ggml_type f16_type = src0->type == GGML_TYPE_BF16 ? GGML_TYPE_BF16 : GGML_TYPE_F16;
#अगर defined(GGML_VULKAN_COOPMAT2_DECODE_VECTOR_GLSLC_SUPPORT)
    // B must already be, or be convertible to, the matmul B type used by this path.
    स्थिर bool y_decode_vector_supported = ctx->device->coopmat2_decode_vector &&
                                           (f16_type != GGML_TYPE_BF16 || ctx->device->coopmat2_bf16_support) &&
                                           (src1->type == GGML_TYPE_F32 || src1->type == f16_type);
    // If B is copied to prealloc_y, we can choose a 4-element-aligned row stride.
    स्थिर bool y_decode_vector_uses_prealloc = !ggml_vk_dim01_contiguous(src1) || src1->type != f16_type;
    // Direct B reads are safe only if row starts and the original buffer offset are 4-element aligned.
    स्थिर bool y_decode_vector_aligned =
        (ne10 % 4 == 0) &&
        (y_decode_vector_uses_prealloc || get_misalign_bytes(ctx, src1) % (4 * ggml_type_आकार(src1->type)) == 0);
    // Stage B only when decode-vector is available and direct B reads would be misaligned.
    स्थिर bool y_decode_vector_staging = y_decode_vector_supported && !y_decode_vector_aligned;
#अन्यथा
    स्थिर bool y_decode_vector_staging = false;
#पूर्ण_अगर
    स्थिर bool can_quantize_y = ctx->device->integer_dot_product && src1->type == GGML_TYPE_F32 &&
                                ggml_is_contiguous(src1) && (ne11 * ne10) % 4 == 0;
    // Intel coopmat1 matrix cores prefer f16 activations for quantized weights
    स्थिर bool prefer_f16_b = ctx->device->coopmat_support && !ctx->device->coopmat2 &&
                              ctx->device->vendor_id == VK_VENDOR_ID_INTEL &&
                              ggml_is_quantized(src0->type) && src1->type == GGML_TYPE_F32;

    स्थिर ggml_type y_kernel_type = ggml_vk_mul_mat_b_type(ctx, src0->type, src1, f16_type,
                                        (ggml_prec)dst->op_params[0], true, can_quantize_y, prefer_f16_b);

    स्थिर bool quantize_y   = y_kernel_type == GGML_TYPE_Q8_1;
    स्थिर bool y_f32_kernel = y_kernel_type == GGML_TYPE_F32;

    स्थिर std::vector<vk_matmul_pipeline_pair>* mmp_map =
        ggml_vk_get_mul_mat_mat_pipeline_map(ctx, src0->type, y_kernel_type, (ggml_prec)dst->op_params[0], true);

    स्थिर bool qx_needs_dequant = mmp_map == nullptr || x_non_contig;

    अगर (qx_needs_dequant) {
        // dequant src0 to f16
        mmp_map = ggml_vk_get_mul_mat_mat_pipeline_map(ctx, f16_type, y_kernel_type, (ggml_prec)dst->op_params[0], true);
    }

    GGML_ASSERT(mmp_map != nullptr);

    // Coopmat2 MUL_MAT_ID BK specialization constants in ggml_vk_load_shaders are at most 64.
    स्थिर uint32_t y_staged_row_stride = ctx->device->coopmat2 && !quantize_y ? ggml_vk_align_आकार(ne10, 64) : ne10;
    स्थिर bool y_needs_k_padding = ne10 != y_staged_row_stride;

    स्थिर bool y_needs_reformat = !quantize_y && (!ggml_vk_dim01_contiguous(src1) || y_kernel_type != src1->type ||
                                                  y_decode_vector_staging || y_needs_k_padding);
    स्थिर bool qy_needs_dequant = y_needs_reformat;

    स्थिर uint32_t kpad = quantize_y ? 0 : ggml_vk_align_आकार(ne10, ggml_vk_guess_matmul_pipeline_align_map(ctx, *mmp_map, ne01, nei1, true));
    स्थिर bool aligned = !quantize_y && ne10 == kpad && ne01 > 8 && nei1 > 8;

    vk_pipeline pipeline = ggml_vk_guess_matmul_pipeline_map(ctx, *mmp_map, ne01, nei1, aligned, true);

    स्थिर uint32_t bn = pipeline->wg_denoms[1];
    स्थिर uint64_t max_tiles = ggml_vk_mul_mat_id_max_tiles(n_as, nei0 * nei1, bn);
    स्थिर uint64_t row_ids_offset = 1 + 2 * max_tiles;
    स्थिर uint64_t row_map_शब्दगण = row_ids_offset + max_tiles * bn;

    अगर (ggml_nbytes(src0) > ctx->device->properties.limits.max_Storage_बफर_Range) {
        pipeline = ggml_vk_get_64b_indexing_pipeline(ctx, pipeline);
    }
    स्थिर uint64_t x_ne = ggml_nelements(src0);
    स्थिर uint64_t y_ne = (uint64_t)y_staged_row_stride * ne11 * ne12 * ne13;
    स्थिर uint64_t d_ne = ggml_nelements(dst);

    स्थिर uint64_t qx_sz = ggml_type_आकार(src0->type) * x_ne / ggml_blck_आकार(src0->type);
    स्थिर uint64_t qy_sz = ggml_type_आकार(src1->type) * ggml_nelements(src1) / ggml_blck_आकार(src1->type);
    स्थिर uint64_t x_sz = !qx_needs_dequant ? qx_sz : माप(ggml_fp16_t) * x_ne;
    स्थिर uint64_t y_sz = quantize_y ? (ggml_vk_align_आकार(y_ne, 128) * ggml_type_आकार(GGML_TYPE_Q8_1) / ggml_blck_आकार(GGML_TYPE_Q8_1)) : (y_f32_kernel ? माप(भग्न) * y_ne : माप(ggml_fp16_t) * y_ne);
    स्थिर uint64_t d_sz = माप(भग्न) * d_ne;

    vk_pipeline to_fp16_vk_0 = nullptr;
    vk_pipeline to_fp16_vk_1 = nullptr;
    vk_pipeline to_q8_1 = nullptr;

    स्वतः make_y_staged_dst = [&]() {
        ggml_tensor y_staged_dst = *src1;
        y_staged_dst.type = f16_type;
        y_staged_dst.nb[0] = ggml_type_आकार(f16_type);
        y_staged_dst.nb[1] = y_staged_dst.nb[0] * y_staged_row_stride;
        y_staged_dst.nb[2] = y_staged_dst.nb[1] * ne11;
        y_staged_dst.nb[3] = y_staged_dst.nb[2] * y_staged_dst.ne[2];
        वापस y_staged_dst;
    };

    अगर (x_non_contig) {
        to_fp16_vk_0 = ggml_vk_get_cpy_pipeline(ctx, src0, nullptr, f16_type);
    } अन्यथा {
        to_fp16_vk_0 = ggml_vk_get_to_fp16(ctx, src0->type);
    }
    अगर (y_needs_reformat) {
        ggml_tensor y_staged_dst;
        स्थिर ggml_tensor * y_staged_dst_ptr = nullptr;
        अगर (y_needs_k_padding) {
            y_staged_dst = make_y_staged_dst();
            y_staged_dst_ptr = &y_staged_dst;
        }

        to_fp16_vk_1 = ggml_vk_get_cpy_pipeline(ctx, src1, y_staged_dst_ptr, f16_type);
    } अन्यथा {
        to_fp16_vk_1 = ggml_vk_get_to_fp16(ctx, src1->type);
    }
    GGML_ASSERT(!qx_needs_dequant || to_fp16_vk_0 != nullptr);  // NOLINT
    GGML_ASSERT(!qy_needs_dequant || to_fp16_vk_1 != nullptr);  // NOLINT

    अगर (quantize_y) {
        to_q8_1 = ggml_vk_get_quantize_pipeline(ctx, GGML_TYPE_Q8_1);
    }
    vk_pipeline गिनती_experts = ctx->device->pipeline_गिनती_experts;

    स्थिर माप_प्रकार expert_तथ्य_आकार = माप(uint32_t) * row_map_शब्दगण;

    {
        अगर (
                (qx_needs_dequant && x_sz > ctx->device->properties.limits.max_Storage_बफर_Range) ||
                (qy_needs_dequant && y_sz > ctx->device->properties.limits.max_Storage_बफर_Range)) {
            GGML_ABORT("Requested preallocation size is too large");
        }
        अगर (qx_needs_dequant && ctx->prealloc_आकार_x < x_sz) {
            ctx->prealloc_आकार_x = x_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        अगर ((qy_needs_dequant || quantize_y) && ctx->prealloc_आकार_y < y_sz) {
            ctx->prealloc_आकार_y = y_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        अगर (ctx->prealloc_आकार_split_k < expert_तथ्य_आकार) {
            ctx->prealloc_आकार_split_k = expert_तथ्य_आकार;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }

        // Request descriptor sets
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
        अगर (qx_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, to_fp16_vk_0, 1);
        }
        अगर (qy_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, to_fp16_vk_1, 1);
        }
        अगर (quantize_y) {
            ggml_pipeline_request_descriptor_sets(ctx, to_q8_1, 1);
        }
        ggml_pipeline_request_descriptor_sets(ctx, गिनती_experts, 1);
    }

    vk_बफर d_D = dst_buf_ctx->dev_बफर;
    स्थिर uint64_t d_buf_offset = vk_tensor_offset(dst) + dst->view_offs;
    GGML_ASSERT(d_D != nullptr);
    vk_बफर d_X;
    uint64_t x_buf_offset = 0;
    vk_बफर d_Y;
    uint64_t y_buf_offset = 0;
    अगर (!src0_uma) {
        d_Qx = src0_buf_ctx->dev_बफर;
        qx_buf_offset = vk_tensor_offset(src0) + src0->view_offs;
        GGML_ASSERT(d_Qx != nullptr);
    }
    अगर (!src1_uma) {
        d_Qy = src1_buf_ctx->dev_बफर;
        qy_buf_offset = vk_tensor_offset(src1) + src1->view_offs;
        GGML_ASSERT(d_Qy != nullptr);
    }
    अगर (qx_needs_dequant) {
        d_X = ctx->prealloc_x;
        GGML_ASSERT(d_X->size >= x_sz);
    } अन्यथा {
        d_X = d_Qx;
        x_buf_offset = qx_buf_offset;
        GGML_ASSERT(qx_sz == x_sz);
    }
    अगर (qy_needs_dequant) {
        d_Y = ctx->prealloc_y;
        GGML_ASSERT(d_Y->size >= y_sz);
    } अन्यथा अगर (quantize_y) {
        d_Y = ctx->prealloc_y;
        GGML_ASSERT(d_Y->size >= CEIL_DIV(y_sz, 144) * 144);
    } अन्यथा {
        d_Y = d_Qy;
        y_buf_offset = qy_buf_offset;
        GGML_ASSERT(qy_sz == y_sz);
    }

    अगर (x_non_contig || qx_needs_dequant) {
        अगर (ctx->prealloc_x_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }
    }
    vk_subbuffer d_ids = ggml_vk_tensor_subbuffer(ctx, ids, true);

    // Count how many times each expert is used
    vk_subbuffer expert_गिनती_buf = { ctx->prealloc_split_k, 0, expert_तथ्य_आकार };
    अगर (ctx->prealloc_split_k_need_sync) {
        ggml_vk_sync_buffers(ctx, subctx);
    }
    {
        vk_op_गिनती_experts_धकेलो_constants pc = { (uint32_t)nei0,
                                           (uint32_t)nei1,
                                           (uint32_t)(nbi0 / ggml_type_आकार(ids->type)),
                                           (uint32_t)(nbi1 / ggml_type_आकार(ids->type)),
                                           (uint32_t)(get_misalign_bytes(ctx, ids) / ggml_type_आकार(ids->type)),
                                           (uint32_t)n_as,
                                           0, 0, bn, (uint32_t)row_ids_offset };
        init_pushconst_fastdiv(pc);
        ggml_vk_dispatch_pipeline(ctx, subctx, गिनती_experts,
            { d_ids, expert_गिनती_buf }, pc,
            { 1, 1, 1 });
    }

    अगर (x_non_contig) {
        ggml_vk_cpy_to_contiguous(ctx, subctx, to_fp16_vk_0, src0, ggml_vk_subbuffer(ctx, d_Qx, qx_buf_offset), ggml_vk_subbuffer(ctx, d_X, 0));
    } अन्यथा अगर (qx_needs_dequant) {
        स्थिर std::vector<uint32_t> pc = { (uint32_t)ne01, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)(ggml_nelements(src0)) };
        ggml_vk_dispatch_pipeline(ctx, subctx, to_fp16_vk_0,
            { vk_subbuffer{ d_Qx, qx_buf_offset, qx_sz }, vk_subbuffer{ d_X, 0, x_sz } }, pc, { (uint32_t)x_ne, 1, 1});
    }
    अगर (y_needs_reformat) {
        अगर (ctx->prealloc_y_last_pipeline_used != to_fp16_vk_1.लो() ||
            ctx->prealloc_y_last_tensor_used != src1 ||
            ctx->prealloc_y_last_k_padded != y_needs_k_padding) {
            अगर (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            अगर (y_needs_k_padding) {
                GGML_ASSERT(y_sz % 4 == 0);
                // Zero B padding because clamping only A can produce 0 * Inf or NaN.
                subctx->s->बफर->buf.fill_बफर(d_Y->बफर, 0, y_sz, 0);
                ggml_vk_sync_buffers(ctx, subctx);
                स्थिर ggml_tensor y_staged_dst = make_y_staged_dst();
                स्थिर uint32_t y_staged_dst_type_आकार = ggml_type_आकार(y_staged_dst.type);
                ggml_vk_cpy_to_strided(
                    ctx, subctx, to_fp16_vk_1, src1,
                    ggml_vk_subbuffer(ctx, d_Qy, qy_buf_offset), ggml_vk_subbuffer(ctx, d_Y, 0),
                    (uint32_t)(y_staged_dst.nb[0] / y_staged_dst_type_आकार),
                    (uint32_t)(y_staged_dst.nb[1] / y_staged_dst_type_आकार),
                    (uint32_t)(y_staged_dst.nb[2] / y_staged_dst_type_आकार),
                    (uint32_t)(y_staged_dst.nb[3] / y_staged_dst_type_आकार));
            } अन्यथा {
                ggml_vk_cpy_to_contiguous(ctx, subctx, to_fp16_vk_1, src1, ggml_vk_subbuffer(ctx, d_Qy, qy_buf_offset), ggml_vk_subbuffer(ctx, d_Y, 0));
            }
            ctx->prealloc_y_last_pipeline_used = to_fp16_vk_1.लो();
            ctx->prealloc_y_last_tensor_used = src1;
            ctx->prealloc_y_last_k_padded = y_needs_k_padding;
        }
    }
    अगर (quantize_y) {
        अगर (ctx->prealloc_y_last_pipeline_used != to_q8_1.लो() ||
            ctx->prealloc_y_last_tensor_used != src1 ||
            ctx->prealloc_y_last_k_padded) {
            अगर (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ggml_vk_quantize_q8_1(ctx, subctx, ggml_vk_subbuffer(ctx, d_Qy, qy_buf_offset), ggml_vk_subbuffer(ctx, d_Y, 0), y_ne);
            ctx->prealloc_y_last_pipeline_used = to_q8_1.लो();
            ctx->prealloc_y_last_tensor_used = src1;
            ctx->prealloc_y_last_k_padded = false;
        }
    }
    ggml_vk_sync_buffers(ctx, subctx);

    स्थिर uint64_t x_range = qx_needs_dequant ? x_sz : ggml_nbytes(src0);
    स्थिर uint64_t y_range = (qy_needs_dequant || quantize_y) ? y_sz : ggml_nbytes(src1);

    uint32_t stride_batch_x = qx_needs_dequant ? ne00*ne01 : ggml_vk_batch_stride(src0);
    uint32_t stride_b_y = y_needs_k_padding ? y_staged_row_stride : ne10;
    uint32_t stride_batch_y = y_needs_k_padding ? y_staged_row_stride * ne11 : ((qy_needs_dequant || quantize_y) ? ne10*ne11 : ggml_vk_batch_stride(src1));

    अगर (!ggml_vk_dim01_contiguous(src0) && !qx_needs_dequant) {
        stride_batch_x = src0->nb[0] / ggml_type_आकार(src0->type);
    }

    अगर (!ggml_vk_dim01_contiguous(src1) && !qy_needs_dequant && !quantize_y) {
        stride_batch_y = src1->nb[0] / ggml_type_आकार(src1->type);
    }

    // compute
    ggml_vk_matmul_id(
        ctx, subctx, pipeline,
        { d_X, x_buf_offset, x_range }, { d_Y, y_buf_offset, y_range },
        { d_D, d_buf_offset, d_sz }, std::move(d_ids), expert_गिनती_buf,
        ne01, ne21, ne10, ne10, stride_b_y, ne01,
        stride_batch_x, stride_batch_y, ne20*ne21,
        n_as, nei0, nei1, nbi1 / ggml_type_आकार(ids->type), ne11, row_ids_offset, max_tiles
    );  // NOLINT

    अगर (x_non_contig || qx_needs_dequant) {
        ctx->prealloc_x_need_sync = true;
    }
    अगर (y_needs_reformat || quantize_y) {
        ctx->prealloc_y_need_sync = true;
    }
    ctx->prealloc_split_k_need_sync = true;
}

अटल व्योम ggml_vk_mul_mat_vec_id_q_f16(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर काष्ठा ggml_cgraph * cgraph, पूर्णांक गाँठ_idx) {
    ggml_tensor * dst = cgraph->nodes[गाँठ_idx];
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];
    ggml_tensor * ids = dst->src[2];
    VK_LOG_DEBUG("ggml_vk_mul_mat_vec_id_q_f16((" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::अ_त्रुटि << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::अ_त्रुटि << "), (" << ids << ", name=" << ids->name << ", type=" << ids->type << ", ne0=" << ids->ne[0] << ", ne1=" << ids->ne[1] << ", ne2=" << ids->ne[2] << ", ne3=" << ids->ne[3] << ", nb0=" << ids->nb[0] << ", nb1=" << ids->nb[1] << ", nb2=" << ids->nb[2] << ", nb3=" << ids->nb[3];
    std::अ_त्रुटि << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::अ_त्रुटि << "))");
    GGML_ASSERT(ggml_vk_dim01_contiguous(src0) || src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16);  // NOLINT
    GGML_ASSERT(ggml_vk_dim01_contiguous(src1) || src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);  // NOLINT
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    स्थिर uint64_t ne00 = src0->ne[0];
    स्थिर uint64_t ne01 = src0->ne[1];
    // const uint64_t ne02 = src0->ne[2];
    // const uint64_t ne03 = src0->ne[3];

    स्थिर uint64_t ne10 = src1->ne[0];
    स्थिर uint64_t ne11 = src1->ne[1];
    स्थिर uint64_t ne12 = src1->ne[2];
    // const uint64_t ne13 = src1->ne[3];

    स्थिर uint64_t nei0 = ids->ne[0];
    स्थिर uint64_t nei1 = ids->ne[1];
    स्थिर uint32_t nbi1 = (uint32_t)(ids->nb[1] / माप(पूर्णांक));

    स्थिर uint64_t ne20 = dst->ne[0];
    स्थिर uint64_t ne21 = dst->ne[1];
    // const uint64_t ne22 = dst->ne[2];
    // const uint64_t ne23 = dst->ne[3];

    स्थिर bool x_non_contig = !ggml_vk_dim01_contiguous(src0);
    स्थिर bool y_non_contig = !ggml_vk_dim01_contiguous(src1);

    स्थिर bool f16_f32_kernel = src1->type == GGML_TYPE_F32;
    bool quantize_y = ctx->device->integer_dot_product && src1->type == GGML_TYPE_F32 && ggml_is_contiguous(src1) && !y_non_contig && (ne11 * ne10) % 4 == 0 && ggml_vk_should_use_mmvq(ctx->device, ne01, ne12, ne10, src0->type);

    vk_pipeline to_fp16_vk_0 = nullptr;
    vk_pipeline to_fp16_vk_1 = nullptr;
    अगर (x_non_contig) {
        to_fp16_vk_0 = ggml_vk_get_cpy_pipeline(ctx, src0, nullptr, src0->type);
    }
    अगर (y_non_contig) {
        to_fp16_vk_1 = ggml_vk_get_cpy_pipeline(ctx, src1, nullptr, src1->type);
    } अन्यथा {
        to_fp16_vk_1 = ggml_vk_get_to_fp16(ctx, src1->type);
    }

    // Check for mmq first
    vk_pipeline dmmv = quantize_y ? ggml_vk_get_dequantize_mul_mat_vec_id(ctx, src0->type, GGML_TYPE_Q8_1, ne20, ne00) : nullptr;
    vk_pipeline to_q8_1 = nullptr;

    अगर (dmmv == nullptr) {
        // Fall back to f16 dequant mul mat
        dmmv = ggml_vk_get_dequantize_mul_mat_vec_id(ctx, src0->type, src1->type, ne20, ne00);
        quantize_y = false;
    }

    अगर (quantize_y) {
        to_q8_1 = ggml_vk_get_quantize_pipeline(ctx, GGML_TYPE_Q8_1);
    }

    स्थिर bool qx_needs_dequant = x_non_contig;
    स्थिर bool qy_needs_dequant = !quantize_y && ((src1->type != GGML_TYPE_F16 && !f16_f32_kernel) || y_non_contig);

    अगर (ggml_nbytes(src0) > ctx->device->properties.limits.max_Storage_बफर_Range) {
        dmmv = ggml_vk_get_64b_indexing_pipeline(ctx, dmmv);
    }

    // Not implemented
    GGML_ASSERT(y_non_contig || !qy_needs_dequant);  // NOLINT
    GGML_ASSERT(!qx_needs_dequant || to_fp16_vk_0 != nullptr);  // NOLINT
    GGML_ASSERT(!qy_needs_dequant || to_fp16_vk_1 != nullptr);  // NOLINT
    GGML_ASSERT(dmmv != nullptr);

    स्थिर uint64_t x_ne = ggml_nelements(src0);
    स्थिर uint64_t y_ne = ggml_nelements(src1);

    स्थिर uint64_t qx_sz = ggml_vk_align_आकार(ggml_type_आकार(src0->type) * x_ne / ggml_blck_आकार(src0->type), ctx->device->properties.limits.min_Storage_बफर_Offset_Alignment);
    स्थिर uint64_t x_sz = x_non_contig ? ggml_vk_align_आकार(ggml_type_आकार(src0->type) * x_ne, ctx->device->properties.limits.min_Storage_बफर_Offset_Alignment) : qx_sz;
    स्थिर uint64_t y_sz = quantize_y ? (ggml_vk_align_आकार(y_ne, 128) * ggml_type_आकार(GGML_TYPE_Q8_1) / ggml_blck_आकार(GGML_TYPE_Q8_1)) :
                                       (f16_f32_kernel ? माप(भग्न) * y_ne : माप(ggml_fp16_t) * y_ne);

    {
        अगर (
                (qx_needs_dequant && x_sz > ctx->device->properties.limits.max_Storage_बफर_Range) ||
                (qy_needs_dequant && y_sz > ctx->device->properties.limits.max_Storage_बफर_Range)) {
            GGML_ABORT("Requested preallocation size is too large");
        }
        अगर (qx_needs_dequant && ctx->prealloc_आकार_x < x_sz) {
            ctx->prealloc_आकार_x = x_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        अगर ((qy_needs_dequant || quantize_y) && ctx->prealloc_आकार_y < y_sz) {
            ctx->prealloc_आकार_y = y_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }

        // Request descriptor sets
        अगर (qx_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, to_fp16_vk_0, 1);
        }
        अगर (qy_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, to_fp16_vk_1, 1);
        }
        अगर (quantize_y) {
            ggml_pipeline_request_descriptor_sets(ctx, to_q8_1, 1);
        }
        ggml_pipeline_request_descriptor_sets(ctx, dmmv, nei1);
    }

    vk_subbuffer d_D = ggml_vk_tensor_subbuffer(ctx, cgraph->nodes[गाँठ_idx + ctx->अंक_additional_fused_ops]);
    vk_subbuffer d_Qx = ggml_vk_tensor_subbuffer(ctx, src0);
    vk_subbuffer d_Qy = ggml_vk_tensor_subbuffer(ctx, src1);
    vk_subbuffer d_ids = ggml_vk_tensor_subbuffer(ctx, ids);
    vk_subbuffer d_F0 = d_D;
    vk_subbuffer d_X, d_Y;

    अगर (qx_needs_dequant) {
        d_X = { ctx->prealloc_x, 0, ctx->prealloc_x->size };
    } अन्यथा {
        d_X = d_Qx;
    }
    अगर (qy_needs_dequant || quantize_y) {
        d_Y = { ctx->prealloc_y, 0, ctx->prealloc_y->size };
    } अन्यथा {
        d_Y = d_Qy;
    }

    अगर (x_non_contig) {
        अगर (ctx->prealloc_x_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }
    }

    अगर (x_non_contig) {
        GGML_ASSERT(x_sz == ggml_vk_align_आकार(ggml_type_आकार(src0->type) * x_ne, ctx->device->properties.limits.min_Storage_बफर_Offset_Alignment));
        ggml_vk_cpy_to_contiguous(ctx, subctx, to_fp16_vk_0, src0, d_Qx, d_X);
    }
    अगर (y_non_contig) {
        GGML_ASSERT(y_sz == ggml_type_आकार(src1->type) * y_ne);
        अगर (ctx->prealloc_y_last_pipeline_used != to_fp16_vk_1.लो() ||
            ctx->prealloc_y_last_tensor_used != src1 ||
            ctx->prealloc_y_last_k_padded) {
            अगर (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ggml_vk_cpy_to_contiguous(ctx, subctx, to_fp16_vk_1, src1, d_Qy, d_Y);
            ctx->prealloc_y_last_pipeline_used = to_fp16_vk_1.लो();
            ctx->prealloc_y_last_tensor_used = src1;
            ctx->prealloc_y_last_k_padded = false;
        }
    }
    अगर (quantize_y) {
        अगर (ctx->prealloc_y_last_pipeline_used != to_q8_1.लो() ||
            ctx->prealloc_y_last_tensor_used != src1 ||
            ctx->prealloc_y_last_k_padded) {
            अगर (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ggml_vk_quantize_q8_1(ctx, subctx, d_Qy, d_Y, y_ne);
            ctx->prealloc_y_last_pipeline_used = to_q8_1.लो();
            ctx->prealloc_y_last_tensor_used = src1;
            ctx->prealloc_y_last_k_padded = false;
        }
    }

    uint32_t stride_batch_x = qx_needs_dequant ? ne00*ne01 : ggml_vk_batch_stride(src0);
    uint32_t stride_batch_y = (qy_needs_dequant || quantize_y) ? ne10*ne11 : ggml_vk_batch_stride(src1);

    अगर (!ggml_vk_dim01_contiguous(src1) && !qy_needs_dequant) {
        stride_batch_y = src1->nb[2] / ggml_type_आकार(src1->type);
    }

    स्थिर uint32_t max_groups_x = ctx->device->properties.limits.max_Compute_Work_Group_गिनती[0];

    uint32_t groups_x = ne01;
    uint32_t groups_z = 1;

    अगर (ne01 > max_groups_x) {
        groups_z = 64;
        groups_x = CEIL_DIV(groups_x, groups_z);
    }

    uint32_t fusion_flags = 0;

    अगर (ctx->अंक_additional_fused_ops > 0) {
        स्थिर ggml_tensor * bias = cgraph->nodes[गाँठ_idx + 1]->src[1];

        d_F0 = ggml_vk_tensor_subbuffer(ctx, bias);

        अगर (cgraph->nodes[गाँठ_idx + 1]->op == GGML_OP_MUL) {
            fusion_flags |= MAT_VEC_FUSION_FLAGS_SCALE0;
        } अन्यथा {
            GGML_ASSERT(cgraph->nodes[गाँठ_idx + 1]->op == GGML_OP_ADD_ID);
            fusion_flags |= MAT_VEC_FUSION_FLAGS_BIAS0;
        }
    }

    vk_subbuffer d_F1 = d_D;
    अगर (ctx->अंक_additional_fused_ops > 1) {
        स्थिर ggml_tensor * scale = cgraph->nodes[गाँठ_idx + 2]->src[1];

        d_F1 = ggml_vk_tensor_subbuffer(ctx, scale);
        fusion_flags |= MAT_VEC_FUSION_FLAGS_SCALE1;
    }

    // Loop over the batch dimension
    क्रम (uint32_t expert_i1 = 0; expert_i1 < nei1; ++expert_i1) {
        स्थिर vk_mat_vec_id_धकेलो_constants pc = {
            (uint32_t)ne00, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)ne01,
            stride_batch_x, stride_batch_y, (uint32_t)(ne20 * ne21),
            fusion_flags,
            (uint32_t)nei0, (uint32_t)ne11, expert_i1, nbi1
        };
        ggml_vk_dispatch_pipeline(ctx, subctx, dmmv,
            {
                d_X,
                d_Y,
                d_D,
                d_F0,
                d_F1,
                d_ids,
            },
            pc, { groups_x, (uint32_t)nei0, groups_z });
    }

    अगर (x_non_contig) {
        ctx->prealloc_x_need_sync = true;
    }
    अगर (y_non_contig || quantize_y) {
        ctx->prealloc_y_need_sync = true;
    }
}

अटल bool ggml_vk_use_mul_mat_vec_id(स्थिर ggml_tensor * dst) {
    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * ids = dst->src[2];
    वापस ids->ne[1] <= 8 && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type));
}

bool ggml_vk_use_mul_mat_vec_id(स्थिर काष्ठा ggml_cgraph * cgraph, पूर्णांक गाँठ_idx) {
    वापस ggml_vk_use_mul_mat_vec_id(cgraph->nodes[गाँठ_idx]);
}

व्योम ggml_vk_mul_mat_id(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर काष्ठा ggml_cgraph * cgraph, पूर्णांक गाँठ_idx) {
    ggml_tensor * dst = cgraph->nodes[गाँठ_idx];
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];
    ggml_tensor * src2 = dst->src[2];
    VK_LOG_DEBUG("ggml_vk_mul_mat_id(" << src0 << ", " << src1 << ", " << src2 << ", " << dst << ")");
    अगर (ggml_vk_use_mul_mat_vec_id(cgraph, गाँठ_idx)) {
        ggml_vk_mul_mat_vec_id_q_f16(ctx, subctx, cgraph, गाँठ_idx);
    } अन्यथा {
        ggml_vk_mul_mat_id_q_f16(ctx, subctx, src0, src1, src2, dst);
    }
}

bool ggml_vk_flash_attn_scalar_shmem_support(स्थिर vk_device& device, स्थिर vk_fa_tuning_params& params, uint32_t hsk, uint32_t hsv, bool f32acc, ggml_type k_type, ggml_type v_type) {
    GGML_UNUSED(f32acc);
    // Needs to be kept up to date on shader changes
    स्थिर uint32_t wg_आकार = params.workgroup_आकार;
    स्थिर uint32_t Br = params.block_rows;
    स्थिर uint32_t Bc = params.block_cols;

    // BF16 uses the fp32 shader (FLOAT_TYPE=float)
    स्थिर uint32_t float_type_आकार = (device->fp16 && k_type != GGML_TYPE_BF16) ? माप(ggml_fp16_t) : माप(भग्न);

    स्थिर bool mmq = ggml_vk_fa_scalar_uses_mmq(device, k_type, v_type);

    // tmpsh is overestimated slightly
    स्थिर uint32_t tmpsh = wg_आकार * माप(भग्न);
    स्थिर uint32_t tmpshv4 = wg_आकार * 4 * float_type_आकार;

    स्थिर uint32_t masksh = Bc * (Br + 1) * float_type_आकार;
    // DATA_A_IQ4_NL is compiled into the FA shaders unconditionally, so its shared table is always allocated.
    स्थिर uint32_t iq_shmem = 16 * float_type_आकार;

    uint32_t Qf, kvsh, kblocksh_आकार;
    अगर (mmq) {
        // block_b_cache: int32_t qs[8] + FLOAT_TYPEV2 ds
        स्थिर uint32_t block_b_आकार = 8 * माप(int32_t) + 2 * float_type_आकार;
        Qf = Br * (hsk / 32) * block_b_आकार;

        // kvsh uses D = HSV (K goes through kblocksh instead)
        kvsh = params.shmem_staging ? Bc * (hsv / 4 + 1) * 4 * float_type_आकार : 4 * float_type_आकार;

        // The mixed MMQ shader uses a superset block_a_cache that fits every
        // FA-supported quant: int32_t qs[8] + uint32_t qh + FLOAT_TYPEV2 dm.
        // Single-scale types leave dm.y unused; non-Q5_* leave qh unused.
        स्थिर uint32_t block_a_आकार = 8 * माप(int32_t) + माप(uint32_t) + 2 * float_type_आकार;
        kblocksh_आकार = params.shmem_staging ? Bc * (hsk / 32) * block_a_आकार : block_a_आकार;
    } अन्यथा {
        Qf = Br * (hsk / 4 + 1) * 4 * float_type_आकार;

        स्थिर uint32_t D = std::max(hsk, hsv);
        kvsh = params.shmem_staging ? Bc * (D / 4 + 1) * 4 * float_type_आकार : 4 * float_type_आकार;

        kblocksh_आकार = 0;
    }

    स्थिर uint32_t योग_आकार = tmpsh + tmpshv4 + masksh + iq_shmem + Qf + kvsh + kblocksh_आकार;
    स्थिर bool supported = योग_आकार <= device->properties.limits.max_Compute_Shared_Memory_आकार;

    VK_LOG_DEBUG("ggml_vk_flash_attn_scalar_shmem_support(HSK=" << hsk << ", HSV=" << hsv << ", mmq=" << mmq << ", total_size=" << योग_आकार << ", supported=" << supported);

    वापस supported;
}

bool ggml_vk_flash_attn_coopmat_shmem_support(स्थिर vk_device& device, स्थिर vk_fa_tuning_params& params, uint32_t hsk, uint32_t hsv, bool f32acc, ggml_type k_type, ggml_type v_type) {
    GGML_UNUSED(v_type);
    // Needs to be kept up to date on shader changes
    स्थिर uint32_t Br = params.block_rows;
    स्थिर uint32_t Bc = params.block_cols;

    स्थिर uint32_t MatBr = 16, MatBc = 16;

    स्थिर uint32_t row_split = Bc / MatBc;

    स्थिर uint32_t hsk_pad = ROUNDUP_POW2(hsk, 16);
    स्थिर uint32_t hsv_pad = ROUNDUP_POW2(hsv, 16);

    स्थिर uint32_t acctype = f32acc ? 4 : 2;
    स्थिर uint32_t f16vec4 = 8;

    स्थिर uint32_t tmpsh = (Bc / MatBc) * माप(भग्न);
    // DATA_A_IQ4_NL is compiled into the FA shaders unconditionally, so its shared table is always allocated.
    स्थिर uint32_t iq_shmem = 16 * माप(ggml_fp16_t);

    स्थिर uint32_t qstride = hsk_pad / 4 + 2;
    स्थिर uint32_t Qf = Br * qstride * f16vec4;

    स्थिर uint32_t psh_stride = Br / 4 + 2;
    स्थिर uint32_t Psh = Bc * psh_stride * f16vec4;

    स्थिर uint32_t sfshstride = (hsk <= 128) ? (Br + 8) : Br;
    स्थिर uint32_t sfsh = Bc * sfshstride * acctype;

    स्थिर uint32_t kvshstride = (params.shmem_staging ? std::max(hsk_pad, hsv_pad) : MatBr) / 4 + 2;
    स्थिर uint32_t vsh_stride = MatBc / 4 * row_split;
    स्थिर uint32_t ksh = ((kvshstride >= vsh_stride) ? (Bc * kvshstride) : (Bc * vsh_stride)) * f16vec4;

    // BF16 PVMat accumulator is f32 (no bf16 accumulator support), so pvsh is vec4 (16 bytes)
    स्थिर uint32_t pvsh_elem_आकार = (k_type == GGML_TYPE_BF16) ? 16u : f16vec4;
    स्थिर uint32_t osh_stride = params.row_split * MatBr / 4;
    स्थिर uint32_t pvsh = MatBc * osh_stride * pvsh_elem_आकार;

    स्थिर uint32_t slope = Br * acctype;

    स्थिर uint32_t योग_आकार = tmpsh + iq_shmem + Qf + Psh + sfsh + ksh + pvsh + slope;
    स्थिर bool supported = योग_आकार <= device->properties.limits.max_Compute_Shared_Memory_आकार;

    VK_LOG_DEBUG("ggml_vk_flash_attn_coopmat_shmem_support(HSK=" << hsk << ", HSV=" << hsv << ", f32acc=" << f32acc << ", total_size=" << योग_आकार << ", supported=" << supported);

    वापस supported;
}

व्योम ggml_vk_flash_attn(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * q, स्थिर ggml_tensor * k, स्थिर ggml_tensor * v, स्थिर ggml_tensor * mask, स्थिर ggml_tensor * sinks, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_vk_flash_attn((" << q << ", name=" << q->name << ", type=" << q->type << ", ne0=" << q->ne[0] << ", ne1=" << q->ne[1] << ", ne2=" << q->ne[2] << ", ne3=" << q->ne[3] << ", nb0=" << q->nb[0] << ", nb1=" << q->nb[1] << ", nb2=" << q->nb[2] << ", nb3=" << q->nb[3];
    std::अ_त्रुटि << "), (" << k << ", name=" << k->name << ", type=" << k->type << ", ne0=" << k->ne[0] << ", ne1=" << k->ne[1] << ", ne2=" << k->ne[2] << ", ne3=" << k->ne[3] << ", nb0=" << k->nb[0] << ", nb1=" << k->nb[1] << ", nb2=" << k->nb[2] << ", nb3=" << k->nb[3];
    std::अ_त्रुटि << "), (" << v << ", name=" << v->name << ", type=" << v->type << ", ne0=" << v->ne[0] << ", ne1=" << v->ne[1] << ", ne2=" << v->ne[2] << ", ne3=" << v->ne[3] << ", nb0=" << v->nb[0] << ", nb1=" << v->nb[1] << ", nb2=" << v->nb[2] << ", nb3=" << v->nb[3];
    std::अ_त्रुटि << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    अगर (sinks) {
        std::अ_त्रुटि << "), (" << sinks << ", name=" << sinks->name << ", type=" << sinks->type << ", ne0=" << sinks->ne[0] << ", ne1=" << sinks->ne[1] << ", ne2=" << sinks->ne[2] << ", ne3=" << sinks->ne[3] << ", nb0=" << sinks->nb[0] << ", nb1=" << sinks->nb[1] << ", nb2=" << sinks->nb[2] << ", nb3=" << sinks->nb[3];
    }
    std::अ_त्रुटि << "))");

    GGML_TENSOR_LOCALS(int64_t, neq, q,   ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nbq, q,   nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k,   ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nbk, k,   nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v,   ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nbv, v,   nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst, ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nb,  dst, nb)

    स्थिर uint32_t nem0 = mask ? mask->ne[0] : 0;
    स्थिर uint32_t nem1 = mask ? mask->ne[1] : 0;
    स्थिर uint32_t nem2 = mask ? mask->ne[2] : 0;
    स्थिर uint32_t nem3 = mask ? mask->ne[3] : 0;

    स्थिर uint32_t HSK = nek0;
    स्थिर uint32_t HSV = nev0;
    uint32_t N = neq1;
    स्थिर uint32_t KV = nek1;

    GGML_ASSERT(ne0 == HSV);
    GGML_ASSERT(ne2 == N);

    // input tensor rows must be contiguous
    GGML_ASSERT(nbq0 == ggml_type_आकार(q->type));
    GGML_ASSERT(nbk0 == ggml_type_आकार(k->type));
    GGML_ASSERT(nbv0 == ggml_type_आकार(v->type));

    GGML_ASSERT(neq0 == HSK);

    GGML_ASSERT(neq1 == N);

    GGML_ASSERT(nev1 == nek1);

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == माप(भग्न));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    निश्चित(dst->type == GGML_TYPE_F32);
    निश्चित(q->type == GGML_TYPE_F32);
    uint32_t gqa_ratio = 1;
    uint32_t qk_ratio = neq2 / nek2;
    uint32_t workgroups_x = (uint32_t)neq1;
    uint32_t workgroups_y = (uint32_t)neq2;
    uint32_t workgroups_z = (uint32_t)neq3;

    स्थिर bool f32acc = !ctx->device->fp16 || dst->op_params[3] == GGML_PREC_F32 || k->type == GGML_TYPE_BF16;

    // dequant K/V once into an f16 scratch, reordered KV layout so FA can read without a stride
    स्वतः is_dense_kv_cache = [](स्थिर ggml_tensor * t) {
        वापस t->nb[0] == ggml_type_आकार(t->type) &&
               t->nb[2] == ggml_row_आकार(t->type, t->ne[0]) &&
               t->nb[1] == t->nb[2] * t->ne[2] &&
               (t->ne[3] == 1 || t->nb[3] == t->nb[1] * t->ne[1]);
    };
    स्थिर bool k_quant = k->type != GGML_TYPE_F16 && k->type != GGML_TYPE_BF16 && k->type != GGML_TYPE_F32;
    स्थिर bool v_quant = v->type != GGML_TYPE_F16 && v->type != GGML_TYPE_BF16 && v->type != GGML_TYPE_F32;
    स्थिर bool kv_raw = (k->type == GGML_TYPE_F16 || k->type == GGML_TYPE_BF16) && k->type == v->type;
    // gated to AMD, except RDNA4, which regressed in benchmarks
    स्थिर bool kv_raw_strided = ctx->device->vendor_id == VK_VENDOR_ID_AMD &&
                                ctx->device->architecture != vk_device_architecture::AMD_RDNA4 &&
                                neq1 >= 256 &&
                                kv_raw &&
                                (k->nb[1] != (uint64_t)HSK * ggml_type_आकार(k->type) ||
                                 v->nb[1] != (uint64_t)HSV * ggml_type_आकार(v->type)) &&
                                (HSK % 8) == 0 && (HSV % 8) == 0;
    स्थिर माप_प्रकार kv_scratch_ts = kv_raw_strided ? ggml_type_आकार(k->type) : माप(ggml_fp16_t);
    स्थिर bool use_dequant_kv = ((k_quant && v_quant) || kv_raw_strided) && neq1 >= 64 &&
                                is_dense_kv_cache(k) && is_dense_kv_cache(v) &&
                                (uint64_t)ggml_nelements(k) * kv_scratch_ts <= ctx->device->properties.limits.max_Storage_बफर_Range &&
                                (uint64_t)ggml_nelements(v) * kv_scratch_ts <= ctx->device->properties.limits.max_Storage_बफर_Range &&
                                (kv_raw_strided ||
                                 (ctx->device->pipeline_dequant_transpose[k->type] != nullptr &&
                                  ctx->device->pipeline_dequant_transpose[v->type] != nullptr)) &&
                                // coopmat2 path does not benefit from the f16 scratch
                                !ctx->device->coopmat2 &&
                                // Intel Xe1 regresses, see PR 25494
                                (ctx->device->vendor_id != VK_VENDOR_ID_INTEL ||
                                 (ctx->device->coopmat_support && ctx->device->architecture != vk_device_architecture::INTEL_XE1));
    स्थिर ggml_type k_type_eff = (use_dequant_kv && !kv_raw_strided) ? GGML_TYPE_F16 : k->type;
    स्थिर ggml_type v_type_eff = (use_dequant_kv && !kv_raw_strided) ? GGML_TYPE_F16 : v->type;

    // For scalar/coopmat1 FA, we can use the "large" size to accommodate qga.
    // For coopmat2 FA, we always use the small size (which is still pretty large for gqa).
    vk_fa_tuning_params tuning_params = get_fa_tuning_params(ctx->device, HSK, HSV, 512, KV, k_type_eff, v_type_eff, f32acc);
    स्थिर uint32_t max_gqa = std::min(tuning_params.block_rows, 32u);

    अगर (N <= 8 && qk_ratio > 1 && qk_ratio <= max_gqa &&
        qk_ratio * nek2 == neq2 && nek2 == nev2 && nem2 <= 1) {
        // grouped query attention - make the N dimension equal to gqa_ratio, reduce
        // workgroups proportionally in y dimension. The shader will detect gqa_ratio > 1
        // and change addressing calculations to index Q's dimension 2.
        gqa_ratio = qk_ratio;
        N = gqa_ratio;
        workgroups_y /= gqa_ratio;
    }

    tuning_params = get_fa_tuning_params(ctx->device, HSK, HSV, N, KV, k_type_eff, v_type_eff, f32acc);

    भग्न scale         = 1.0f;
    भग्न max_bias      = 0.0f;
    भग्न logit_softcap = 0.0f;

    स_नकल(&scale,         (स्थिर भग्न *) dst->op_params + 0, माप(भग्न));
    स_नकल(&max_bias,      (स्थिर भग्न *) dst->op_params + 1, माप(भग्न));
    स_नकल(&logit_softcap, (स्थिर भग्न *) dst->op_params + 2, माप(भग्न));

    अगर (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    // Sparse mask hint (op_params[4]): compact the <= n_kv_max finite positions and gather only those.
    स्थिर int32_t n_kv_max = mask ? ggml_get_op_params_i32(dst, 4) : 0;
    अटल स्थिर bool disable_sparse = दो_पर्या("GGML_VK_FA_SPARSE_DISABLE") != nullptr;
    स्थिर bool kv_f16 = k_type_eff == GGML_TYPE_F16 && v_type_eff == GGML_TYPE_F16;
    // cm2 dense is fast, so it needs a larger reduction to win.
    // With quantized K/V, sparse only breaks even around 16x (measured on RDNA3/RDNA4).
    स्थिर int64_t min_ratio = tuning_params.path == FA_COOPMAT2 ? 4 : (kv_f16 ? 2 : 16);
    // coopmat2 vector decode requires 8B strides.
    स्वतः sparse_gather_aligned = [](स्थिर ggml_tensor * t) {
        वापस (t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_BF16) ||
               (t->nb[1] | t->nb[2] | t->nb[3]) % (4 * माप(ggml_fp16_t)) == 0;
    };
    स्थिर bool use_sparse = !disable_sparse && n_kv_max > 0 && mask &&
                            max_bias == 0.0f && logit_softcap == 0.0f &&
                            (tuning_params.path != FA_COOPMAT2 || (sparse_gather_aligned(k) && sparse_gather_aligned(v))) &&
                            nem0 == KV &&
                            (int64_t)KV >= std::max<int64_t>(4096, min_ratio * (int64_t)n_kv_max) &&
                            (gqa_ratio > 1 || (tuning_params.path == FA_SCALAR && N == 1));

    स्थिर uint32_t q_stride = (uint32_t)(nbq1 / ggml_type_आकार(q->type));
    uint32_t k_stride = (uint32_t)(nbk1 / ggml_type_आकार(k->type));
    uint32_t v_stride = (uint32_t)(nbv1 / ggml_type_आकार(v->type));

    // For F32, the shader treats it as a block of size 4 (for vec4 loads)
    अगर (k->type == GGML_TYPE_F32) {
        k_stride /= 4;
    }
    अगर (v->type == GGML_TYPE_F32) {
        v_stride /= 4;
    }

    uint32_t nbk2_eff = (uint32_t)nbk2, nbk3_eff = (uint32_t)nbk3;
    uint32_t nbv2_eff = (uint32_t)nbv2, nbv3_eff = (uint32_t)nbv3;
    अगर (use_dequant_kv) {
        k_stride = HSK;
        v_stride = HSV;
        nbk2_eff = (uint32_t)((uint64_t)HSK * KV * माप(ggml_fp16_t));
        nbk3_eff = (uint32_t)((uint64_t)HSK * KV * nek2 * माप(ggml_fp16_t));
        nbv2_eff = (uint32_t)((uint64_t)HSV * KV * माप(ggml_fp16_t));
        nbv3_eff = (uint32_t)((uint64_t)HSV * KV * nev2 * माप(ggml_fp16_t));
    }
    स्थिर uint32_t alignment = tuning_params.block_cols;
    bool aligned = (KV % alignment) == 0 &&
                   // the "aligned" shader variant will forcibly align strides, for performance
                   (q_stride & 7) == 0 && (k_stride & 7) == 0 && (v_stride & 7) == 0;

    // Need to use the coopmat2 variant that clamps loads when HSK/HSV aren't sufficiently aligned.
    अगर (((HSK | HSV) % 16) != 0 && tuning_params.path == FA_COOPMAT2) {
        aligned = false;
    }

    // Only use mask opt when the mask is fairly large. This hasn't been tuned extensively.
    bool use_mask_opt = mask && !use_sparse && nem1 >= 32 && nem0 * nem1 > 32768 && nem0 >= tuning_params.block_cols * 16
                        && (ctx->device->architecture != vk_device_architecture::AMD_GCN || HSK > 256 || HSV > 256);
    vk_fa_pipeline_state fa_pipeline_state = get_fa_pipeline_state(ctx->device, tuning_params, HSK, HSV, aligned, f32acc,
                                                                   mask != nullptr, use_mask_opt, logit_softcap != 0, use_sparse, k_type_eff, v_type_eff);

    vk_pipeline pipeline = nullptr;

    bool xe_fa_opt = false;
    bool fa_copy_qstate = false;
    bool xe_fa_supported_platform =
        (ctx->device.लो()->architecture == INTEL_XE2 && ctx->device.लो()->properties.deviceID != 0xFD80 && ctx->device.लो()->properties.deviceID != 0xFD81) ||
        (ctx->device.लो()->architecture == INTEL_XE1 && ctx->device.लो()->coopmat_support && ctx->device.लो()->uma);
    bool xe_fa_supported_usage = neq0 % 32 == 0 && nev0 % 16 == 0 && q->nb[1] > q->nb[2] && k->nb[1] > k->nb[2] && v->nb[1] > v->nb[2] && mask != nullptr;
    bool xe_fa_supported_dtype = q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16 && (mask != nullptr && mask->type == GGML_TYPE_F16);
    std::pair<vk_pipeline, vk_pipeline> xe_fa_pipeline_dual_phases = { nullptr , nullptr };
    vk_pipeline xe_fa_pipeline = nullptr;
    माप_प्रकार आकार_p = 0;
    माप_प्रकार आकार_group_max = 0;

    {
        std::lock_guard<std::mutex> guard(ctx->device->compile_mutex);
        स्वतः &pipelines = ctx->device->pipeline_flash_attn_f32_f16;
        स्वतः it = pipelines.find(fa_pipeline_state);
        अगर (it != pipelines.end()) {
            pipeline = it->second;
        } अन्यथा {
            pipelines[fa_pipeline_state] = pipeline = std::make_shared<vk_pipeline_struct>();
        }
    }

    निश्चित(pipeline);
    // Compile early to initialize wg_denoms.
    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    uint32_t split_kv = KV;
    uint32_t split_k = 1;

    // Intel Alchemist prefers more workgroups
    स्थिर uint32_t shader_core_गिनती_multiplier = (ctx->device->vendor_id == VK_VENDOR_ID_INTEL && ctx->device->architecture != INTEL_XE2) ? 2 : 1;

    // Use a placeholder core count if one isn't available. split_k is a big help for perf.
    स्थिर uint32_t shader_core_गिनती = ctx->device->shader_core_गिनती ? ctx->device->shader_core_गिनती * shader_core_गिनती_multiplier : 16;

    स्थिर uint32_t Br = fa_pipeline_state.Br;
    स्थिर uint32_t Bc = fa_pipeline_state.Bc;

    GGML_ASSERT(Br == pipeline->wg_denoms[0]);
    स्थिर uint32_t Tr = CEIL_DIV(N, Br);

    // Try to use split_k when KV is large enough to be worth the overhead.
    // Sparse: split_kv carries n_kv_max, split_k partitions its blocks for occupancy.
    अगर (use_sparse) {
        split_kv = (uint32_t)n_kv_max;
        स्थिर uint32_t योग_blocks = CEIL_DIV((uint32_t)n_kv_max, Bc);
        स्थिर uint32_t base_wgs = (gqa_ratio > 1 ? workgroups_x : Tr) * workgroups_y * workgroups_z;
        अगर (base_wgs < shader_core_गिनती * 2) {
            split_k = shader_core_गिनती * 2 / base_wgs;
        }
        split_k = std::max(1u, std::min(split_k, योग_blocks));
        // Match the shader's per-split block count so no split is empty.
        स्थिर uint32_t per_blocks = CEIL_DIV(योग_blocks, split_k);
        split_k = CEIL_DIV(योग_blocks, per_blocks);
    } अन्यथा अगर (gqa_ratio > 1 && workgroups_x <= Br) {
        split_k = shader_core_गिनती * 2 / (workgroups_x * workgroups_y * workgroups_z);
    } अन्यथा अगर (gqa_ratio <= 1) {
        uint32_t योग_wgs_no_split = Tr * workgroups_y * workgroups_z;
        अगर (योग_wgs_no_split < shader_core_गिनती * 2) {
            split_k = shader_core_गिनती * 2 / योग_wgs_no_split;
        }
    }

    अगर (!use_sparse && split_k > 1) {
        // Try to evenly split KV into split_k chunks, but it needs to be a multiple
        // of "align", so recompute split_k based on that.
        split_kv = ROUNDUP_POW2(std::max(1u, KV / split_k), alignment);
        split_k = CEIL_DIV(KV, split_kv);
        xe_fa_opt = xe_fa_supported_platform && xe_fa_supported_usage && xe_fa_supported_dtype;
        अगर (xe_fa_opt) {
            स्थिर uint32_t split_p_आकार = 32;
            स्थिर माप_प्रकार max_dim = (nek1 + split_p_आकार - 1) / split_p_आकार;
            स्थिर माप_प्रकार p_dim = max_dim * split_p_आकार;
#अगर defined(VK_KHR_cooperative_matrix) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
            std::lock_guard<std::mutex> guard(ctx->device->compile_mutex);
            स्वतः& pipelines = ctx->device->pipeline_xe_fa_decode_dual_phases;
            स्वतः it = pipelines.find({ (uint32_t)neq0, (uint32_t)nev0, qk_ratio, (uint32_t)neq1 });
            अगर (it != pipelines.end()) {
                xe_fa_pipeline_dual_phases = it->second;
            } अन्यथा {
                pipelines[{(uint32_t)neq0, (uint32_t)nev0, qk_ratio, (uint32_t)neq1}] = xe_fa_pipeline_dual_phases = std::make_pair(std::make_shared<vk_pipeline_struct>(), std::make_shared<vk_pipeline_struct>());
            }
#पूर्ण_अगर
            अगर (xe_fa_pipeline_dual_phases.first == nullptr || xe_fa_pipeline_dual_phases.second == nullptr) {
                xe_fa_opt = false;
                fa_copy_qstate = false;
            } अन्यथा {
                आकार_p = neq1 * neq2 * p_dim * neq3 * माप(ggml_fp16_t);
                आकार_group_max = neq1 * neq2 * max_dim * neq3 * माप(भग्न);
                माप_प्रकार अस्थायी_आकार = ggml_nelements(q) * माप(ggml_fp16_t) + आकार_p + आकार_group_max;
                fa_copy_qstate = true;
                अगर (ctx->prealloc_आकार_x < अस्थायी_आकार) {
                    ctx->prealloc_आकार_x = अस्थायी_आकार;
                    ggml_vk_preallocate_buffers(ctx, subctx);
                }

                अगर (ctx->prealloc_x_need_sync) {
                    ggml_vk_sync_buffers(ctx, subctx);
                }
            }
        }
    }

    अगर (xe_fa_opt == true) {
        use_mask_opt = false;
    }

    // Reserve space for split_k temporaries. For each split x batch, we need to store the O matrix (D x ne1)
    // and the per-row m and L values (ne1 rows). We store all the matrices first, followed by the rows.
    // For matrices, the order is (inner to outer) [HSV, ne1, k, ne2, ne3].
    // For L/M, the order is (inner to outer) [ne1, k, ne2, ne3].
    स्थिर uint64_t split_k_आकार = split_k > 1 ? (HSV * ne1 * माप(भग्न) + ne1 * माप(भग्न) * 2) * split_k * ne2 * ne3 : 0;
    अगर (split_k_आकार > ctx->device->properties.limits.max_Storage_बफर_Range) {
        GGML_ABORT("Requested preallocation size is too large");
    }
    अगर (ctx->prealloc_आकार_split_k < split_k_आकार) {
        ctx->prealloc_आकार_split_k = split_k_आकार;
        ggml_vk_preallocate_buffers(ctx, subctx);
    }

    स्थिर uint32_t mask_opt_अंक_dwords = CEIL_DIV(nem0, 16 * Bc);
    स्थिर uint64_t mask_opt_आकार = माप(uint32_t) * mask_opt_अंक_dwords * CEIL_DIV(nem1, Br) * nem2 * nem3;

    vk_pipeline pipeline_fa_mask_opt = nullptr;
    अगर (use_mask_opt) {
        {
            std::lock_guard<std::mutex> guard(ctx->device->compile_mutex);
            स्वतः &pipelines = ctx->device->pipeline_fa_mask_opt;
            स्वतः it = pipelines.find({Br, Bc});
            अगर (it != pipelines.end()) {
                pipeline_fa_mask_opt = it->second;
            } अन्यथा {
                pipelines[{Br, Bc}] = pipeline_fa_mask_opt = std::make_shared<vk_pipeline_struct>();
            }
        }
        निश्चित(pipeline_fa_mask_opt);
        ggml_pipeline_request_descriptor_sets(ctx, pipeline_fa_mask_opt, 1);

        अगर (ctx->prealloc_आकार_y < mask_opt_आकार) {
            ctx->prealloc_आकार_y = mask_opt_आकार;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        अगर (ctx->prealloc_y_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }
    }

    // Sparse index scratch reuses prealloc_y (mutually exclusive with mask opt).
    स्थिर uint64_t sparse_idx_आकार = use_sparse
        ? माप(int32_t) * (uint64_t)n_kv_max * nem1 * nem2 * nem3
        : 0;
    vk_pipeline sparse_compact_pipeline = ctx->device->fa_sparse_compact_use_subgroups
        ? ctx->device->pipeline_fa_sparse_compact_subgroup
        : ctx->device->pipeline_fa_sparse_compact;
    अगर (use_sparse) {
        ggml_pipeline_request_descriptor_sets(ctx, sparse_compact_pipeline, 1);
        अगर (ctx->prealloc_आकार_y < sparse_idx_आकार) {
            ctx->prealloc_आकार_y = sparse_idx_आकार;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        अगर (ctx->prealloc_y_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }
    }

    स्थिर uint32_t n_शीर्ष_kv   = neq2;
    स्थिर uint32_t n_शीर्ष_log2 = 1u << (uint32_t) floorf(log2f((भग्न) n_शीर्ष_kv));
    स्थिर भग्न m0 = powf(2.0f, -(max_bias       ) / n_शीर्ष_log2);
    स्थिर भग्न m1 = powf(2.0f, -(max_bias / 2.0f) / n_शीर्ष_log2);

    vk_subbuffer q_buf = ggml_vk_tensor_subbuffer(ctx, q);
    vk_subbuffer k_buf = ggml_vk_tensor_subbuffer(ctx, k);
    vk_subbuffer v_buf = ggml_vk_tensor_subbuffer(ctx, v);
    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
    vk_subbuffer mask_buf = mask ? ggml_vk_tensor_subbuffer(ctx, mask) : q_buf;
    vk_subbuffer sinks_buf = sinks ? ggml_vk_tensor_subbuffer(ctx, sinks) : q_buf;
    vk_subbuffer mask_opt_buf = use_mask_opt ? ggml_vk_subbuffer(ctx, ctx->prealloc_y, 0) : q_buf;
    vk_subbuffer sparse_buf = use_sparse ? ggml_vk_subbuffer(ctx, ctx->prealloc_y, 0) : q_buf;
    अगर (use_mask_opt || use_sparse) {
        // the mask opt bits and the sparse index list overwrite a matmul input converted into prealloc_y
        ctx->prealloc_y_last_pipeline_used = nullptr;
        ctx->prealloc_y_last_tensor_used = nullptr;
    }

    अगर (use_dequant_kv) {
        स्थिर uint64_t k_sz = (uint64_t)ggml_nelements(k) * kv_scratch_ts;
        स्थिर uint64_t v_sz = (uint64_t)ggml_nelements(v) * kv_scratch_ts;
        अगर (ctx->prealloc_आकार_x < k_sz + v_sz) {
            ctx->prealloc_आकार_x = k_sz + v_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        vk_subbuffer k_dst = vk_subbuffer{ ctx->prealloc_x, 0,    k_sz };
        vk_subbuffer v_dst = vk_subbuffer{ ctx->prealloc_x, k_sz, v_sz };
        स्थिर uint32_t k_nel = (uint32_t)ggml_nelements(k);
        स्थिर uint32_t v_nel = (uint32_t)ggml_nelements(v);
        अगर (kv_raw_strided) {
            vk_pipeline cp_k = ggml_vk_get_cpy_pipeline(ctx, k, nullptr, k->type);
            vk_pipeline cp_v = ggml_vk_get_cpy_pipeline(ctx, v, nullptr, v->type);
            ggml_pipeline_request_descriptor_sets(ctx, cp_k, 1);
            ggml_pipeline_request_descriptor_sets(ctx, cp_v, 1);
            अगर (ctx->prealloc_x_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            स्वतः make_pc = [](uint32_t hs, uint32_t nh, uint32_t kv, uint32_t nel) {
                vk_op_unary_धकेलो_constants pc{};
                pc.ne = nel;
                pc.ne00 = hs; pc.ne01 = nh;      pc.ne02 = kv; pc.ne03 = nel / (hs * nh * kv);
                pc.nb00 = 1;  pc.nb01 = hs;      pc.nb02 = hs * nh; pc.nb03 = hs * nh * kv;
                pc.ne10 = hs; pc.ne11 = nh;      pc.ne12 = kv; pc.ne13 = pc.ne03;
                pc.nb10 = 1;  pc.nb11 = hs * kv; pc.nb12 = hs; pc.nb13 = hs * kv * nh;
                init_pushconst_fastdiv(pc);
                वापस pc;
            };
            स्वतः cpy_elems = [](uint32_t ne) -> std::array<uint32_t, 3> {
                अगर (ne > 262144) { वापस { 512, 512, CEIL_DIV(ne, 262144) }; }
                अगर (ne > 512)    { वापस { 512, CEIL_DIV(ne, 512), 1 }; }
                वापस { ne, 1, 1 };
            };
            ggml_vk_dispatch_pipeline(ctx, subctx, cp_k, { k_buf, k_dst },
                make_pc((uint32_t)HSK, (uint32_t)nek2, (uint32_t)KV, k_nel), cpy_elems(k_nel));
            ggml_vk_dispatch_pipeline(ctx, subctx, cp_v, { v_buf, v_dst },
                make_pc((uint32_t)HSV, (uint32_t)nev2, (uint32_t)KV, v_nel), cpy_elems(v_nel));
        } अन्यथा {
            vk_pipeline tr_k = ctx->device->pipeline_dequant_transpose[k->type];
            vk_pipeline tr_v = ctx->device->pipeline_dequant_transpose[v->type];
            ggml_pipeline_request_descriptor_sets(ctx, tr_k, 1);
            ggml_pipeline_request_descriptor_sets(ctx, tr_v, 1);
            अगर (ctx->prealloc_x_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            { स्थिर std::vector<uint32_t> pc = { (uint32_t)HSK, (uint32_t)nek2, (uint32_t)KV, 0, k_nel };
              ggml_vk_dispatch_pipeline(ctx, subctx, tr_k, { k_buf, k_dst }, pc, { k_nel, 1, 1 }); }
            { स्थिर std::vector<uint32_t> pc = { (uint32_t)HSV, (uint32_t)nev2, (uint32_t)KV, 0, v_nel };
              ggml_vk_dispatch_pipeline(ctx, subctx, tr_v, { v_buf, v_dst }, pc, { v_nel, 1, 1 }); }
        }
        ggml_vk_sync_buffers(ctx, subctx);
        k_buf = k_dst;
        v_buf = v_dst;
    }

    uint32_t mask_n_शीर्ष_log2 = ((sinks != nullptr) << 24) | n_शीर्ष_log2;

    अगर (use_mask_opt)
    {
        स्थिर vk_op_flash_attn_mask_opt_धकेलो_constants opt_pc = {
            nem0,
            nem1,
            nem2,
            (uint32_t)(mask->nb[1] / माप(ggml_fp16_t)),
            (uint32_t)(mask->nb[2] / माप(ggml_fp16_t)),
            (uint32_t)(mask->nb[3] / माप(ggml_fp16_t)),
            mask_opt_अंक_dwords,
            mask_opt_अंक_dwords * CEIL_DIV(nem1, Br),
            mask_opt_अंक_dwords * CEIL_DIV(nem1, Br) * nem2,
        };

        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline_fa_mask_opt,
                                  { mask_buf, mask_opt_buf }, opt_pc,
                                  { mask_opt_अंक_dwords, CEIL_DIV(nem1, Br), nem2 * nem3 });
        ggml_vk_sync_buffers(ctx, subctx);
    }

    अगर (use_sparse)
    {
        स्थिर vk_op_flash_attn_sparse_compact_धकेलो_constants sc_pc = {
            KV,
            nem1,
            nem2,
            (uint32_t)(mask->nb[1] / माप(ggml_fp16_t)),
            (uint32_t)(mask->nb[2] / माप(ggml_fp16_t)),
            (uint32_t)(mask->nb[3] / माप(ggml_fp16_t)),
            (uint32_t)n_kv_max,
        };

        ggml_vk_dispatch_pipeline(ctx, subctx, sparse_compact_pipeline,
                                  { mask_buf, sparse_buf }, sc_pc,
                                  { nem1, nem2, nem3 });
        ggml_vk_sync_buffers(ctx, subctx);
    }

    स्थिर vk_flash_attn_धकेलो_constants pc = { N, KV,
                                              (uint32_t)ne1, (uint32_t)ne2, (uint32_t)ne3,
                                              (uint32_t)neq2, (uint32_t)neq3,
                                              (uint32_t)nek2, (uint32_t)nek3,
                                              (uint32_t)nev2, (uint32_t)nev3,
                                              nem1, nem2, nem3,
                                              q_stride, (uint32_t)nbq2, (uint32_t)nbq3,
                                              k_stride, nbk2_eff, nbk3_eff,
                                              v_stride, nbv2_eff, nbv3_eff,
                                              scale, max_bias, logit_softcap,
                                              mask_n_शीर्ष_log2, m0, m1,
                                              gqa_ratio, split_kv, split_k };

    अगर (xe_fa_opt && split_k > 1) {
        स्वतः upper_power_of_2 = [&](uint32_t in) {
            GGML_ASSERT(in != 0);
            अगर (in <= 1) वापस 1u;
            uint32_t ret = in - 1;
            ret |= ret >> 1;
            ret |= ret >> 2;
            ret |= ret >> 4;
            ret |= ret >> 8;
            ret |= ret >> 16;
            वापस ret + 1;
        };
        स्वतः to_fp16_vk_0 = ggml_vk_get_to_fp16(ctx, q->type);
        स्थिर uint32_t out_dim_per_wg = qk_ratio > 16 ? 8 : 16;
        माप_प्रकार x_ne = ggml_nelements(q);
        माप_प्रकार अस्थायी_buf_offset = 0;
        uint32_t शीर्ष_stride_k = uint32_t(nbk2 / ggml_type_आकार(k->type));
        uint32_t शीर्ष_stride_v = uint32_t(nbv2 / ggml_type_आकार(v->type));
        uint32_t batch_stride_q = uint32_t(nbq3 / ggml_type_आकार(q->type));
        uint32_t batch_stride_k = uint32_t(nbk3 / ggml_type_आकार(k->type));
        uint32_t batch_stride_v = uint32_t(nbv3 / ggml_type_आकार(v->type));
        uint32_t batch_stride_m = mask ? uint32_t(mask->nb[3] / ggml_type_आकार(mask->type)) : 0u;
        uint32_t batch_stride_o = uint32_t(nb3 / ggml_type_आकार(dst->type));
        vk_fa_xe_opt_धकेलो_constants pc_ph1 = { (uint32_t)nek1, (uint32_t)neq1, (uint32_t)neq2, (uint32_t)nek2, qk_ratio, 1, (sinks != nullptr) ? 1u : 0u, (uint32_t)k_stride, शीर्ष_stride_k,
            batch_stride_q, batch_stride_k, batch_stride_v, batch_stride_m, batch_stride_o, scale };
        vk_fa_xe_opt_धकेलो_constants pc_ph2 = pc_ph1;
        pc_ph2.nbkv_tok = v_stride;
        pc_ph2.nbkv_शीर्ष = शीर्ष_stride_v;
        vk_subbuffer q_अस्थायी_buf = fa_copy_qstate ? ggml_vk_subbuffer(ctx, ctx->prealloc_x, अस्थायी_buf_offset) : q_buf;
        अस्थायी_buf_offset += fa_copy_qstate ? x_ne * माप(ggml_fp16_t) : 0;
        vk_subbuffer p_अस्थायी_buf = ggml_vk_subbuffer(ctx, ctx->prealloc_x, अस्थायी_buf_offset);
        अस्थायी_buf_offset += आकार_p;
        vk_subbuffer max_अस्थायी_buf = ggml_vk_subbuffer(ctx, ctx->prealloc_x, अस्थायी_buf_offset);
        अस्थायी_buf_offset += आकार_group_max;
        uint32_t xe_native_sub_group_आकार = ctx->device.लो()->architecture == INTEL_XE1 ? 8 : 16;
        uint32_t aligned_gqa_ratio = upper_power_of_2(qk_ratio);
        uint32_t out_per_wg_ph1 = std::min(256u / xe_native_sub_group_आकार, (uint32_t)neq1);
        uint32_t out_per_wg_ph2 = std::min(std::max(16u / aligned_gqa_ratio, 1u), (uint32_t)neq1);
        uint32_t ph1_wg = ((neq1 + out_per_wg_ph1 - 1) / out_per_wg_ph1) * nek2;
        uint32_t ph2_wg = ((neq1 + out_per_wg_ph2 - 1) / out_per_wg_ph2) * ne0 / out_dim_per_wg;
        अगर (fa_copy_qstate) {
            स्थिर std::vector<uint32_t> pc_cpy_fp16 =
            { (uint32_t)q->ne[0], (uint32_t)q->ne[1], (uint32_t)q->ne[2], (uint32_t)q->ne[3], (uint32_t)(x_ne) };
            ggml_vk_sync_buffers(ctx, subctx);
            ggml_pipeline_request_descriptor_sets(ctx, to_fp16_vk_0, 1);
            ggml_vk_dispatch_pipeline(ctx, subctx, to_fp16_vk_0, { q_buf, q_अस्थायी_buf }, pc_cpy_fp16, { (uint32_t)(x_ne), 1, 1 });
        }

        ggml_vk_sync_buffers(ctx, subctx);
        ggml_pipeline_request_descriptor_sets(ctx, xe_fa_pipeline_dual_phases.first, 1);
        ggml_vk_dispatch_pipeline(ctx, subctx, xe_fa_pipeline_dual_phases.first,
            { q_अस्थायी_buf, k_buf, mask_buf, p_अस्थायी_buf, max_अस्थायी_buf },
            pc_ph1, { (uint32_t)ph1_wg, (uint32_t)nek1, (uint32_t)neq3 });

        ggml_vk_sync_buffers(ctx, subctx);
        ggml_pipeline_request_descriptor_sets(ctx, xe_fa_pipeline_dual_phases.second, 1);
        ggml_vk_dispatch_pipeline(ctx, subctx, xe_fa_pipeline_dual_phases.second,
            { p_अस्थायी_buf, v_buf, max_अस्थायी_buf, sinks_buf, dst_buf },
            pc_ph2, { (uint32_t)ph2_wg, (uint32_t)nev2, (uint32_t)neq3 });

        ctx->prealloc_x_need_sync = true;
    } अन्यथा अगर (split_k > 1) {
        ggml_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_flash_attn_split_k_reduce, 1);

        अगर (ctx->prealloc_split_k_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }

        // We reuse workgroups_x to mean the number of splits, so we need to
        // cancel out the divide by wg_denoms[0].
        uint32_t dispatch_x;
        अगर (gqa_ratio > 1) {
            workgroups_x *= pipeline->wg_denoms[0];
            dispatch_x = split_k * workgroups_x;
        } अन्यथा {
            dispatch_x = Tr * split_k * pipeline->wg_denoms[0];
        }

        vk_subbuffer split_k_buf = ggml_vk_subbuffer(ctx, ctx->prealloc_split_k, 0);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
                                    {q_buf, k_buf, v_buf, mask_buf, sinks_buf, split_k_buf, mask_opt_buf, sparse_buf},
                                    pc, { dispatch_x, workgroups_y, workgroups_z });

        ggml_vk_sync_buffers(ctx, subctx);
        स्थिर vk_op_flash_attn_split_k_reduce_धकेलो_constants pc2 = { HSV, (uint32_t)ne1, (uint32_t)ne2, (uint32_t)ne3, split_k, (sinks != nullptr) };
        ggml_vk_dispatch_pipeline(ctx, subctx, ctx->device->pipeline_flash_attn_split_k_reduce,
                                    {split_k_buf, sinks_buf, dst_buf},
                                    pc2, { (uint32_t)ne1, HSV, (uint32_t)(ne2 * ne3) });
        ctx->prealloc_split_k_need_sync = true;
    } अन्यथा {
        अगर (gqa_ratio > 1) {
            // When using gqa, we want one actual workgroup per batch, so cancel out wg_denoms
            workgroups_x *= pipeline->wg_denoms[0];
        }
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
                                    {q_buf, k_buf, v_buf, mask_buf, sinks_buf, dst_buf, mask_opt_buf, sparse_buf},
                                    pc, { workgroups_x, workgroups_y, workgroups_z });
    }

    अगर (use_dequant_kv) {
        ctx->prealloc_x_need_sync = true;
    }
    अगर (use_mask_opt || use_sparse) {
        ctx->prealloc_y_need_sync = true;
    }
}

अटल vk_conv_shapes ggml_vk_conv_select_shape(ggml_backend_vk_context * ctx, uint32_t K, uint32_t NPQ) {
    स्वतः n_tiles = [&](vk_conv_shapes s) {
        वापस CEIL_DIV(K, vk_conv_block_sizes[s].K)
            * CEIL_DIV(NPQ, vk_conv_block_sizes[s].NPQ);
    };

    // We can't query number of shader cores on Intel, use 32 as a placeholder
    // so small convolutions will still choose a smaller tile.
    स्थिर uint32_t shader_core_गिनती = ctx->device->shader_core_गिनती > 0 ? ctx->device->shader_core_गिनती : 32;

    // 128x128 isn't used with cm1 due to shared memory size; fall through to a smaller tile.
    bool allow_128x128 = true;
#अगर defined(VK_KHR_cooperative_matrix) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    अगर (!ctx->device->coopmat2 && ctx->device->coopmat_support && ctx->device->coopmat_support_16x16x16_f16acc) {
        allow_128x128 = false;
    }
#पूर्ण_अगर

    अगर (allow_128x128 && K > 64 && n_tiles(CONV_SHAPE_128x128) >= shader_core_गिनती * 2) {
        वापस CONV_SHAPE_128x128;
    } अन्यथा अगर (K <= 32 && n_tiles(CONV_SHAPE_32x256) >= shader_core_गिनती * 2) {
        वापस CONV_SHAPE_32x256;
    } अन्यथा अगर (K <= 64 && n_tiles(CONV_SHAPE_64x128) >= shader_core_गिनती * 2) {
        वापस CONV_SHAPE_64x128;
    } अन्यथा अगर (!allow_128x128 && K > 64 && n_tiles(CONV_SHAPE_64x128) >= shader_core_गिनती * 2) {
        // cm1 fallback for large K when 128x128 isn't available
        वापस CONV_SHAPE_64x128;
    } अन्यथा {
        वापस CONV_SHAPE_64x32;
    }
}

अटल vk_pipeline ggml_vk_op_get_pipeline(ggml_backend_vk_context * ctx, स्थिर ggml_tensor * src0, स्थिर ggml_tensor * src1, स्थिर ggml_tensor * src2, स्थिर ggml_tensor * dst, ggml_op op) {
    चयन (op) {
    हाल GGML_OP_GET_ROWS:
        GGML_ASSERT(src1->type == GGML_TYPE_I32);
        अगर (src0->type == GGML_TYPE_I32) {
            // i32 src only supports i32 result
            GGML_ASSERT(dst->type == GGML_TYPE_I32);
            वापस ctx->device->pipeline_get_rows[src0->type];
        }
        अगर (dst->type == GGML_TYPE_F16) {
            वापस ctx->device->pipeline_get_rows[src0->type];
        }
        अगर (dst->type == GGML_TYPE_F32) {
            वापस ctx->device->pipeline_get_rows_f32[src0->type];
        }
        वापस nullptr;
    हाल GGML_OP_GET_ROWS_BACK:
        अगर (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32) {
            वापस ctx->device->pipeline_get_rows_back_f32;
        }
        वापस nullptr;
    हाल GGML_OP_ACC:
        अगर (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            वापस ctx->device->pipeline_acc_f32;
        }
        वापस nullptr;
    हाल GGML_OP_SET:
        अगर (src0->type == src1->type && src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_I32)) {
            वापस ctx->device->pipeline_set_f32;
        }
        वापस nullptr;
    हाल GGML_OP_ADD:
    हाल GGML_OP_SUB:
    हाल GGML_OP_MUL:
    हाल GGML_OP_DIV:
        अगर ((src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) ||
            (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) ||
            (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16)) {
            वापस nullptr;
        }
        चयन (op) {
        हाल GGML_OP_ADD:
        {
            अगर (ctx->अंक_additional_fused_ops > 0) {
                अगर (ctx->do_add_rms_partials) {
                    वापस ctx->device->pipeline_multi_add_rms[ctx->अंक_additional_fused_ops];
                } अन्यथा {
                    वापस ctx->device->pipeline_multi_add[ctx->अंक_additional_fused_ops];
                }
            }
            अगर (ctx->do_add_rms_partials) {
                स्वतः pipelines = ggml_are_same_shape(src0, src1) ? ctx->device->pipeline_add_rms_norepeat : ctx->device->pipeline_add_rms;
                वापस pipelines[src0->type == GGML_TYPE_F16][src1->type == GGML_TYPE_F16][dst->type == GGML_TYPE_F16];
            } अन्यथा {
                स्वतः pipelines = ggml_are_same_shape(src0, src1) ? ctx->device->pipeline_add_norepeat : ctx->device->pipeline_add;
                वापस pipelines[src0->type == GGML_TYPE_F16][src1->type == GGML_TYPE_F16][dst->type == GGML_TYPE_F16];
            }
        }
        हाल GGML_OP_SUB:
        {
            स्वतः pipelines = ggml_are_same_shape(src0, src1) ? ctx->device->pipeline_sub_norepeat : ctx->device->pipeline_sub;
            वापस pipelines[src0->type == GGML_TYPE_F16][src1->type == GGML_TYPE_F16][dst->type == GGML_TYPE_F16];
        }
        हाल GGML_OP_MUL:
        {
            स्वतः pipelines = ggml_are_same_shape(src0, src1) ? ctx->device->pipeline_mul_norepeat : ctx->device->pipeline_mul;
            वापस pipelines[src0->type == GGML_TYPE_F16][src1->type == GGML_TYPE_F16][dst->type == GGML_TYPE_F16];
        }
        हाल GGML_OP_DIV:
        {
            स्वतः pipelines = ggml_are_same_shape(src0, src1) ? ctx->device->pipeline_div_norepeat : ctx->device->pipeline_div;
            वापस pipelines[src0->type == GGML_TYPE_F16][src1->type == GGML_TYPE_F16][dst->type == GGML_TYPE_F16];
        }
        शेष:
            अवरोध;
        }
        वापस nullptr;
    हाल GGML_OP_ADD_ID:
        अगर (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && src2->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32) {
            वापस ctx->device->pipeline_add_id_f32;
        }
        वापस nullptr;
    हाल GGML_OP_OUT_PROD:
        अगर (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            वापस ctx->device->pipeline_out_prod_f32;
        }
        वापस nullptr;
    हाल GGML_OP_CONCAT: {
        अगर (!ggml_vk_concat_supported(src0, src1, dst)) {
            वापस nullptr;
        }
        चयन (ggml_vk_concat_unit_आकार(src0->type)) {
        हाल 1:
            वापस ctx->device->pipeline_concat_i8;
        हाल 2:
            वापस ctx->device->pipeline_concat_i16;
        हाल 4:
            वापस ctx->device->pipeline_concat_i32;
        हाल 8:
            वापस ctx->device->pipeline_concat_i64;
        शेष:
            वापस nullptr;
        }
    }
    हाल GGML_OP_UPSCALE:
        अगर (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            uint32_t mode = (ggml_get_op_params_i32(dst, 0) & (0xFF | GGML_SCALE_FLAG_ANTIALIAS));
            चयन (mode) {
                हाल GGML_SCALE_MODE_NEAREST:
                    वापस ctx->device->pipeline_upscale_nearest_f32;
                हाल GGML_SCALE_MODE_BILINEAR:
                    वापस ctx->device->pipeline_upscale_bilinear_f32;
                हाल GGML_SCALE_MODE_BICUBIC:
                    वापस ctx->device->pipeline_upscale_bicubic_f32;
                हाल GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ANTIALIAS:
                    वापस ctx->device->pipeline_upscale_bilinear_antialias_f32;
                शेष:
                    वापस nullptr;
            }
        }
        वापस nullptr;
    हाल GGML_OP_SCALE:
        अगर (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            वापस ctx->device->pipeline_scale_f32;
        }
        वापस nullptr;
    हाल GGML_OP_SQR:
        अगर (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            वापस ctx->device->pipeline_sqr[dst->type == GGML_TYPE_F16];
        }
        वापस nullptr;
    हाल GGML_OP_SQRT:
        अगर (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            वापस ctx->device->pipeline_sqrt[dst->type == GGML_TYPE_F16];
        }
        वापस nullptr;
    हाल GGML_OP_SIN:
        अगर (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            वापस ctx->device->pipeline_sin[dst->type == GGML_TYPE_F16];
        }
        वापस nullptr;
    हाल GGML_OP_COS:
        अगर (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            वापस ctx->device->pipeline_cos[dst->type == GGML_TYPE_F16];
        }
        वापस nullptr;
    हाल GGML_OP_LOG:
        अगर (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            वापस ctx->device->pipeline_log[dst->type == GGML_TYPE_F16];
        }
        वापस nullptr;
    हाल GGML_OP_TRI:
        अगर (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            वापस ctx->device->pipeline_tri[dst->type == GGML_TYPE_F16];
        }
        वापस nullptr;
    हाल GGML_OP_DIAG:
        अगर (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            वापस ctx->device->pipeline_diag[dst->type == GGML_TYPE_F16];
        }
        वापस nullptr;
    हाल GGML_OP_CLAMP:
        अगर (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            वापस ctx->device->pipeline_clamp[dst->type == GGML_TYPE_F16];
        }
        वापस nullptr;
    हाल GGML_OP_PAD:
        अगर (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            वापस ctx->device->pipeline_pad_f32;
        }
        वापस nullptr;
    हाल GGML_OP_PAD_REFLECT_1D:
        अगर (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            वापस ctx->device->pipeline_pad_reflect_1d_f32;
        }
        वापस nullptr;
    हाल GGML_OP_ROLL:
        अगर (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            वापस ctx->device->pipeline_roll_f32;
        }
        वापस nullptr;
    हाल GGML_OP_REPEAT:
        अगर (ggml_type_आकार(src0->type) == माप(भग्न) && ggml_type_आकार(dst->type) == माप(भग्न)) {
            वापस ctx->device->pipeline_repeat_i32;
        }
        अगर (ggml_type_आकार(src0->type) == 2 && ggml_type_आकार(dst->type) == 2) {
            वापस ctx->device->pipeline_repeat_i16;
        }
        वापस nullptr;
    हाल GGML_OP_REPEAT_BACK:
        अगर (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            वापस ctx->device->pipeline_repeat_back_f32;
        }
        वापस nullptr;
    हाल GGML_OP_CPY:
    हाल GGML_OP_CONT:
    हाल GGML_OP_DUP:
        वापस ggml_vk_get_cpy_pipeline(ctx, src0, dst, dst->type);
    हाल GGML_OP_SET_ROWS:
        {
            अगर (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) {
                वापस nullptr;
            }
            स्थिर पूर्णांक src_idx = src0->type == GGML_TYPE_F16;
            अगर (src1->type == GGML_TYPE_I64) {
                वापस ctx->device->pipeline_set_rows_i64[src_idx][dst->type];
            } अन्यथा अगर (src1->type == GGML_TYPE_I32) {
                वापस ctx->device->pipeline_set_rows_i32[src_idx][dst->type];
            }
            वापस nullptr;
        }
    हाल GGML_OP_SILU_BACK:
        अगर (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            वापस ctx->device->pipeline_silu_back_f32;
        }
        वापस nullptr;
    हाल GGML_OP_NORM:
        अगर (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            वापस ctx->device->pipeline_norm_f32;
        }
        वापस nullptr;
    हाल GGML_OP_GROUP_NORM:
        अगर (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            वापस ctx->device->pipeline_group_norm_f32;
        }
        वापस nullptr;
    हाल GGML_OP_RMS_NORM:
        अगर (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            अगर (ctx->do_add_rms_partials) {
                वापस ctx->fused_rms_norm_mode == RMS_NORM_MUL ? ctx->device->pipeline_rms_norm_mul_partials_f32 : ctx->device->pipeline_rms_norm_partials_f32;
            }
            वापस ctx->fused_rms_norm_mode == RMS_NORM_MUL ? ctx->device->pipeline_rms_norm_mul_f32 : ctx->device->pipeline_rms_norm_f32;
        }
        वापस nullptr;
    हाल GGML_OP_RMS_NORM_BACK:
        अगर (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            वापस ctx->device->pipeline_rms_norm_back_f32;
        }
        वापस nullptr;
    हाल GGML_OP_L2_NORM:
        अगर (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            वापस ctx->device->pipeline_l2_norm_f32;
        }
        वापस nullptr;
    हाल GGML_OP_UNARY:
        अगर ((src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) ||
            (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) ||
            (src0->type != dst->type)) {
            वापस nullptr;
        }

        चयन (ggml_get_unary_op(dst)) {
            हाल GGML_UNARY_OP_EXP:
                वापस ctx->device->pipeline_exp[dst->type == GGML_TYPE_F16];
            हाल GGML_UNARY_OP_EXPM1:
                वापस ctx->device->pipeline_expm1[dst->type == GGML_TYPE_F16];
            हाल GGML_UNARY_OP_ELU:
                वापस ctx->device->pipeline_elu[dst->type == GGML_TYPE_F16];
            हाल GGML_UNARY_OP_SILU:
                वापस ctx->device->pipeline_silu[dst->type == GGML_TYPE_F16];
            हाल GGML_UNARY_OP_GELU:
                वापस ctx->device->pipeline_gelu[dst->type == GGML_TYPE_F16];
            हाल GGML_UNARY_OP_GELU_ERF:
                वापस ctx->device->pipeline_gelu_erf[dst->type == GGML_TYPE_F16];
            हाल GGML_UNARY_OP_GELU_QUICK:
                वापस ctx->device->pipeline_gelu_quick[dst->type == GGML_TYPE_F16];
            हाल GGML_UNARY_OP_RELU:
                वापस ctx->device->pipeline_relu[dst->type == GGML_TYPE_F16];
            हाल GGML_UNARY_OP_XIELU:
                वापस ctx->device->pipeline_xielu[dst->type == GGML_TYPE_F16];
            हाल GGML_UNARY_OP_NEG:
                वापस ctx->device->pipeline_neg[dst->type == GGML_TYPE_F16];
            हाल GGML_UNARY_OP_TANH:
                वापस ctx->device->pipeline_tanh[dst->type == GGML_TYPE_F16];
            हाल GGML_UNARY_OP_SIGMOID:
                वापस ctx->device->pipeline_sigmoid[dst->type == GGML_TYPE_F16];
            हाल GGML_UNARY_OP_HARDSIGMOID:
                वापस ctx->device->pipeline_hardsigmoid[dst->type == GGML_TYPE_F16];
            हाल GGML_UNARY_OP_HARDSWISH:
                वापस ctx->device->pipeline_hardswish[dst->type == GGML_TYPE_F16];
            हाल GGML_UNARY_OP_ABS:
                वापस ctx->device->pipeline_abs[dst->type == GGML_TYPE_F16];
            हाल GGML_UNARY_OP_SOFTPLUS:
                वापस ctx->device->pipeline_softplus[dst->type == GGML_TYPE_F16];
            हाल GGML_UNARY_OP_STEP:
                वापस ctx->device->pipeline_step[dst->type == GGML_TYPE_F16];
            हाल GGML_UNARY_OP_ROUND:
                वापस ctx->device->pipeline_round[dst->type == GGML_TYPE_F16];
            हाल GGML_UNARY_OP_CEIL:
                वापस ctx->device->pipeline_ceil[dst->type == GGML_TYPE_F16];
            हाल GGML_UNARY_OP_FLOOR:
                वापस ctx->device->pipeline_floor[dst->type == GGML_TYPE_F16];
            हाल GGML_UNARY_OP_TRUNC:
                वापस ctx->device->pipeline_trunc[dst->type == GGML_TYPE_F16];
            हाल GGML_UNARY_OP_SGN:
                वापस ctx->device->pipeline_sgn[dst->type == GGML_TYPE_F16];
            शेष:
                अवरोध;
        }
        वापस nullptr;
    हाल GGML_OP_GLU:
        अगर ((src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) ||
            (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) ||
            (src0->type != dst->type)) {
            वापस nullptr;
        }

        चयन (ggml_get_glu_op(dst)) {
            हाल GGML_GLU_OP_GEGLU:
                वापस ctx->device->pipeline_geglu[dst->type == GGML_TYPE_F16];
            हाल GGML_GLU_OP_REGLU:
                वापस ctx->device->pipeline_reglu[dst->type == GGML_TYPE_F16];
            हाल GGML_GLU_OP_SWIGLU:
                वापस ctx->device->pipeline_swiglu[dst->type == GGML_TYPE_F16];
            हाल GGML_GLU_OP_SWIGLU_OAI:
                वापस ctx->device->pipeline_swiglu_oai[dst->type == GGML_TYPE_F16];
            हाल GGML_GLU_OP_SWIGLU_CLAMP:
                वापस ctx->device->pipeline_swiglu_clamp[dst->type == GGML_TYPE_F16];
            हाल GGML_GLU_OP_GEGLU_ERF:
                वापस ctx->device->pipeline_geglu_erf[dst->type == GGML_TYPE_F16];
            हाल GGML_GLU_OP_GEGLU_QUICK:
                वापस ctx->device->pipeline_geglu_quick[dst->type == GGML_TYPE_F16];
            शेष:
                अवरोध;
        }
        वापस nullptr;
    हाल GGML_OP_DIAG_MASK_INF:
        अगर (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            वापस ctx->device->pipeline_diag_mask_inf_f32;
        }
        वापस nullptr;
    हाल GGML_OP_SOFT_MAX:
        GGML_ASSERT(!src1 || src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
        GGML_ASSERT(!src2 || src2->type == GGML_TYPE_F32);

        अगर (ctx->अंक_additional_fused_ops) {
            uint32_t idx = (uint32_t)ceilf(log2f(भग्न(dst->ne[0])));
            GGML_ASSERT(idx < अंक_topk_moe_pipelines);
            // use n_experts from push constant if it's not equal to the power of two spec constant
            bool use_धकेलो = dst->ne[0] != (1u << idx);
            वापस ctx->device->pipeline_topk_moe[idx][use_धकेलो];
        }

        अगर (src0->type == GGML_TYPE_F32 && (src1 == nullptr || src1->type == GGML_TYPE_F32) && dst->type == GGML_TYPE_F32) {
            वापस src0->ne[0] > 1024 ? ctx->device->pipeline_soft_max_f32_wg512 : ctx->device->pipeline_soft_max_f32;
        }
        अगर (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) {
            वापस src0->ne[0] > 1024 ? ctx->device->pipeline_soft_max_f32_f16_wg512 : ctx->device->pipeline_soft_max_f32_f16;
        }
        वापस nullptr;
    हाल GGML_OP_SOFT_MAX_BACK:
        अगर (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            वापस ctx->device->pipeline_soft_max_back_f32;
        }
        वापस nullptr;
    हाल GGML_OP_ROPE:
    हाल GGML_OP_ROPE_BACK:
        {
            स्थिर ggml_tensor *rope = ctx->अंक_additional_fused_ops == 2 ? dst->src[0]->src[0] : dst;
            स्थिर पूर्णांक mode = ((स्थिर int32_t *) rope->op_params)[2];
            स्थिर bool is_neox = mode & GGML_ROPE_TYPE_NEOX;
            स्थिर bool is_mrope = mode & GGML_ROPE_TYPE_MROPE;
            स्थिर bool is_vision = mode == GGML_ROPE_TYPE_VISION;

            अगर (is_neox) {
                अगर (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
                    वापस ctx->device->pipeline_rope_neox_f32;
                }
                अगर (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
                    वापस ctx->device->pipeline_rope_neox_f32_f16;
                }
                अगर (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
                    वापस ctx->device->pipeline_rope_neox_f16;
                }
            } अन्यथा अगर (is_mrope && !is_vision) {
                अगर (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
                    वापस ctx->device->pipeline_rope_multi_f32;
                }
                अगर (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
                    वापस ctx->device->pipeline_rope_multi_f32_f16;
                }
                अगर (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
                    वापस ctx->device->pipeline_rope_multi_f16;
                }
            } अन्यथा अगर (is_vision) {
                अगर (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
                    वापस ctx->device->pipeline_rope_vision_f32;
                }
                अगर (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
                    वापस ctx->device->pipeline_rope_vision_f16;
                }
            } अन्यथा {
                अगर (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
                    वापस ctx->device->pipeline_rope_norm_f32;
                }
                अगर (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
                    वापस ctx->device->pipeline_rope_norm_f32_f16;
                }
                अगर (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
                    वापस ctx->device->pipeline_rope_norm_f16;
                }
            }
            वापस nullptr;
        }
    हाल GGML_OP_योग:
    हाल GGML_OP_योग_ROWS:
    हाल GGML_OP_MEAN:
        अगर (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            वापस ctx->device->pipeline_योग_rows_f32;
        }
        वापस nullptr;
    हाल GGML_OP_CROSS_ENTROPY_LOSS:
        अगर (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            वापस src0->ne[0] > 1024 ? ctx->device->pipeline_cross_entropy_loss_f32_wg512 : ctx->device->pipeline_cross_entropy_loss_f32;
        }
        वापस nullptr;
    हाल GGML_OP_CROSS_ENTROPY_LOSS_BACK:
        // src0 is the scalar grad; src1 is logits
        अगर (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && src2 && src2->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            वापस src1->ne[0] > 1024 ? ctx->device->pipeline_cross_entropy_loss_back_f32_wg512 : ctx->device->pipeline_cross_entropy_loss_back_f32;
        }
        वापस nullptr;
    हाल GGML_OP_CUMSUM:
        अगर (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            अगर (src0->ne[0] <= 512) {
                वापस ctx->device->pipeline_cumsum_small_f32;
            } अन्यथा {
                वापस ctx->device->pipeline_cumsum_f32;
            }
        }
        वापस nullptr;
    हाल GGML_OP_SOLVE_TRI:
        अगर (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {

            vk_solve_tri_pipeline_state solve_tri_pipeline_state(src0->ne[0], src1->ne[0]);

            vk_pipeline pipeline = nullptr;

            {
                std::lock_guard<std::mutex> guard(ctx->device->compile_mutex);
                स्वतः it = ctx->device->pipeline_solve_tri_f32.find(solve_tri_pipeline_state);
                अगर (it != ctx->device->pipeline_solve_tri_f32.end()) {
                    pipeline = it->second;
                } अन्यथा {
                    ctx->device->pipeline_solve_tri_f32[solve_tri_pipeline_state] = pipeline = std::make_shared<vk_pipeline_struct>();
                }
            }

            वापस pipeline;
        }
        वापस nullptr;
    हाल GGML_OP_ARGMAX:
        अगर (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_I32) {
            वापस ctx->device->pipeline_argmax_f32;
        }
        वापस nullptr;
    हाल GGML_OP_गिनती_EQUAL:
        अगर (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_I64) {
            वापस ctx->device->pipeline_गिनती_equal_i32;
        }
        वापस nullptr;
    हाल GGML_OP_IM2COL:
        अगर (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            वापस ctx->device->pipeline_im2col_f32;
        }
        अगर (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
            वापस ctx->device->pipeline_im2col_f32_f16;
        }
        वापस nullptr;
    हाल GGML_OP_IM2COL_3D:
        अगर (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            वापस ctx->device->pipeline_im2col_3d_f32;
        }
        अगर (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
            वापस ctx->device->pipeline_im2col_3d_f32_f16;
        }
        वापस nullptr;
    हाल GGML_OP_TIMESTEP_EMBEDDING:
        अगर (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            वापस ctx->device->pipeline_timestep_embedding_f32;
        }
        वापस nullptr;
    हाल GGML_OP_CONV_TRANSPOSE_1D:
        अगर (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            वापस ctx->device->pipeline_conv_transpose_1d_f32;
        }
        वापस nullptr;
    हाल GGML_OP_COL2IM_1D:
        चयन (src0->type) {
            हाल GGML_TYPE_F32:  वापस ctx->device->pipeline_col2im_1d_f32;
            हाल GGML_TYPE_F16:  वापस ctx->device->pipeline_col2im_1d_f16;
            हाल GGML_TYPE_BF16: वापस ctx->device->pipeline_col2im_1d_bf16;
            शेष:             वापस nullptr;
        }
    हाल GGML_OP_POOL_1D:
        अगर (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            वापस ctx->device->pipeline_pool1d_f32;
        }
        वापस nullptr;
    हाल GGML_OP_POOL_2D:
        अगर (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            // Wide windows use neighboring subgroup lanes to read neighboring inputs.
            अगर (ctx->device->pipeline_pool2d_subgroup_f32 && dst->op_params[1] >= int32_t(ctx->device->subgroup_आकार)) {
                वापस ctx->device->pipeline_pool2d_subgroup_f32;
            }
            वापस ctx->device->pipeline_pool2d_f32;
        }
        वापस nullptr;
    हाल GGML_OP_RWKV_WKV6:
        अगर (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            वापस ctx->device->pipeline_rwkv_wkv6_f32;
        }
        वापस nullptr;
    हाल GGML_OP_RWKV_WKV7:
        अगर (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            वापस ctx->device->pipeline_rwkv_wkv7_f32;
        }
        वापस nullptr;
    हाल GGML_OP_GATED_LINEAR_ATTN:
        अगर (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            वापस ctx->device->pipeline_gated_linear_attn_f32;
        }
        वापस nullptr;
    हाल GGML_OP_LIGHTNING_INDEXER:
        // only the k type selects a pipeline, the other types are fixed by ggml_lightning_indexer()
        अगर (ggml_vk_lightning_indexer_k_type_supported(src1->type)) {
            वापस ctx->device->pipeline_lightning_indexer_f32[src1->type];
        }
        वापस nullptr;
    हाल GGML_OP_GATED_DELTA_NET:
        अगर (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            स्थिर uint32_t S_v = dst->src[2]->ne[0];
            स्थिर uint32_t kda = (dst->src[3]->ne[0] == (int64_t)S_v) ? 1 : 0;
            uint32_t si;
            चयन (S_v) {
                हाल 16:  si = 0; अवरोध;
                हाल 32:  si = 1; अवरोध;
                हाल 64:  si = 2; अवरोध;
                हाल 128: si = 3; अवरोध;
                शेष: वापस nullptr;
            }
            वापस ctx->device->pipeline_gated_delta_net[si][kda];
        }
        वापस nullptr;
    हाल GGML_OP_SSM_SCAN:
        अगर (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            स्थिर uint32_t d_state = src0->ne[0];
            अगर (d_state == 128) {
                वापस ctx->device->pipeline_ssm_scan_f32_d128;
            } अन्यथा अगर (d_state == 256) {
                वापस ctx->device->pipeline_ssm_scan_f32_d256;
            }
        }
        वापस nullptr;
    हाल GGML_OP_SSM_CONV:
        अगर (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            चयन (ctx->अंक_additional_fused_ops) {
                हाल 0:  वापस ctx->device->pipeline_ssm_conv_f32;
                हाल 1:  वापस ctx->device->pipeline_ssm_conv_silu_f32;
                हाल 2:  वापस ctx->device->pipeline_ssm_conv_bias_silu_f32;
                शेष: वापस nullptr;
            }
        }
        वापस nullptr;
    हाल GGML_OP_OPT_STEP_ADAMW:
        अगर (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            वापस ctx->device->pipeline_opt_step_adamw_f32;
        }
        वापस nullptr;
    हाल GGML_OP_OPT_STEP_SGD:
        अगर (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            वापस ctx->device->pipeline_opt_step_sgd_f32;
        }
        वापस nullptr;
    हाल GGML_OP_LEAKY_RELU:
        अगर (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            वापस ctx->device->pipeline_leaky_relu[dst->type == GGML_TYPE_F16];
        }
        वापस nullptr;
    हाल GGML_OP_CONV_2D:
    हाल GGML_OP_CONV_TRANSPOSE_2D:
        अगर (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            uint32_t K = dst->ne[2]; // Cout
            uint32_t NPQ = dst->ne[3] * dst->ne[1] * dst->ne[0]; // N * OH * OW
            vk_conv_shapes shape = ggml_vk_conv_select_shape(ctx, K, NPQ);

            bool transpose = dst->op == GGML_OP_CONV_TRANSPOSE_2D;
            uint32_t KW = (uint32_t)src0->ne[0];
            uint32_t KH = (uint32_t)src0->ne[1];
            uint32_t s0 = (uint32_t)(ggml_get_op_params_i32(dst, 0));
            uint32_t s1 = !transpose ? (uint32_t)ggml_get_op_params_i32(dst, 1) : s0;
            uint32_t p0 = !transpose ? (uint32_t)ggml_get_op_params_i32(dst, 2) : 0;
            uint32_t p1 = !transpose ? (uint32_t)ggml_get_op_params_i32(dst, 3) : 0;
            uint32_t d0 = !transpose ? (uint32_t)ggml_get_op_params_i32(dst, 4) : 1;
            uint32_t d1 = !transpose ? (uint32_t)ggml_get_op_params_i32(dst, 5) : 1;

            // tile-aligned shapes let the shader skip bounds checks
            स्थिर uint32_t Cin = (uint32_t)src1->ne[2];
            स्थिर uint32_t CRS = Cin * KW * KH;
            स्थिर uint32_t BS_K   = vk_conv_block_sizes[shape].K;
            स्थिर uint32_t BS_CRS = vk_conv_block_sizes[shape].CRS;
            स्थिर uint32_t BS_NPQ = vk_conv_block_sizes[shape].NPQ;
            स्थिर uint32_t aligned = ((K   % BS_K   == 0) &&
                                      (CRS % BS_CRS == 0) &&
                                      (NPQ % BS_NPQ == 0)) ? 1u : 0u;

            vk_conv2d_pipeline_state conv2d_pipeline_state(s0, s1, p0, p1, d0, d1, KW, KH, aligned);

            std::map<vk_conv2d_pipeline_state, vk_pipeline> *pipelines = nullptr;
            अगर (op == GGML_OP_CONV_2D) {
                अगर (src0->type == GGML_TYPE_F32) {
                    pipelines = &ctx->device->pipeline_conv2d_f32[shape];
                } अन्यथा अगर (src0->type == GGML_TYPE_F16) {
                    pipelines = &ctx->device->pipeline_conv2d_f16_f32[shape];
                }
            } अन्यथा अगर (op == GGML_OP_CONV_TRANSPOSE_2D) {
                अगर (src0->type == GGML_TYPE_F32) {
                    pipelines = &ctx->device->pipeline_conv_transpose_2d_f32[shape];
                } अन्यथा अगर (src0->type == GGML_TYPE_F16) {
                    pipelines = &ctx->device->pipeline_conv_transpose_2d_f16_f32[shape];
                }
            }

            vk_pipeline pipeline = nullptr;

            {
                std::lock_guard<std::mutex> guard(ctx->device->compile_mutex);
                स्वतः it = pipelines->find(conv2d_pipeline_state);
                अगर (it != pipelines->end()) {
                    pipeline = it->second;
                } अन्यथा {
                    (*pipelines)[conv2d_pipeline_state] = pipeline = std::make_shared<vk_pipeline_struct>();
                }
            }

            वापस pipeline;
        }
        वापस nullptr;
    हाल GGML_OP_CONV_2D_DW:
        अगर (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            अगर (ggml_is_contiguous(src1)) {
                वापस ctx->device->pipeline_conv2d_dw_whcn_f32;
            } अन्यथा अगर (ggml_is_contiguous_channels(src1)) {
                वापस ctx->device->pipeline_conv2d_dw_cwhn_f32;
            }
        } अन्यथा अगर (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) {
            अगर (ggml_is_contiguous(src1)) {
                वापस ctx->device->pipeline_conv2d_dw_whcn_f16_f32;
            } अन्यथा अगर (ggml_is_contiguous_channels(src1)) {
                वापस ctx->device->pipeline_conv2d_dw_cwhn_f16_f32;
            }
        }
        वापस nullptr;
    हाल GGML_OP_CONV_3D:
        अगर (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            स्थिर uint32_t OC = (uint32_t)ggml_get_op_params_i32(dst, 11);
            स्थिर uint32_t IC = (uint32_t)ggml_get_op_params_i32(dst, 9);
            स्थिर uint32_t N  = (uint32_t)ggml_get_op_params_i32(dst, 10);
            स्थिर uint32_t NPQ = N * dst->ne[2] * dst->ne[1] * dst->ne[0];
            स्थिर vk_conv_shapes shape = ggml_vk_conv_select_shape(ctx, OC, NPQ);

            स्थिर uint32_t KW = (uint32_t)src0->ne[0];
            स्थिर uint32_t KH = (uint32_t)src0->ne[1];
            स्थिर uint32_t KD = (uint32_t)src0->ne[2];
            स्थिर uint32_t s0 = (uint32_t)ggml_get_op_params_i32(dst, 0);
            स्थिर uint32_t s1 = (uint32_t)ggml_get_op_params_i32(dst, 1);
            स्थिर uint32_t s2 = (uint32_t)ggml_get_op_params_i32(dst, 2);
            स्थिर uint32_t p0 = (uint32_t)ggml_get_op_params_i32(dst, 3);
            स्थिर uint32_t p1 = (uint32_t)ggml_get_op_params_i32(dst, 4);
            स्थिर uint32_t p2 = (uint32_t)ggml_get_op_params_i32(dst, 5);
            स्थिर uint32_t d0 = (uint32_t)ggml_get_op_params_i32(dst, 6);
            स्थिर uint32_t d1 = (uint32_t)ggml_get_op_params_i32(dst, 7);
            स्थिर uint32_t d2 = (uint32_t)ggml_get_op_params_i32(dst, 8);

            स्थिर uint32_t CRS = IC * KW * KH * KD;
            स्थिर uint32_t BS_K   = vk_conv_block_sizes[shape].K;
            स्थिर uint32_t BS_CRS = vk_conv_block_sizes[shape].CRS;
            स्थिर uint32_t BS_NPQ = vk_conv_block_sizes[shape].NPQ;
            स्थिर uint32_t aligned = ((OC  % BS_K   == 0) &&
                                      (CRS % BS_CRS == 0) &&
                                      (NPQ % BS_NPQ == 0)) ? 1u : 0u;

            vk_conv3d_pipeline_state conv3d_pipeline_state(s0, s1, s2, p0, p1, p2, d0, d1, d2, KW, KH, KD, aligned);

            std::map<vk_conv3d_pipeline_state, vk_pipeline> *pipelines = nullptr;
            अगर (src0->type == GGML_TYPE_F32) {
                pipelines = &ctx->device->pipeline_conv3d_f32[shape];
            } अन्यथा अगर (src0->type == GGML_TYPE_F16) {
                pipelines = &ctx->device->pipeline_conv3d_f16_f32[shape];
            } अन्यथा {
                वापस nullptr;
            }

            vk_pipeline pipeline = nullptr;

            {
                std::lock_guard<std::mutex> guard(ctx->device->compile_mutex);
                स्वतः it = pipelines->find(conv3d_pipeline_state);
                अगर (it != pipelines->end()) {
                    pipeline = it->second;
                } अन्यथा {
                    (*pipelines)[conv3d_pipeline_state] = pipeline = std::make_shared<vk_pipeline_struct>();
                }
            }

            वापस pipeline;
        }
        वापस nullptr;
    हाल GGML_OP_ADD1:
        अगर (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
            वापस ctx->device->pipeline_add1_f16_f16;
        }
        अगर (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
            वापस ctx->device->pipeline_add1_f16_f32;
        }
        अगर (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            वापस ctx->device->pipeline_add1_f32_f32;
        }
        वापस nullptr;
    हाल GGML_OP_ARANGE:
        अगर (dst->type == GGML_TYPE_F32) {
            वापस ctx->device->pipeline_arange_f32;
        }
        वापस nullptr;
    हाल GGML_OP_FILL:
        अगर (dst->type == GGML_TYPE_F32) {
            वापस ctx->device->pipeline_fill_f32;
        }
        अगर (dst->type == GGML_TYPE_F16) {
            वापस ctx->device->pipeline_fill_f16;
        }
        वापस nullptr;
    शेष:
        वापस nullptr;
    }

    GGML_UNUSED(src2);
}

ढाँचा<typename PC>
अटल व्योम ggml_vk_op_f32(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, स्थिर ggml_tensor * src1, स्थिर ggml_tensor * src2, स्थिर ggml_tensor * src3, ggml_tensor * dst, ggml_op op, PC&& pc, vk_pipeline pipeline_override = nullptr) {
    VK_LOG_DEBUG("ggml_vk_op_f32((" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    अगर (src1 != nullptr) {
        std::अ_त्रुटि << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    }
    अगर (src2 != nullptr) {
        std::अ_त्रुटि << "), (" << src2 << ", name=" << src2->name << ", type=" << src2->type << ", ne0=" << src2->ne[0] << ", ne1=" << src2->ne[1] << ", ne2=" << src2->ne[2] << ", ne3=" << src2->ne[3] << ", nb0=" << src2->nb[0] << ", nb1=" << src2->nb[1] << ", nb2=" << src2->nb[2] << ", nb3=" << src2->nb[3];
    }
    अगर (src3 != nullptr) {
        std::अ_त्रुटि << "), (" << src3 << ", name=" << src3->name << ", type=" << src3->type << ", ne0=" << src3->ne[0] << ", ne1=" << src3->ne[1] << ", ne2=" << src3->ne[2] << ", ne3=" << src3->ne[3] << ", nb0=" << src3->nb[0] << ", nb1=" << src3->nb[1] << ", nb2=" << src3->nb[2] << ", nb3=" << src3->nb[3];
    }
    std::अ_त्रुटि << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::अ_त्रुटि << "), " << ggml_op_name(op) << ")");
    GGML_ASSERT(op == GGML_OP_GET_ROWS || op == GGML_OP_CPY || op == GGML_OP_CONCAT || (!ggml_is_quantized(src0->type) && (src1 == nullptr || !ggml_is_quantized(src1->type))));  // NOLINT
    GGML_ASSERT(dst->बफर != nullptr);
    स्थिर uint64_t ne00 = src0->ne[0];
    स्थिर uint64_t ne01 = src0->ne[1];
    स्थिर uint64_t ne02 = src0->ne[2];
    स्थिर uint64_t ne03 = src0->ne[3];

    स्थिर bool use_src1 = src1 != nullptr;
    स्थिर uint64_t ne10 = use_src1 ? src1->ne[0] : 0;
    स्थिर uint64_t ne11 = use_src1 ? src1->ne[1] : 0;
    स्थिर uint64_t ne12 = use_src1 ? src1->ne[2] : 0;
    स्थिर uint64_t ne13 = use_src1 ? src1->ne[3] : 0;

    स्थिर bool use_src2 = src2 != nullptr;
    स्थिर bool use_src3 = src3 != nullptr;

    init_pushconst_fastdiv(pc);

    vk_pipeline pipeline;
    अगर (pipeline_override) {
        pipeline = pipeline_override;
    } अन्यथा {
        pipeline = ggml_vk_op_get_pipeline(ctx, src0, src1, src2, dst, op);
    }

    अगर (pipeline == nullptr) {
        std::अ_त्रुटि << "ggml_vulkan: Error: Missing op: " << ggml_op_name(op) << " for " << ggml_type_name(src0->type);
        अगर (src1 != nullptr) {
            std::अ_त्रुटि << " and " << ggml_type_name(src1->type);
        }
        std::अ_त्रुटि << " to " << ggml_type_name(dst->type) << std::प_पूर्ण;
        GGML_ABORT("fatal error");
    }

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer src0_buf = ggml_vk_tensor_subbuffer(ctx, src0, true);
    vk_subbuffer src1_buf = use_src1 ? ggml_vk_tensor_subbuffer(ctx, src1, true) : vk_subbuffer{};
    vk_subbuffer src2_buf = use_src2 ? ggml_vk_tensor_subbuffer(ctx, src2, true) : vk_subbuffer{};
    vk_subbuffer src3_buf = use_src3 ? ggml_vk_tensor_subbuffer(ctx, src3, true) : vk_subbuffer{};
    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst, true);

    // Compute misalignment offset for descriptors and store it in in push constants.
    init_pushconst_tensor_offsets(ctx, pc, src0, src1, src2, src3, dst);

    std::array<uint32_t, 3> elements;

    चयन (op) {
    हाल GGML_OP_NORM:
    हाल GGML_OP_RMS_NORM_BACK:
    हाल GGML_OP_L2_NORM:
    हाल GGML_OP_SOFT_MAX:
    हाल GGML_OP_SOFT_MAX_BACK:
    हाल GGML_OP_योग_ROWS:
    हाल GGML_OP_CUMSUM:
    हाल GGML_OP_MEAN:
    हाल GGML_OP_ARGMAX:
        {
            स्थिर uint32_t nr = ggml_nrows(src0);
            अगर (nr > 262144) {
                elements = { 512, 512, CEIL_DIV(nr, 262144) };
            } अन्यथा अगर (nr > 512) {
                elements = { 512, CEIL_DIV(nr, 512), 1 };
            } अन्यथा {
                elements = { nr, 1, 1 };
            }
        } अवरोध;
    हाल GGML_OP_SOLVE_TRI:
        {
            uint32_t nr = (uint32_t)(ne02 * ne03);
            अगर (nr > 262144) {
                elements = { 512, 512, CEIL_DIV(nr, 262144) };
            } अन्यथा अगर (nr > 512) {
                elements = { 512, CEIL_DIV(nr, 512), 1 };
            } अन्यथा {
                elements = { nr, 1, 1 };
            }
        }
        अवरोध;
    हाल GGML_OP_RMS_NORM:
        अगर (ctx->do_add_rms_partials) {
            // Run one element per thread, 128 threads per workgroup
            elements = { (uint32_t)CEIL_DIV(ne00, 128), 1, 1 };
        } अन्यथा {
            elements = { (uint32_t)ne01, (uint32_t)ne02, (uint32_t)ne03 };
            elements[1] = std::min(elements[1], ctx->device->properties.limits.max_Compute_Work_Group_गिनती[1]);
            elements[2] = std::min(elements[2], ctx->device->properties.limits.max_Compute_Work_Group_गिनती[2]);
        }
        अवरोध;

    हाल GGML_OP_योग:
        // We use GGML_OP_SUM_ROWS with 1 row.
        elements = { 1, 1, 1 };
        अवरोध;
    हाल GGML_OP_GROUP_NORM:
        {
            स्थिर uint32_t अंक_groups = dst->op_params[0];
            elements = { अंक_groups * (uint32_t)src0->ne[3], 1, 1 };
        } अवरोध;
    हाल GGML_OP_DIAG_MASK_INF:
        elements = { (uint32_t)ggml_nrows(src0), (uint32_t)ne00, 1 };
        अवरोध;
    हाल GGML_OP_ROPE:
    हाल GGML_OP_ROPE_BACK:
        {
            uint32_t nrows = (uint32_t)ggml_nrows(src0);
            uint32_t z = 1;
            अगर (nrows > ctx->device->properties.limits.max_Compute_Work_Group_गिनती[0]) {
                z = CEIL_DIV(nrows, 32768);
                nrows = 32768;
            }
            elements = { nrows, (uint32_t)ne00, z };

        } अवरोध;
    हाल GGML_OP_GET_ROWS:
        elements = { (uint32_t)ne00, (uint32_t)ne10, (uint32_t)(ne11 * ne12) };
        elements[1] = std::min(elements[1], ctx->device->properties.limits.max_Compute_Work_Group_गिनती[1]);
        elements[2] = std::min(elements[2], ctx->device->properties.limits.max_Compute_Work_Group_गिनती[2]);
        अवरोध;
    हाल GGML_OP_GET_ROWS_BACK:
        elements = { (uint32_t)dst->ne[0], (uint32_t)dst->ne[1], 1 };
        elements[1] = std::min(elements[1], ctx->device->properties.limits.max_Compute_Work_Group_गिनती[1]);
        अवरोध;
    हाल GGML_OP_ARGSORT:
        GGML_ASSERT(0);
        अवरोध;
    हाल GGML_OP_IM2COL:
        {
            स्थिर bool is_2D = dst->op_params[6] == 1;

            स्थिर uint32_t IC = src1->ne[is_2D ? 2 : 1];

            स्थिर uint32_t KH = is_2D ? src0->ne[1] : 1;
            स्थिर uint32_t KW =         src0->ne[0];

            स्थिर uint32_t OH = is_2D ? dst->ne[2] : 1;
            स्थिर uint32_t OW =         dst->ne[1];

            स्थिर uint32_t batch = src1->ne[is_2D ? 3 : 2];

            स्थिर uint32_t CHW = IC * KH * KW;
            // Cap X workgroups to limit concurrent IC channel reads.
            // The shader loops over X to cover the full CHW dimension.
            // AMD prefers a lower limit
            स्थिर uint32_t min_cap = ctx->device->vendor_id == VK_VENDOR_ID_AMD ? 512u : 4096u;
            स्थिर uint32_t x_elements = std::min(CHW, std::max(min_cap, OW * KH * KW));
            elements = { x_elements, OW, OH * batch };
            elements[1] = std::min(elements[1], ctx->device->properties.limits.max_Compute_Work_Group_गिनती[1]);
            elements[2] = std::min(elements[2], ctx->device->properties.limits.max_Compute_Work_Group_गिनती[2]);
        } अवरोध;
    हाल GGML_OP_IM2COL_3D:
        {
            स्थिर uint32_t IC = ((स्थिर uint32_t *)(dst->op_params))[9];

            स्थिर uint32_t N  = ne13 / IC;

            स्थिर uint32_t KD = ne02;
            स्थिर uint32_t KH = ne01;
            स्थिर uint32_t KW = ne00;

            स्थिर uint32_t OD = dst->ne[3] / N;
            स्थिर uint32_t OH = dst->ne[2];
            स्थिर uint32_t OW = dst->ne[1];

            स्थिर uint32_t IC_KD_KH_KW = IC*KD*KH*KW;
            स्थिर uint32_t N_OD_OH = N*OD*OH;

            elements = { IC_KD_KH_KW, OW, N_OD_OH };
            elements[2] = std::min(elements[2], ctx->device->properties.limits.max_Compute_Work_Group_गिनती[2]);
        } अवरोध;
    हाल GGML_OP_TIMESTEP_EMBEDDING:
        {
            स्थिर uint32_t dim = dst->op_params[0];
            uint32_t half_ceil = (dim + 1) / 2;
            elements = { half_ceil, (uint32_t)src0->ne[0], 1 };
        } अवरोध;
    हाल GGML_OP_CONV_TRANSPOSE_1D:
        {
            elements = {uint32_t(src0->ne[1]), 1, 1}; // parallelize in {Cout, 1, 1}
        } अवरोध;
    हाल GGML_OP_COL2IM_1D:
        {
            elements = { uint32_t(dst->ne[0]), uint32_t(dst->ne[1]), 1 };
        } अवरोध;
    हाल GGML_OP_POOL_1D:
        {
            स्थिर uint32_t N = dst->ne[3] * dst->ne[2];
            स्थिर uint32_t OC = dst->ne[1];
            स्थिर uint32_t OL = dst->ne[0];
            elements = { N * OC * OL, 1, 1};
        } अवरोध;
    हाल GGML_OP_POOL_2D:
        {
            स्थिर uint32_t outputs = ggml_nelements(dst);
            स्थिर uint32_t outputs_per_workgroup = pipeline == ctx->device->pipeline_pool2d_subgroup_f32 ? 128 / ctx->device->subgroup_आकार : 512;
            स्थिर uint32_t workgroups = CEIL_DIV(outputs, outputs_per_workgroup);
            स्थिर uint32_t groups_y = CEIL_DIV(workgroups, ctx->device->properties.limits.max_Compute_Work_Group_गिनती[0]);
            स्थिर uint32_t groups_x = CEIL_DIV(workgroups, groups_y);
            elements = { groups_x, groups_y, 1 };
        } अवरोध;
    हाल GGML_OP_CONV_2D:
    हाल GGML_OP_CONV_TRANSPOSE_2D:
        अगर constexpr (std::is_same_v<PC, vk_op_conv2d_धकेलो_constants>) {
            स्थिर uint32_t NPQ = pc.N * pc.OH * pc.OW;
            स्थिर vk_conv_shapes shape = ggml_vk_conv_select_shape(ctx, pc.Cout, NPQ);
            स्थिर uint32_t NPQ_blocks = CEIL_DIV(NPQ, vk_conv_block_sizes[shape].NPQ);

            elements = { pc.Cout, NPQ_blocks, 1 };
            अगर (elements[1] > 512) {
                elements[2] = CEIL_DIV(elements[1], 512);
                elements[1] = 512;
            }
        } अन्यथा {
            GGML_ABORT("invalid push constant type for CONV_2D");
        }
        अवरोध;
    हाल GGML_OP_CONV_3D:
        अगर constexpr (std::is_same_v<PC, vk_op_conv3d_धकेलो_constants>) {
            स्थिर uint32_t NPQ = pc.N * pc.OD * pc.OH * pc.OW;
            स्थिर vk_conv_shapes shape = ggml_vk_conv_select_shape(ctx, pc.OC, NPQ);
            स्थिर uint32_t NPQ_blocks = CEIL_DIV(NPQ, vk_conv_block_sizes[shape].NPQ);

            elements = { pc.OC, NPQ_blocks, 1 };
            अगर (elements[1] > 512) {
                elements[2] = CEIL_DIV(elements[1], 512);
                elements[1] = 512;
            }
        } अन्यथा {
            GGML_ABORT("invalid push constant type for CONV_3D");
        }
        अवरोध;
    हाल GGML_OP_ADD:
    हाल GGML_OP_SUB:
    हाल GGML_OP_DIV:
    हाल GGML_OP_MUL:
    हाल GGML_OP_ADD1:
    हाल GGML_OP_OUT_PROD:
    हाल GGML_OP_ARANGE:
    हाल GGML_OP_FILL:
    हाल GGML_OP_SCALE:
    हाल GGML_OP_SQR:
    हाल GGML_OP_SQRT:
    हाल GGML_OP_SIN:
    हाल GGML_OP_COS:
    हाल GGML_OP_LOG:
    हाल GGML_OP_TRI:
    हाल GGML_OP_DIAG:
    हाल GGML_OP_CLAMP:
    हाल GGML_OP_LEAKY_RELU:
    हाल GGML_OP_PAD:
    हाल GGML_OP_PAD_REFLECT_1D:
    हाल GGML_OP_ROLL:
    हाल GGML_OP_REPEAT:
    हाल GGML_OP_REPEAT_BACK:
    हाल GGML_OP_CPY:
    हाल GGML_OP_CONCAT:
    हाल GGML_OP_UPSCALE:
    हाल GGML_OP_UNARY:
    हाल GGML_OP_GLU:
    हाल GGML_OP_CONV_2D_DW:
        {
            uint32_t ne = ggml_nelements(dst);
            अगर (op == GGML_OP_CPY && ggml_is_quantized(src0->type) && ggml_is_quantized(dst->type)) {
                // Convert from number of logical elements to 2- or 4-byte units.
                ne /= ggml_blck_आकार(src0->type);
                अगर ((ggml_type_आकार(src0->type) % 4) == 0) {
                    ne *= ggml_type_आकार(src0->type) / 4;
                } अन्यथा {
                    ne *= ggml_type_आकार(src0->type) / 2;
                }
            }
            अगर (op == GGML_OP_CONCAT && ggml_is_quantized(dst->type)) {
                ne = ne / ggml_blck_आकार(dst->type) * ggml_type_आकार(dst->type) / ggml_vk_concat_unit_आकार(dst->type);
            }
            // copy_to_quant has block size of 32, and each thread does QUANT_K elements.
            // Splitting into 512x512xZ wouldn't work well since each workgroup does 1024 elements.
            // So divide by block size here before splitting into 512x512 groups.
            अगर (op == GGML_OP_CPY && !ggml_is_quantized(src0->type) && ggml_is_quantized(dst->type)) {
                ne = CEIL_DIV(ne, ggml_blck_आकार(dst->type));
            }
            अगर (ne > 262144) {
                elements = { 512, 512, CEIL_DIV(ne, 262144) };
            } अन्यथा अगर (ne > 512) {
                elements = { 512, CEIL_DIV(ne, 512), 1 };
            } अन्यथा {
                elements = { ne, 1, 1 };
            }

            अगर (pipeline == ctx->device->pipeline_cpy_transpose_02_32 ||
                pipeline == ctx->device->pipeline_cpy_transpose_02_16) {
                // 32x32 tiles over dims 0 and 2; dim1 and dim3 are the batch
                elements[0] = (uint32_t)CEIL_DIV(dst->ne[0], 32);
                elements[1] = (uint32_t)CEIL_DIV(dst->ne[2], 32);
                elements[2] = (uint32_t)(dst->ne[1]*dst->ne[3]);
                elements[0] = std::min(elements[0], ctx->device->properties.limits.max_Compute_Work_Group_गिनती[0]);
                elements[1] = std::min(elements[1], ctx->device->properties.limits.max_Compute_Work_Group_गिनती[1]);
                elements[2] = std::min(elements[2], ctx->device->properties.limits.max_Compute_Work_Group_गिनती[2]);
            } अन्यथा अगर (pipeline == ctx->device->pipeline_cpy_transpose_32 ||
                pipeline == ctx->device->pipeline_cpy_transpose_16) {
                // 32x32 tiles
                elements[0] = (uint32_t)CEIL_DIV(dst->ne[0], 32);
                elements[1] = (uint32_t)CEIL_DIV(dst->ne[1], 32);
                elements[2] = (uint32_t)(dst->ne[2]*dst->ne[3]);
                elements[0] = std::min(elements[0], ctx->device->properties.limits.max_Compute_Work_Group_गिनती[0]);
                elements[1] = std::min(elements[1], ctx->device->properties.limits.max_Compute_Work_Group_गिनती[1]);
                elements[2] = std::min(elements[2], ctx->device->properties.limits.max_Compute_Work_Group_गिनती[2]);
            }
        } अवरोध;
    हाल GGML_OP_ADD_ID:
        {
            elements = { (uint32_t)ne01, (uint32_t)ne02, 1 };
        } अवरोध;
    हाल GGML_OP_SET_ROWS:
        {
            uint32_t ne = ggml_nelements(src0);
            अगर (ggml_is_quantized(dst->type)) {
                // quants run 32 threads each doing QUANT_K elements
                ne = CEIL_DIV(ne, 32 * ggml_blck_आकार(dst->type));
            } अन्यथा {
                // scalar types do one element per thread, running 512 threads
                ne = CEIL_DIV(ne, 512);
            }
            अगर (ne > 262144) {
                elements = { 512, 512, CEIL_DIV(ne, 262144) };
            } अन्यथा अगर (ne > 512) {
                elements = { 512, CEIL_DIV(ne, 512), 1 };
            } अन्यथा {
                elements = { ne, 1, 1 };
            }
        }
        अवरोध;
    हाल GGML_OP_SSM_CONV:
        {
            स्थिर uint32_t nr  = src0->ne[1];
            स्थिर uint32_t n_t = dst->ne[1];
            स्थिर uint32_t n_s = dst->ne[2];
            elements = { nr, n_t, n_s };
        }
        अवरोध;
    शेष:
        elements = { (uint32_t)ggml_nelements(src0), 1, 1 };
        अवरोध;
    }

    अगर (op == GGML_OP_ADD || op == GGML_OP_RMS_NORM) {
        vk_subbuffer a_buf = src0_buf;
        अगर (ctx->do_add_rms_partials) {
            a_buf = ggml_vk_subbuffer(ctx, ctx->prealloc_add_rms_partials, ctx->prealloc_आकार_add_rms_partials_offset);
        }
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
            { src0_buf, src1_buf, dst_buf, a_buf }, pc, elements);
    } अन्यथा अगर (op == GGML_OP_GLU) {
        // Empty src1 is possible in glu, but the shader needs a buffer
        vk_subbuffer subbuf1 = use_src1 ? src1_buf : src0_buf;
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, subbuf1, dst_buf }, pc, elements);
    } अन्यथा अगर (op == GGML_OP_SOFT_MAX) {
        // Empty src1 and src2 is possible in soft_max, but the shader needs a buffer
        vk_subbuffer subbuf1 = use_src1 ? src1_buf : src0_buf;
        vk_subbuffer subbuf2 = use_src2 ? src2_buf : src0_buf;
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, subbuf1, subbuf2, dst_buf }, pc, elements);
    } अन्यथा अगर (op == GGML_OP_ROPE || op == GGML_OP_ROPE_BACK) {
        // Empty src2 and src3 is possible in rope, but the shader needs a buffer
        vk_subbuffer subbuf2 = use_src2 ? src2_buf : src0_buf;
        vk_subbuffer subbuf3 = use_src3 ? src3_buf : src0_buf;
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, subbuf2, dst_buf, subbuf3 }, pc, elements);
    } अन्यथा अगर (op == GGML_OP_IM2COL || op == GGML_OP_IM2COL_3D) {
        अगर (ctx->device->shader_int64 && ctx->device->बफर_device_address) {
            // buffer device address path doesn't use dst buffer
            dst_buf.size = 1;
        }
        // im2col uses only src1 and dst buffers
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src1_buf, dst_buf }, pc, elements);
    } अन्यथा अगर (op == GGML_OP_गिनती_EQUAL) {
        // count_equal assumes that destination buffer is initialized with zeroes
        ggml_vk_बफर_memset_async(subctx, dst_buf.बफर, dst_buf.offset, 0, dst_buf.size);
        ggml_vk_sync_buffers(ctx, subctx);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, dst_buf }, pc, elements);
    } अन्यथा अगर (op == GGML_OP_OPT_STEP_SGD) {
        // OPT_STEP_SGD works on src0, it does not need dst
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, src2_buf }, pc, elements);
    } अन्यथा अगर (use_src3) {
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, src2_buf, src3_buf, dst_buf }, pc, elements);
    } अन्यथा अगर (use_src2) {
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, src2_buf, dst_buf }, pc, elements);
    } अन्यथा अगर (use_src1) {
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, dst_buf }, pc, elements);
    } अन्यथा {
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, dst_buf }, pc, elements);
    }
}

व्योम ggml_vk_get_rows(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, स्थिर ggml_tensor * src1, ggml_tensor * dst) {
    स्थिर uint32_t src0_type_आकार = ggml_type_आकार(src0->type);
    स्थिर uint32_t src1_type_आकार = ggml_type_आकार(src1->type);
    स्थिर uint32_t dst_type_आकार = ggml_type_आकार(dst->type);

    ggml_vk_op_f32<vk_op_द्विआधारी_धकेलो_constants>(ctx, subctx, src0, src1, nullptr, nullptr, dst, GGML_OP_GET_ROWS, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_आकार, (uint32_t)src0->nb[1] / src0_type_आकार, (uint32_t)src0->nb[2] / src0_type_आकार, (uint32_t)src0->nb[3] / src0_type_आकार,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_आकार, (uint32_t)src1->nb[1] / src1_type_आकार, (uint32_t)src1->nb[2] / src1_type_आकार, (uint32_t)src1->nb[3] / src1_type_आकार,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_आकार, (uint32_t) dst->nb[1] /  dst_type_आकार, (uint32_t) dst->nb[2] /  dst_type_आकार, (uint32_t) dst->nb[3] /  dst_type_आकार,
        0,
        0.0f, 0.0f, 0,
    });
}

व्योम ggml_vk_get_rows_back(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, स्थिर ggml_tensor * src1, ggml_tensor * dst) {
    स्थिर uint32_t src0_type_आकार = ggml_type_आकार(src0->type);
    स्थिर uint32_t src1_type_आकार = ggml_type_आकार(src1->type);
    स्थिर uint32_t dst_type_आकार = ggml_type_आकार(dst->type);

    ggml_vk_op_f32<vk_op_द्विआधारी_धकेलो_constants>(ctx, subctx, src0, src1, nullptr, nullptr, dst, GGML_OP_GET_ROWS_BACK, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2], (uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_आकार, (uint32_t)src0->nb[1] / src0_type_आकार, (uint32_t)src0->nb[2] / src0_type_आकार, (uint32_t)src0->nb[3] / src0_type_आकार,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2], (uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_आकार, (uint32_t)src1->nb[1] / src1_type_आकार, (uint32_t)src1->nb[2] / src1_type_आकार, (uint32_t)src1->nb[3] / src1_type_आकार,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2], (uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_आकार, (uint32_t) dst->nb[1] /  dst_type_आकार, (uint32_t) dst->nb[2] /  dst_type_आकार, (uint32_t) dst->nb[3] /  dst_type_आकार,
        0,
        0.0f, 0.0f, 0,
    });
}

व्योम ggml_vk_acc(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, स्थिर ggml_tensor * src1, ggml_tensor * dst) {
    स्थिर uint32_t src0_type_आकार = ggml_type_आकार(src0->type);
    स्थिर uint32_t src1_type_आकार = ggml_type_आकार(src1->type);
    स्थिर uint32_t dst_type_आकार = ggml_type_आकार(dst->type);

    पूर्णांक nb1 = dst->op_params[0] / src0_type_आकार; // 4 bytes of float32
    पूर्णांक nb2 = dst->op_params[1] / src0_type_आकार; // 4 bytes of float32
    पूर्णांक nb3 = dst->op_params[2] / src0_type_आकार; // 4 bytes of float32
    पूर्णांक offset = dst->op_params[3] / src0_type_आकार; // offset in bytes

    ggml_vk_op_f32<vk_op_द्विआधारी_धकेलो_constants>(ctx, subctx, src0, src1, nullptr, nullptr, dst, dst->op, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_आकार, (uint32_t)nb1, (uint32_t)nb2, (uint32_t)nb3,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_आकार, (uint32_t)src1->nb[1] / src1_type_आकार, (uint32_t)src1->nb[2] / src1_type_आकार, (uint32_t)src1->nb[3] / src1_type_आकार,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_आकार, (uint32_t)nb1, (uint32_t)nb2, (uint32_t)nb3,
        0,
        0.0f, 0.0f, offset,
    });
}

व्योम ggml_vk_multi_add(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_cgraph * cgraph, पूर्णांक गाँठ_idx) {
    स्थिर ggml_tensor *first_गाँठ = cgraph->nodes[गाँठ_idx];
    स्थिर ggml_tensor *dst = cgraph->nodes[गाँठ_idx + ctx->अंक_additional_fused_ops];

    // Make a list of all the tensors used by the op.
    // Last element of the list is the dest tensor.
    स्थिर ggml_tensor *tensors[MAX_PARAMETER_गिनती];
    uint32_t अंक_srcs = ctx->अंक_additional_fused_ops + 2;
    uint32_t अंक_tensors = अंक_srcs + 1;
    GGML_ASSERT(अंक_tensors + ctx->do_add_rms_partials <= MAX_PARAMETER_गिनती);

    tensors[0] = first_गाँठ->src[0];
    tensors[1] = first_गाँठ->src[1];
    क्रम (int32_t i = 0; i < ctx->अंक_additional_fused_ops; ++i) {
        // check whether the previous result is src[0] or src[1]
        अगर (cgraph->nodes[गाँठ_idx + i] == cgraph->nodes[गाँठ_idx + i + 1]->src[0]) {
            tensors[i+2] = cgraph->nodes[गाँठ_idx + i + 1]->src[1];
        } अन्यथा {
            tensors[i+2] = cgraph->nodes[गाँठ_idx + i + 1]->src[0];
        }
    }
    tensors[अंक_srcs] = dst;

    vk_op_multi_add_धकेलो_constants pc;
    pc.ne20 = (uint32_t)dst->ne[0];
    pc.ne21 = (uint32_t)dst->ne[1];
    pc.ne22 = (uint32_t)dst->ne[2];
    pc.ne23 = (uint32_t)dst->ne[3];

    क्रम (uint32_t i = 0; i < अंक_tensors; ++i) {
        स्थिर ggml_tensor *t = tensors[i];
        pc.nb[i][0] = (uint32_t)t->nb[0] / माप(भग्न);
        pc.nb[i][1] = (uint32_t)t->nb[1] / माप(भग्न);
        pc.nb[i][2] = (uint32_t)t->nb[2] / माप(भग्न);
        pc.nb[i][3] = (uint32_t)t->nb[3] / माप(भग्न);
    }
    pc.rms_partials = ctx->do_add_rms_partials;

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, tensors[0], tensors[1], nullptr, dst, dst->op);

    अगर (pipeline == nullptr) {
        std::अ_त्रुटि << "ggml_vulkan: Error: Missing multi_add";
        GGML_ABORT("fatal error");
    }

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    ggml_backend_vk_बफर_context * buf_ctx[MAX_PARAMETER_गिनती];
    vk_बफर buf[MAX_PARAMETER_गिनती];
    माप_प्रकार offset[MAX_PARAMETER_गिनती];
    bool uma[MAX_PARAMETER_गिनती];

    क्रम (uint32_t i = 0; i < अंक_tensors; ++i) {
        buf_ctx[i] = (ggml_backend_vk_बफर_context *)tensors[i]->बफर->context;
        buf[i] = nullptr;
        offset[i] = 0;
        uma[i] = false;

        अगर (ctx->device->uma) {
            ggml_vk_host_get(ctx->device, tensors[i]->तथ्य, buf[i], offset[i]);
            uma[i] = buf[i] != nullptr;
        }
        अगर (!uma[i]) {
            buf[i] = buf_ctx[i]->dev_बफर;
            offset[i] = vk_tensor_offset(tensors[i]) + tensors[i]->view_offs;
        }
        GGML_ASSERT(buf[i] != nullptr);
    }
    // If any remaining descriptors are unused, just point them at src[0]
    क्रम (uint32_t i = अंक_tensors; i < MAX_PARAMETER_गिनती; ++i) {
        buf[i] = buf[0];
        offset[i] = 0;
    }
    अगर (ctx->do_add_rms_partials) {
        buf[अंक_tensors] = ctx->prealloc_add_rms_partials;
        offset[अंक_tensors] = ctx->prealloc_आकार_add_rms_partials_offset;
    }

    std::array<uint32_t, 3> elements;

    uint32_t ne = ggml_nelements(dst);
    अगर (ne > 262144) {
        elements = { 512, 512, CEIL_DIV(ne, 262144) };
    } अन्यथा अगर (ne > 512) {
        elements = { 512, CEIL_DIV(ne, 512), 1 };
    } अन्यथा {
        elements = { ne, 1, 1 };
    }

    static_assert(MAX_PARAMETER_गिनती == 12);
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        {
            ggml_vk_subbuffer(ctx, buf[0], offset[0]),
            ggml_vk_subbuffer(ctx, buf[1], offset[1]),
            ggml_vk_subbuffer(ctx, buf[2], offset[2]),
            ggml_vk_subbuffer(ctx, buf[3], offset[3]),
            ggml_vk_subbuffer(ctx, buf[4], offset[4]),
            ggml_vk_subbuffer(ctx, buf[5], offset[5]),
            ggml_vk_subbuffer(ctx, buf[6], offset[6]),
            ggml_vk_subbuffer(ctx, buf[7], offset[7]),
            ggml_vk_subbuffer(ctx, buf[8], offset[8]),
            ggml_vk_subbuffer(ctx, buf[9], offset[9]),
            ggml_vk_subbuffer(ctx, buf[10], offset[10]),
            ggml_vk_subbuffer(ctx, buf[11], offset[11]),
        }, pc, elements);
}

व्योम ggml_vk_add(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, स्थिर ggml_tensor * src1, ggml_tensor * dst) {
    स्थिर uint32_t src0_type_आकार = ggml_type_आकार(src0->type);
    स्थिर uint32_t src1_type_आकार = ggml_type_आकार(src1->type);
    स्थिर uint32_t dst_type_आकार = ggml_type_आकार(dst->type);

    ggml_vk_op_f32<vk_op_द्विआधारी_धकेलो_constants>(ctx, subctx, src0, src1, nullptr, nullptr, dst, GGML_OP_ADD, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_आकार, (uint32_t)src0->nb[1] / src0_type_आकार, (uint32_t)src0->nb[2] / src0_type_आकार, (uint32_t)src0->nb[3] / src0_type_आकार,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_आकार, (uint32_t)src1->nb[1] / src1_type_आकार, (uint32_t)src1->nb[2] / src1_type_आकार, (uint32_t)src1->nb[3] / src1_type_आकार,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_आकार, (uint32_t) dst->nb[1] /  dst_type_आकार, (uint32_t) dst->nb[2] /  dst_type_आकार, (uint32_t) dst->nb[3] /  dst_type_आकार,
        0,
        0.0f, 0.0f, ctx->do_add_rms_partials,
    });
}

व्योम ggml_vk_out_prod(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, स्थिर ggml_tensor * src1, ggml_tensor * dst) {
    स्थिर uint32_t src0_type_आकार = ggml_type_आकार(src0->type);
    स्थिर uint32_t src1_type_आकार = ggml_type_आकार(src1->type);
    स्थिर uint32_t dst_type_आकार = ggml_type_आकार(dst->type);

    ggml_vk_op_f32<vk_op_द्विआधारी_धकेलो_constants>(ctx, subctx, src0, src1, nullptr, nullptr, dst, GGML_OP_OUT_PROD, {
        (uint32_t)ggml_nelements(dst),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3],
        (uint32_t)src0->nb[0] / src0_type_आकार, (uint32_t)src0->nb[1] / src0_type_आकार, (uint32_t)src0->nb[2] / src0_type_आकार, (uint32_t)src0->nb[3] / src0_type_आकार,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3],
        (uint32_t)src1->nb[0] / src1_type_आकार, (uint32_t)src1->nb[1] / src1_type_आकार, (uint32_t)src1->nb[2] / src1_type_आकार, (uint32_t)src1->nb[3] / src1_type_आकार,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3],
        (uint32_t) dst->nb[0] /  dst_type_आकार, (uint32_t) dst->nb[1] /  dst_type_आकार, (uint32_t) dst->nb[2] /  dst_type_आकार, (uint32_t) dst->nb[3] /  dst_type_आकार,
        0,
        0.0f, 0.0f, 0,
    });
}

व्योम ggml_vk_sub(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, स्थिर ggml_tensor * src1, ggml_tensor * dst) {
    स्थिर uint32_t src0_type_आकार = ggml_type_आकार(src0->type);
    स्थिर uint32_t src1_type_आकार = ggml_type_आकार(src1->type);
    स्थिर uint32_t dst_type_आकार = ggml_type_आकार(dst->type);

    ggml_vk_op_f32<vk_op_द्विआधारी_धकेलो_constants>(ctx, subctx, src0, src1, nullptr, nullptr, dst, GGML_OP_SUB, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_आकार, (uint32_t)src0->nb[1] / src0_type_आकार, (uint32_t)src0->nb[2] / src0_type_आकार, (uint32_t)src0->nb[3] / src0_type_आकार,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_आकार, (uint32_t)src1->nb[1] / src1_type_आकार, (uint32_t)src1->nb[2] / src1_type_आकार, (uint32_t)src1->nb[3] / src1_type_आकार,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_आकार, (uint32_t) dst->nb[1] /  dst_type_आकार, (uint32_t) dst->nb[2] /  dst_type_आकार, (uint32_t) dst->nb[3] /  dst_type_आकार,
        0,
        0.0f, 0.0f, 0,
    });
}

व्योम ggml_vk_mul(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, स्थिर ggml_tensor * src1, ggml_tensor * dst) {
    स्थिर uint32_t src0_type_आकार = ggml_type_आकार(src0->type);
    स्थिर uint32_t src1_type_आकार = ggml_type_आकार(src1->type);
    स्थिर uint32_t dst_type_आकार = ggml_type_आकार(dst->type);

    ggml_vk_op_f32<vk_op_द्विआधारी_धकेलो_constants>(ctx, subctx, src0, src1, nullptr, nullptr, dst, GGML_OP_MUL, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_आकार, (uint32_t)src0->nb[1] / src0_type_आकार, (uint32_t)src0->nb[2] / src0_type_आकार, (uint32_t)src0->nb[3] / src0_type_आकार,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_आकार, (uint32_t)src1->nb[1] / src1_type_आकार, (uint32_t)src1->nb[2] / src1_type_आकार, (uint32_t)src1->nb[3] / src1_type_आकार,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_आकार, (uint32_t) dst->nb[1] /  dst_type_आकार, (uint32_t) dst->nb[2] /  dst_type_आकार, (uint32_t) dst->nb[3] /  dst_type_आकार,
        0,
        0.0f, 0.0f, 0,
    });
}

पूर्णांक ggml_vk_unary_mul_op_सूचकांक(ggml_unary_op op) {
    चयन (op) {
        हाल GGML_UNARY_OP_GELU:     वापस 0;
        हाल GGML_UNARY_OP_SIGMOID:  वापस 1;
        हाल GGML_UNARY_OP_SILU:     वापस 2;
        हाल GGML_UNARY_OP_SOFTPLUS: वापस 3;
        शेष:                     वापस -1;
    }
}

व्योम ggml_vk_unary_mul(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर काष्ठा ggml_cgraph * cgraph, पूर्णांक गाँठ_idx) {
    स्थिर ggml_tensor * unary = cgraph->nodes[गाँठ_idx];
    ggml_tensor * mul = cgraph->nodes[गाँठ_idx + 1];

    // unary on src1 that tiles into src0
    स्थिर bool op_on_b = mul->src[1] == unary &&
                         !ggml_are_same_shape(unary->src[0], mul->src[0]) &&
                         ggml_can_repeat(unary, mul->src[0]);

    स्थिर ggml_tensor * src0 = op_on_b ? mul->src[0] : unary->src[0];
    स्थिर ggml_tensor * src1 = op_on_b ? unary->src[0] :
        ((mul->src[0] == unary) ? mul->src[1] : mul->src[0]);

    स्थिर bool f16 = src0->type == GGML_TYPE_F16;
    स्थिर bool norepeat = ggml_are_same_shape(src0, src1);
    स्थिर पूर्णांक oi = ggml_vk_unary_mul_op_सूचकांक(ggml_get_unary_op(unary));
    अगर (oi < 0) {
        GGML_ABORT("fatal error");
    }
    vk_pipeline pipeline = ctx->device->pipeline_unary_mul[oi][f16][norepeat][op_on_b];

    स्थिर uint32_t src0_type_आकार = ggml_type_आकार(src0->type);
    स्थिर uint32_t src1_type_आकार = ggml_type_आकार(src1->type);
    स्थिर uint32_t dst_type_आकार = ggml_type_आकार(mul->type);

    ggml_vk_op_f32<vk_op_द्विआधारी_धकेलो_constants>(ctx, subctx, src0, src1, nullptr, nullptr, mul, GGML_OP_UNARY, {
        (uint32_t)ggml_nelements(op_on_b ? mul : src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_आकार, (uint32_t)src0->nb[1] / src0_type_आकार, (uint32_t)src0->nb[2] / src0_type_आकार, (uint32_t)src0->nb[3] / src0_type_आकार,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_आकार, (uint32_t)src1->nb[1] / src1_type_आकार, (uint32_t)src1->nb[2] / src1_type_आकार, (uint32_t)src1->nb[3] / src1_type_आकार,
        (uint32_t) mul->ne[0], (uint32_t) mul->ne[1], (uint32_t) mul->ne[2],(uint32_t) mul->ne[3], (uint32_t) mul->nb[0] /  dst_type_आकार, (uint32_t) mul->nb[1] /  dst_type_आकार, (uint32_t) mul->nb[2] /  dst_type_आकार, (uint32_t) mul->nb[3] /  dst_type_आकार,
        0,
        0.0f, 0.0f, 0,
    }, pipeline);
}

व्योम ggml_vk_div(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, स्थिर ggml_tensor * src1, ggml_tensor * dst) {
    स्थिर uint32_t src0_type_आकार = ggml_type_आकार(src0->type);
    स्थिर uint32_t src1_type_आकार = ggml_type_आकार(src1->type);
    स्थिर uint32_t dst_type_आकार = ggml_type_आकार(dst->type);

    ggml_vk_op_f32<vk_op_द्विआधारी_धकेलो_constants>(ctx, subctx, src0, src1, nullptr, nullptr, dst, GGML_OP_DIV, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_आकार, (uint32_t)src0->nb[1] / src0_type_आकार, (uint32_t)src0->nb[2] / src0_type_आकार, (uint32_t)src0->nb[3] / src0_type_आकार,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_आकार, (uint32_t)src1->nb[1] / src1_type_आकार, (uint32_t)src1->nb[2] / src1_type_आकार, (uint32_t)src1->nb[3] / src1_type_आकार,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_आकार, (uint32_t) dst->nb[1] /  dst_type_आकार, (uint32_t) dst->nb[2] /  dst_type_आकार, (uint32_t) dst->nb[3] /  dst_type_आकार,
        0,
        0.0f, 0.0f, 0,
    });
}

व्योम ggml_vk_add_id(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, स्थिर ggml_tensor * src1, स्थिर ggml_tensor * src2, ggml_tensor * dst) {
    स्थिर uint32_t src0_type_आकार = ggml_type_आकार(src0->type);
    स्थिर uint32_t src1_type_आकार = ggml_type_आकार(src1->type);
    स्थिर uint32_t src2_type_आकार = ggml_type_आकार(src2->type);

    ggml_vk_op_f32<vk_op_add_id_धकेलो_constants>(ctx, subctx, src0, src1, src2, nullptr, dst, GGML_OP_ADD_ID, {
        (uint32_t)dst->ne[0],
        (uint32_t)dst->ne[1],
        (uint32_t)src0->nb[1] / src0_type_आकार,
        (uint32_t)src0->nb[2] / src0_type_आकार,
        (uint32_t)src1->nb[1] / src1_type_आकार,
        (uint32_t)src2->nb[1] / src2_type_आकार,
    });
}

अटल व्योम ggml_vk_op_f32_wkv(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst, स्थिर vk_op_rwkv_wkv6_धकेलो_constants&& pc, पूर्णांक version) {
    GGML_ASSERT(version == 6 || version == 7);
    पूर्णांक अंक_srcs = version == 6 ? 6 : 7;

    क्रम (पूर्णांक i = 0; i < अंक_srcs; i++) {
        GGML_ASSERT(!ggml_is_quantized(dst->src[i]->type));
    }

    GGML_ASSERT(dst->बफर != nullptr);

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, dst->src[0], dst->src[1], dst->src[2], dst, dst->op);
    GGML_ASSERT(pipeline != nullptr);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
    vk_subbuffer src_buf[7] = {};
    क्रम (पूर्णांक i = 0; i < अंक_srcs; i++) {
        src_buf[i] = ggml_vk_tensor_subbuffer(ctx, dst->src[i]);
    }

    std::array<uint32_t, 3> elements = {
        (uint32_t)(pc.B * pc.H),
        1,
        1
    };

    अगर (version == 6) {
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
            {src_buf[0], src_buf[1], src_buf[2], src_buf[3], src_buf[4], src_buf[5], dst_buf},
            pc, elements);
    } अन्यथा अगर (version == 7) {
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
            {src_buf[0], src_buf[1], src_buf[2], src_buf[3], src_buf[4], src_buf[5], src_buf[6], dst_buf},
            pc, elements);
    } अन्यथा {
        // shouldn't happen
        GGML_ASSERT(false);
    }
}

व्योम ggml_vk_rwkv_wkv6(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    स्थिर माप_प्रकार seq_लंबाई = dst->src[0]->ne[2];
    स्थिर माप_प्रकार n_embed = dst->ne[0];
    स्थिर माप_प्रकार n_heads = dst->src[0]->ne[1];
    स्थिर माप_प्रकार n_seqs = dst->src[5]->ne[1];

    ggml_vk_op_f32_wkv(
        ctx, subctx, dst,
        {
            (uint32_t)n_seqs,
            (uint32_t)seq_लंबाई,
            (uint32_t)n_embed,
            (uint32_t)n_heads,
        },
        6
    );
}

व्योम ggml_vk_rwkv_wkv7(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    स्थिर माप_प्रकार seq_लंबाई = dst->src[0]->ne[2];
    स्थिर माप_प्रकार n_embed = dst->ne[0];
    स्थिर माप_प्रकार n_heads = dst->src[0]->ne[1];
    स्थिर माप_प्रकार n_seqs = dst->src[6]->ne[1];

    ggml_vk_op_f32_wkv(
        ctx, subctx, dst,
        {
            (uint32_t)n_seqs,
            (uint32_t)seq_लंबाई,
            (uint32_t)n_embed,
            (uint32_t)n_heads,
        },
        7
    );
}

व्योम ggml_vk_gated_linear_attn(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    स्थिर माप_प्रकार seq_लंबाई = dst->src[0]->ne[2];
    स्थिर माप_प्रकार n_embed    = dst->ne[0];
    स्थिर माप_प्रकार n_heads    = dst->src[0]->ne[1];
    स्थिर माप_प्रकार n_seqs     = dst->src[4]->ne[1];

    भग्न scale;
    स_नकल(&scale, dst->op_params, माप(भग्न));

    GGML_ASSERT(dst->बफर != nullptr);

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, dst->src[0], dst->src[1], dst->src[2], dst, dst->op);
    GGML_ASSERT(pipeline != nullptr);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
    vk_subbuffer src_buf[5] = {};
    क्रम (पूर्णांक i = 0; i < 5; i++) {
        src_buf[i] = ggml_vk_tensor_subbuffer(ctx, dst->src[i]);
    }

    स्थिर vk_op_gated_linear_attn_धकेलो_constants pc = {
        (uint32_t)n_seqs,
        (uint32_t)seq_लंबाई,
        (uint32_t)n_embed,
        (uint32_t)n_heads,
        scale,
    };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        {src_buf[0], src_buf[1], src_buf[2], src_buf[3], src_buf[4], dst_buf},
        pc, { (uint32_t)(n_seqs * n_heads), 1, 1 });
}

व्योम ggml_vk_lightning_indexer(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    स्थिर ggml_tensor * q = dst->src[0];
    स्थिर ggml_tensor * k = dst->src[1];
    स्थिर ggml_tensor * w = dst->src[2];
    स्थिर ggml_tensor * m = dst->src[3];

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, q, k, w, dst, dst->op);
    GGML_ASSERT(pipeline != nullptr);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    स्थिर uint32_t n_kv      = k->ne[2];
    स्थिर uint32_t n_heads   = q->ne[1];
    स्थिर uint32_t n_tokens  = q->ne[2];
    स्थिर uint32_t n_streams = q->ne[3];
    स्थिर uint32_t n_masks   = m->ne[3];

    // one workgroup per tile of 64 keys and 8 tokens, see lightning_indexer.comp
    स्थिर uint32_t n_tiles_kv = CEIL_DIV(n_kv, 64);
    स्थिर uint32_t n_tiles_t  = CEIL_DIV(n_tokens, 8);

    // q, w and dst are f32 and m is f16, so their strides are passed in elements;
    // k may be quantized, so its strides stay in bytes
    स्थिर uint32_t q_nb1 = q->nb[1] / माप(भग्न);
    स्थिर uint32_t q_nb2 = q->nb[2] / माप(भग्न);
    स्थिर uint32_t q_nb3 = q->nb[3] / माप(भग्न);
    स्थिर uint32_t k_nb2 = k->nb[2];
    स्थिर uint32_t k_nb3 = k->nb[3];
    स्थिर uint32_t w_nb1 = w->nb[1] / माप(भग्न);
    स्थिर uint32_t w_nb3 = w->nb[3] / माप(भग्न);
    स्थिर uint32_t m_nb1 = m->nb[1] / माप(ggml_fp16_t);
    स्थिर uint32_t m_nb3 = m->nb[3] / माप(ggml_fp16_t);
    स्थिर uint32_t d_nb1 = dst->nb[1] / माप(भग्न);
    स्थिर uint32_t d_nb3 = dst->nb[3] / माप(भग्न);

    स्थिर vk_op_lightning_indexer_धकेलो_constants pc = {
        n_kv, n_heads, n_tokens, n_masks,
        q_nb1, q_nb2, q_nb3,
        k_nb2, k_nb3,
        w_nb1, w_nb3,
        m_nb1, m_nb3,
        d_nb1, d_nb3,
    };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        {ggml_vk_tensor_subbuffer(ctx, q), ggml_vk_tensor_subbuffer(ctx, k), ggml_vk_tensor_subbuffer(ctx, w), ggml_vk_tensor_subbuffer(ctx, m), ggml_vk_tensor_subbuffer(ctx, dst)},
        pc, {n_tiles_kv, n_tiles_t, n_streams});
}

व्योम ggml_vk_gated_delta_net(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    स्थिर ggml_tensor * src_q     = dst->src[0];
    स्थिर ggml_tensor * src_v     = dst->src[2];
    स्थिर ggml_tensor * src_beta  = dst->src[4];

    GGML_ASSERT(dst->बफर != nullptr);

    स्थिर uint32_t S_v      = (uint32_t)src_v->ne[0];
    स्थिर uint32_t H        = (uint32_t)src_v->ne[1];
    स्थिर uint32_t n_tokens = (uint32_t)src_v->ne[2];
    स्थिर uint32_t n_seqs   = (uint32_t)src_v->ne[3];

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    स्थिर uint32_t K = (uint32_t)ggml_get_op_params_i32(dst, 0);

    स्थिर uint32_t s_off = S_v * H * n_tokens * n_seqs;

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, dst->src[0], dst->src[1], dst->src[2], dst, dst->op);
    GGML_ASSERT(pipeline != nullptr);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
    vk_subbuffer src_buf[6] = {};
    क्रम (पूर्णांक i = 0; i < 6; i++) {
        src_buf[i] = ggml_vk_tensor_subbuffer(ctx, dst->src[i]);
    }

    स्थिर uint32_t sq1 = (uint32_t)(src_q->nb[1] / माप(भग्न));
    स्थिर uint32_t sq2 = (uint32_t)(src_q->nb[2] / माप(भग्न));
    स्थिर uint32_t sq3 = (uint32_t)(src_q->nb[3] / माप(भग्न));
    स्थिर uint32_t sv1 = (uint32_t)(src_v->nb[1] / माप(भग्न));
    स्थिर uint32_t sv2 = (uint32_t)(src_v->nb[2] / माप(भग्न));
    स्थिर uint32_t sv3 = (uint32_t)(src_v->nb[3] / माप(भग्न));
    स्थिर uint32_t sb1 = (uint32_t)(src_beta->nb[1] / माप(भग्न));
    स्थिर uint32_t sb2 = (uint32_t)(src_beta->nb[2] / माप(भग्न));
    स्थिर uint32_t sb3 = (uint32_t)(src_beta->nb[3] / माप(भग्न));

    स्थिर uint32_t neq1 = (uint32_t)src_q->ne[1];
    स्थिर uint32_t rq3  = (uint32_t)(src_v->ne[3] / src_q->ne[3]);

    स्थिर भग्न scale = 1.0f / sqrtf((भग्न)S_v);
    स्थिर vk_op_gated_delta_net_धकेलो_constants pc = {
        H, n_tokens, n_seqs, s_off,
        sq1, sq2, sq3,
        sv1, sv2, sv3,
        sb1, sb2, sb3,
        neq1, rq3,
        scale,
        K
    };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        {src_buf[0], src_buf[1], src_buf[2], src_buf[3], src_buf[4], src_buf[5], dst_buf},
        pc, { H, n_seqs, S_v });
}

व्योम ggml_vk_ssm_scan(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];
    स्थिर ggml_tensor * src2 = dst->src[2];
    स्थिर ggml_tensor * src3 = dst->src[3];
    स्थिर ggml_tensor * src4 = dst->src[4];
    स्थिर ggml_tensor * src5 = dst->src[5];

    GGML_ASSERT(dst->बफर != nullptr);

    स्थिर uint32_t शीर्ष_dim = src0->ne[1];
    स्थिर uint32_t n_शीर्ष = src1->ne[1];
    स्थिर uint32_t n_group = src4->ne[1];
    स्थिर uint32_t n_tok = src1->ne[2];
    स्थिर uint32_t n_seq = src1->ne[3];

    bool is_mamba2 = (src3->nb[1] == माप(भग्न));
    GGML_ASSERT(is_mamba2);

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, src0, src1, src2, dst, dst->op);
    GGML_ASSERT(pipeline != nullptr);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    स्थिर int64_t s_off = ggml_nelements(src1) * माप(भग्न);

    स्थिर vk_op_ssm_scan_धकेलो_constants pc = {
        (uint32_t)src0->nb[2], (uint32_t)src0->nb[3],
        (uint32_t)src1->nb[2], (uint32_t)src1->nb[3],
        (uint32_t)src2->nb[1], (uint32_t)src2->nb[2],
        (uint32_t)src3->nb[1],
        (uint32_t)src4->nb[2], (uint32_t)src4->nb[3],
        (uint32_t)src5->nb[2], (uint32_t)src5->nb[3],
        (uint32_t)s_off,
        n_शीर्ष, शीर्ष_dim, n_group, n_tok,
        n_seq, (uint32_t) ggml_get_op_params_i32(dst, 0)
    };

    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
    vk_subbuffer src_buf[7] = {};
    क्रम (पूर्णांक i = 0; i < 7 && dst->src[i] != nullptr; i++) {
        src_buf[i] = ggml_vk_tensor_subbuffer(ctx, dst->src[i]);
    }

    std::array<uint32_t, 3> elements;

    स्थिर uint32_t d_state = src0->ne[0];
    uint32_t अंक_subgroups = d_state / ctx->device->subgroup_आकार;
    स्थिर uint32_t अंक_workgroups_x = CEIL_DIV(n_शीर्ष * शीर्ष_dim, अंक_subgroups);
    स्थिर uint32_t अंक_workgroups_y = n_seq;
    elements = { अंक_workgroups_x, अंक_workgroups_y, 1 };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        {src_buf[0], src_buf[1], src_buf[2], src_buf[3], src_buf[4], src_buf[5], src_buf[6], dst_buf},
        pc, elements);
}

व्योम ggml_vk_ssm_conv(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर काष्ठा ggml_cgraph * cgraph, पूर्णांक गाँठ_idx) {
    ggml_tensor * conv = cgraph->nodes[गाँठ_idx];
    स्थिर ggml_tensor * src0 = conv->src[0];
    स्थिर ggml_tensor * src1 = conv->src[1];

    // Pick the destination tensor (last node in the fused chain) and the optional bias.
    // Fusion modes: 0 = ssm_conv, 1 = ssm_conv+silu, 2 = ssm_conv+add(bias)+silu.
    ggml_tensor * dst = conv;
    स्थिर ggml_tensor * bias = nullptr;

    अगर (ctx->अंक_additional_fused_ops == 1) {
        dst = cgraph->nodes[गाँठ_idx + 1]; // silu
    } अन्यथा अगर (ctx->अंक_additional_fused_ops == 2) {
        ggml_tensor * add = cgraph->nodes[गाँठ_idx + 1];
        bias = (add->src[0] == conv) ? add->src[1] : add->src[0];
        dst = cgraph->nodes[गाँठ_idx + 2]; // silu
    }

    // The shader always declares 4 bindings; bind src0 as a dummy when bias isn't fused.
    स्थिर ggml_tensor * src2 = bias ? bias : src0;

    ggml_vk_op_f32<vk_op_ssm_conv_धकेलो_constants>(ctx, subctx, src0, src1, src2, nullptr, dst, GGML_OP_SSM_CONV, {
        (uint32_t)src0->nb[1], (uint32_t)src0->nb[2],
        (uint32_t)src1->nb[1],
        (uint32_t)dst->nb[0], (uint32_t)dst->nb[1], (uint32_t)dst->nb[2],
        (uint32_t)src1->ne[0],
        (uint32_t)src0->ne[0],
        (uint32_t)src0->ne[1],
        (uint32_t)dst->ne[1],
        (uint32_t)dst->ne[2],
    });
}

अटल व्योम ggml_vk_op_f32_opt_step_adamw(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst, स्थिर vk_op_धकेलो_constants&& pc) {
    स्थिर ggml_tensor * x = dst->src[0];
    स्थिर ggml_tensor * g = dst->src[1];
    स्थिर ggml_tensor * gm = dst->src[2];
    स्थिर ggml_tensor * gv = dst->src[3];
    स्थिर ggml_tensor * p = dst->src[4];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(g->type == GGML_TYPE_F32);
    GGML_ASSERT(gm->type == GGML_TYPE_F32);
    GGML_ASSERT(gv->type == GGML_TYPE_F32);
    GGML_ASSERT(p->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->बफर != nullptr);
    GGML_ASSERT(ggml_is_contiguous(x));
    GGML_ASSERT(ggml_is_contiguous(g));
    GGML_ASSERT(ggml_is_contiguous(gm));
    GGML_ASSERT(ggml_is_contiguous(gv));
    GGML_ASSERT(ggml_is_contiguous(p));
    GGML_ASSERT(ggml_are_same_shape(x, g));
    GGML_ASSERT(ggml_are_same_shape(x, gm));
    GGML_ASSERT(ggml_are_same_shape(x, gv));
    GGML_ASSERT(ggml_nelements(p) == 7);

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, g, gm, gv, dst, GGML_OP_OPT_STEP_ADAMW);
    GGML_ASSERT(pipeline != nullptr);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer x_buf = ggml_vk_tensor_subbuffer(ctx, x);
    vk_subbuffer g_buf = ggml_vk_tensor_subbuffer(ctx, g);
    vk_subbuffer gm_buf = ggml_vk_tensor_subbuffer(ctx, gm);
    vk_subbuffer gv_buf = ggml_vk_tensor_subbuffer(ctx, gv);
    vk_subbuffer p_buf = ggml_vk_tensor_subbuffer(ctx, p);

    std::array<uint32_t, 3> elements = { (uint32_t)ggml_nelements(x), 1, 1 };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        {x_buf, g_buf, gm_buf, gv_buf, p_buf},
        pc, elements);
}

व्योम ggml_vk_opt_step_adamw(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    स्थिर माप_प्रकार n = ggml_nelements(dst->src[0]);

    ggml_vk_op_f32_opt_step_adamw(
        ctx, subctx, dst,
        { (uint32_t)n, 0, 0.0f, 0.0f, 0.0f, 0.0f }
    );
}

व्योम ggml_vk_opt_step_sgd(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, स्थिर ggml_tensor * src1, स्थिर ggml_tensor * src2, ggml_tensor * dst) {
    स्थिर माप_प्रकार n = ggml_nelements(dst->src[0]);

    ggml_vk_op_f32<vk_op_धकेलो_constants>(ctx, subctx, src0, src1, src2, nullptr, dst, GGML_OP_OPT_STEP_SGD, { (uint32_t)n, 0, 0.0f, 0.0f, 0.0f, 0.0f });
}

व्योम ggml_vk_concat(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, स्थिर ggml_tensor * src1, ggml_tensor * dst) {
    पूर्णांक * op_params = (पूर्णांक *)dst->op_params;

    स्थिर uint32_t unit_आकार = ggml_vk_concat_unit_आकार(dst->type);
    स्थिर uint32_t units_per_block = ggml_type_आकार(dst->type) / unit_आकार;
    स्थिर uint32_t block_आकार_2 = ggml_blck_आकार(dst->type);
    स्थिर bool quantized = ggml_is_quantized(dst->type);

    // Address dimension 0 in packed storage units; higher strides may be noncontiguous.
    स्थिर uint32_t ne00 = src0->ne[0] / block_आकार_2 * units_per_block;
    स्थिर uint32_t ne10 = src1->ne[0] / block_आकार_2 * units_per_block;
    स्थिर uint32_t ne20 =  dst->ne[0] / block_आकार_2 * units_per_block;
    स्थिर uint32_t nb00 = quantized ? 1 : src0->nb[0] / unit_आकार;
    स्थिर uint32_t nb10 = quantized ? 1 : src1->nb[0] / unit_आकार;
    स्थिर uint32_t nb20 = quantized ? 1 :  dst->nb[0] / unit_आकार;

    vk_op_concat_धकेलो_constants pc {{
        ne20 * (uint32_t)dst->ne[1] * (uint32_t)dst->ne[2] * (uint32_t)dst->ne[3],
        ne00, (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], nb00, (uint32_t)src0->nb[1] / unit_आकार, (uint32_t)src0->nb[2] / unit_आकार, (uint32_t)src0->nb[3] / unit_आकार,
        ne10, (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], nb10, (uint32_t)src1->nb[1] / unit_आकार, (uint32_t)src1->nb[2] / unit_आकार, (uint32_t)src1->nb[3] / unit_आकार,
        ne20, (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], nb20, (uint32_t) dst->nb[1] / unit_आकार, (uint32_t) dst->nb[2] / unit_आकार, (uint32_t) dst->nb[3] / unit_आकार,
        0,
        0.0f, 0.0f, op_params[0],
    }};
    ggml_vk_op_f32<vk_op_concat_धकेलो_constants>(ctx, subctx, src0, src1, nullptr, nullptr, dst, GGML_OP_CONCAT, std::move(pc));
}

व्योम ggml_vk_upscale(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, ggml_tensor * dst) {
    स्थिर uint32_t src0_type_आकार = ggml_type_आकार(src0->type);
    स्थिर uint32_t mode = (uint32_t)ggml_get_op_params_i32(dst, 0);

    GGML_TENSOR_UNARY_OP_LOCALS

    भग्न sf0 = (भग्न)ne0 / ne00;
    भग्न sf1 = (भग्न)ne1 / ne01;
    भग्न sf2 = (भग्न)ne2 / ne02;
    भग्न sf3 = (भग्न)ne3 / ne03;
    भग्न pixel_offset = 0.5f;

    अगर (mode & GGML_SCALE_FLAG_ALIGN_CORNERS) {
        sf0 = ne0 > 1 && ne00 > 1 ? (भग्न)(ne0 - 1) / (ne00 - 1) : sf0;
        sf1 = ne1 > 1 && ne01 > 1 ? (भग्न)(ne1 - 1) / (ne01 - 1) : sf1;
        pixel_offset = 0.0f;
    }

    ggml_vk_op_f32<vk_op_upscale_धकेलो_constants>(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_UPSCALE, {
        (uint32_t)ggml_nelements(dst), 0, 0,
        (uint32_t)ne00, (uint32_t)ne01,
        (uint32_t)nb00 / src0_type_आकार, (uint32_t)nb01 / src0_type_आकार, (uint32_t)nb02 / src0_type_आकार, (uint32_t)nb03 / src0_type_आकार,
        (uint32_t)ne0, (uint32_t)ne1, (uint32_t)ne2, (uint32_t)ne3,
        sf0, sf1, sf2, sf3, pixel_offset
    });
}

व्योम ggml_vk_scale(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_unary_धकेलो_constants p = vk_op_unary_धकेलो_constants_init(src0, dst);
    p.param1 = ggml_get_op_params_f32(dst, 0);
    p.param2 = ggml_get_op_params_f32(dst, 1);

    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_SCALE, std::move(p));
}

व्योम ggml_vk_sqr(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, ggml_tensor * dst) {
    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_SQR, vk_op_unary_धकेलो_constants_init(src0, dst));
}

व्योम ggml_vk_sqrt(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, ggml_tensor * dst) {
    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_SQRT, vk_op_unary_धकेलो_constants_init(src0, dst));
}

व्योम ggml_vk_add1(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, स्थिर ggml_tensor * src1, ggml_tensor * dst) {
    स्थिर uint32_t src0_type_आकार = ggml_type_आकार(src0->type);
    स्थिर uint32_t src1_type_आकार = ggml_type_आकार(src1->type);
    स्थिर uint32_t dst_type_आकार = ggml_type_आकार(dst->type);

    ggml_vk_op_f32<vk_op_द्विआधारी_धकेलो_constants>(ctx, subctx, src0, src1, nullptr, nullptr, dst, GGML_OP_ADD1, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_आकार, (uint32_t)src0->nb[1] / src0_type_आकार, (uint32_t)src0->nb[2] / src0_type_आकार, (uint32_t)src0->nb[3] / src0_type_आकार,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_आकार, (uint32_t)src1->nb[1] / src1_type_आकार, (uint32_t)src1->nb[2] / src1_type_आकार, (uint32_t)src1->nb[3] / src1_type_आकार,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_आकार, (uint32_t) dst->nb[1] /  dst_type_आकार, (uint32_t) dst->nb[2] /  dst_type_आकार, (uint32_t) dst->nb[3] /  dst_type_आकार,
        0,
        0.0f, 0.0f, 0,
    });
}

व्योम ggml_vk_arange(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_vk_arange(dst=" << dst << ", ne=" << ggml_nelements(dst) << ")");

    vk_op_धकेलो_constants pc = {
        (uint32_t)ggml_nelements(dst),
        1,
        ggml_get_op_params_f32(dst, 0),
        ggml_get_op_params_f32(dst, 2),
        0.0f, 0.0f,
    };

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, nullptr, nullptr, nullptr, dst, GGML_OP_ARANGE);
    GGML_ASSERT(pipeline != nullptr);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst, false);

    std::array<uint32_t, 3> elements = { (uint32_t)ggml_nelements(dst), 1, 1 };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { dst_buf }, pc, elements);
}

व्योम ggml_vk_fill(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_vk_fill(dst=" << dst << ", ne=" << ggml_nelements(dst) << ")");
    स्थिर uint64_t n = ggml_nelements(dst);
    GGML_ASSERT(n > 0);

    vk_op_धकेलो_constants pc = {
        (uint32_t)n,
        1,
        ggml_get_op_params_f32(dst, 0),
        0.0f,
        0.0f, 0.0f,
    };

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, nullptr, nullptr, nullptr, dst, GGML_OP_FILL);
    GGML_ASSERT(pipeline != nullptr);

    // Split the task distribution to 2D to avoid exceeding maxComputeWorkGroupCount
    स्थिर uint32_t योग_wg = CEIL_DIV(n, pipeline->wg_denoms[0]);
    स्थिर uint32_t wg_x = std::min(योग_wg, ctx->device->properties.limits.max_Compute_Work_Group_गिनती[0]);
    स्थिर uint32_t wg_y = CEIL_DIV(योग_wg, wg_x);
    GGML_ASSERT(wg_y <= ctx->device->properties.limits.max_Compute_Work_Group_गिनती[1]);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst, false);

    std::array<uint32_t, 3> elements = { wg_x * pipeline->wg_denoms[0], wg_y * pipeline->wg_denoms[1], 1 };
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { dst_buf }, pc, elements);
}

व्योम ggml_vk_sin(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, ggml_tensor * dst) {
    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_SIN, vk_op_unary_धकेलो_constants_init(src0, dst));
}

व्योम ggml_vk_cos(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, ggml_tensor * dst) {
    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_COS, vk_op_unary_धकेलो_constants_init(src0, dst));
}

व्योम ggml_vk_log(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, ggml_tensor * dst) {
    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_LOG, vk_op_unary_धकेलो_constants_init(src0, dst));
}

व्योम ggml_vk_tri(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_unary_धकेलो_constants p = vk_op_unary_धकेलो_constants_init(src0, dst);
    p.param1 = ggml_get_op_params_f32(dst, 0);

    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_TRI, std::move(p));
}

व्योम ggml_vk_diag(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_unary_धकेलो_constants p = vk_op_unary_धकेलो_constants_init(src0, dst, ggml_nelements(dst));

    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_DIAG, std::move(p));
}

व्योम ggml_vk_clamp(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_unary_धकेलो_constants p = vk_op_unary_धकेलो_constants_init(src0, dst);
    p.param1 = ggml_get_op_params_f32(dst, 0);
    p.param2 = ggml_get_op_params_f32(dst, 1);

    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_CLAMP, std::move(p));
}

व्योम ggml_vk_pad(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_pad_धकेलो_constants p = vk_op_pad_धकेलो_constants_init(src0, dst);
    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_PAD, std::move(p));
}

व्योम ggml_vk_pad_reflect_1d(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, ggml_tensor * dst) {
    स्थिर uint32_t p0 = (uint32_t)dst->op_params[0];
    स्थिर uint32_t p1 = (uint32_t)dst->op_params[1];

    vk_op_unary_धकेलो_constants p = vk_op_unary_धकेलो_constants_init(src0, dst, ggml_nelements(dst));
    स_नकल(&p.param1, &p0, माप(भग्न));
    स_नकल(&p.param2, &p1, माप(भग्न));

    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_PAD_REFLECT_1D, std::move(p));
}

व्योम ggml_vk_roll(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, ggml_tensor * dst) {
    स्थिर int32_t s0 = ggml_get_op_params_i32(dst, 0);
    स्थिर int32_t s1 = ggml_get_op_params_i32(dst, 1);
    स्थिर int32_t s2 = ggml_get_op_params_i32(dst, 2);
    स्थिर int32_t s3 = ggml_get_op_params_i32(dst, 3);
    स्थिर uint32_t s01_packed = ((s0 + 0x8000) << 16) | (s1 + 0x8000);
    स्थिर uint32_t s23_packed = ((s2 + 0x8000) << 16) | (s3 + 0x8000);

    vk_op_unary_धकेलो_constants p = vk_op_unary_धकेलो_constants_init(src0, dst);
    स_नकल(&p.param1, &s01_packed, माप(भग्न));
    स_नकल(&p.param2, &s23_packed, माप(भग्न));

    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_ROLL, std::move(p));
}

व्योम ggml_vk_repeat(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_unary_धकेलो_constants p = vk_op_unary_धकेलो_constants_init(src0, dst, ggml_nelements(dst));
    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_REPEAT, std::move(p));
}

व्योम ggml_vk_repeat_back(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_unary_धकेलो_constants p = vk_op_unary_धकेलो_constants_init(src0, dst, ggml_nelements(dst));
    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_REPEAT_BACK, std::move(p));
}

व्योम ggml_vk_cpy(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, ggml_tensor * dst) {
    uint32_t ne = (uint32_t)ggml_nelements(src0);
    अगर (ggml_is_quantized(src0->type) && ggml_is_quantized(dst->type)) {
        // Convert from number of logical elements to 2- or 4-byte units.
        ne /= ggml_blck_आकार(src0->type);
        अगर ((ggml_type_आकार(src0->type) % 4) == 0) {
            ne *= ggml_type_आकार(src0->type) / 4;
        } अन्यथा {
            ne *= ggml_type_आकार(src0->type) / 2;
        }
    }

    vk_op_unary_धकेलो_constants p = vk_op_unary_धकेलो_constants_init(src0, dst, ne);
    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_CPY, std::move(p));
}

व्योम ggml_vk_set_rows(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, स्थिर ggml_tensor * src1, ggml_tensor * dst) {
    स्थिर uint32_t src0_type_आकार = ggml_type_आकार(src0->type);
    स्थिर uint32_t src1_type_आकार = ggml_type_आकार(src1->type);
    स्थिर uint32_t dst_type_आकार = ggml_type_आकार(dst->type);

    // Skip empty skip_rows operations. For most ops the empty check at the start
    // of ggml_vk_build_graph is sufficient, but set_rows can have a nonempty dst
    // with empty srcs.
    अगर (ggml_is_empty(src0) || ggml_is_empty(src1)) {
        वापस;
    }

    ggml_vk_op_f32<vk_op_द्विआधारी_धकेलो_constants>(ctx, subctx, src0, src1, nullptr, nullptr, dst, GGML_OP_SET_ROWS, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_आकार, (uint32_t)src0->nb[1] / src0_type_आकार, (uint32_t)src0->nb[2] / src0_type_आकार, (uint32_t)src0->nb[3] / src0_type_आकार,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_आकार, (uint32_t)src1->nb[1] / src1_type_आकार, (uint32_t)src1->nb[2] / src1_type_आकार, (uint32_t)src1->nb[3] / src1_type_आकार,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_आकार, (uint32_t) dst->nb[1] /  dst_type_आकार, (uint32_t) dst->nb[2] /  dst_type_आकार, (uint32_t) dst->nb[3] /  dst_type_आकार,
        0,
        0.0f, 0.0f, 0,
    });
}

व्योम ggml_vk_silu_back(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, स्थिर ggml_tensor * src1, ggml_tensor * dst) {
    ggml_vk_op_f32<vk_op_धकेलो_constants>(ctx, subctx, src0, src1, nullptr, nullptr, dst, GGML_OP_SILU_BACK, { (uint32_t)ggml_nelements(src0), 0, 0.0f, 0.0f, 0.0f, 0.0f });
}

व्योम ggml_vk_norm(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, ggml_tensor * dst) {
    भग्न * op_params = (भग्न *)dst->op_params;
    vk_op_unary_धकेलो_constants p = vk_op_unary_धकेलो_constants_init(src0, dst);
    p.param1 = op_params[0];

    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_NORM, std::move(p));
}

व्योम ggml_vk_group_norm(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, ggml_tensor * dst) {
    स्थिर पूर्णांक * int_op_params = (स्थिर पूर्णांक *)dst->op_params;
    स्थिर भग्न * float_op_params = (स्थिर भग्न *)dst->op_params;

    स्थिर uint32_t अंक_groups = int_op_params[0];
    स्थिर भग्न eps = float_op_params[1];
    स्थिर uint32_t group_आकार = src0->ne[0] * src0->ne[1] * ((src0->ne[2] + अंक_groups - 1) / अंक_groups);

    ggml_vk_op_f32<vk_op_धकेलो_constants>(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_GROUP_NORM, { group_आकार, 0, eps, 0.0f, 0.0f, 0.0f });
}

अटल uint32_t ggml_vk_rms_अंक_partials(ggml_backend_vk_context * ctx, स्थिर ggml_tensor *गाँठ) {
    स्थिर uint32_t ne = (uint32_t)गाँठ->ne[0];
    स्थिर uint32_t denom = ctx->device->pipeline_add_rms[0][0][0]->wg_denoms[0];
    स्थिर uint32_t अंक_partials = CEIL_DIV(ne, denom);
    वापस अंक_partials;
}

uint32_t ggml_vk_rms_partials_आकार(ggml_backend_vk_context * ctx, स्थिर ggml_tensor *गाँठ) {
    स्थिर uint32_t अंक_partials = ggml_vk_rms_अंक_partials(ctx, गाँठ);
    स्थिर uint32_t अंक_bytes = ROUNDUP_POW2(अंक_partials * माप(uint32_t), ctx->device->partials_binding_alignment);
    वापस अंक_bytes;
}

अटल vk_op_rope_धकेलो_constants ggml_vk_make_rope_constants(स्थिर ggml_tensor *dst, स्थिर ggml_tensor *src0, स्थिर bool has_ff, bool backprop, स्थिर uint32_t set_rows_stride) {
    स्थिर पूर्णांक n_dims        = ((स्थिर int32_t *) dst->op_params)[1];
    स्थिर पूर्णांक mode          = ((स्थिर int32_t *) dst->op_params)[2];
    स्थिर पूर्णांक n_offs        = ((स्थिर int32_t *) dst->op_params)[15];
    // const int n_ctx         = ((const int32_t *) dst->op_params)[3];
    स्थिर पूर्णांक n_ctx_orig    = ((स्थिर int32_t *) dst->op_params)[4];
    स्थिर भग्न freq_base   = ((स्थिर भग्न *)   dst->op_params)[5];
    स्थिर भग्न freq_scale  = ((स्थिर भग्न *)   dst->op_params)[6];
    स्थिर भग्न ext_factor  = ((स्थिर भग्न *)   dst->op_params)[7];
    स्थिर भग्न attn_factor = ((स्थिर भग्न *)   dst->op_params)[8];
    स्थिर भग्न beta_fast   = ((स्थिर भग्न *)   dst->op_params)[9];
    स्थिर भग्न beta_slow   = ((स्थिर भग्न *)   dst->op_params)[10];
    पूर्णांक sections[4] {};
    अगर (mode & GGML_ROPE_TYPE_MROPE) {
        स_नकल(sections, (स्थिर int32_t *) dst->op_params + 11, माप(पूर्णांक)*4);
    }

    स्थिर bool is_imrope = mode == GGML_ROPE_TYPE_IMROPE;

    भग्न corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    स्थिर भग्न theta_scale = powf(freq_base, -2.0f/n_dims);

    uint32_t nb01 = src0->nb[1] / ggml_type_आकार(src0->type);
    uint32_t nb02 = src0->nb[2] / ggml_type_आकार(src0->type);
    uint32_t nb03 = src0->nb[3] / ggml_type_आकार(src0->type);

    uint32_t nb11 = dst->nb[1] / ggml_type_आकार(dst->type);
    uint32_t nb12 = dst->nb[2] / ggml_type_आकार(dst->type);
    uint32_t nb13 = dst->nb[3] / ggml_type_आकार(dst->type);

    vk_op_rope_धकेलो_constants rope {
        (uint32_t)mode, (uint32_t)ggml_nrows(src0), (uint32_t)n_dims, (uint32_t)n_offs, freq_scale,
        freq_base, ext_factor, attn_factor, {corr_dims[0], corr_dims[1]}, theta_scale, has_ff,
        { sections[0], sections[1], sections[2], sections[3] }, is_imrope, backprop, set_rows_stride,

        (uint32_t)src0->ne[0],
        (uint32_t)src0->ne[1],
        (uint32_t)src0->ne[2],
        nb01, nb02, nb03,
        nb11, nb12, nb13,
        0, 0, // a_offset, d_offset filled in by init_pushconst_tensor_offsets
    };

    वापस rope;
}

अटल व्योम ggml_vk_rms_norm_finish(ggml_backend_vk_context * ctx, स्थिर ggml_tensor * src0) {
    अगर (ctx->do_add_rms_partials_offset_calculation) {
        ctx->prealloc_आकार_add_rms_partials_offset += ggml_vk_rms_partials_आकार(ctx, src0);
        ctx->do_add_rms_partials = false;
        ctx->do_add_rms_partials_offset_calculation = false;
    }
}

व्योम ggml_vk_rms_norm(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर काष्ठा ggml_cgraph * cgraph, पूर्णांक गाँठ_idx, भग्न * op_params) {
    ggml_tensor * rms = cgraph->nodes[गाँठ_idx];
    स्थिर ggml_tensor * src0 = rms->src[0];

    अगर (ctx->fused_rms_norm_mode == RMS_NORM_VIEW_SET_ROWS) {
        GGML_ASSERT(ctx->अंक_additional_fused_ops == 2);
        ggml_tensor * set_rows = cgraph->nodes[गाँठ_idx + 2];
        स्थिर ggml_tensor * indices = set_rows->src[1];
        vk_op_द्विआधारी_धकेलो_constants pc = ggml_vk_rms_norm_धकेलो_constants(src0, src0, set_rows, op_params[0], 0);
        init_pushconst_tensor_offsets(ctx, pc, src0, src0, nullptr, nullptr, set_rows);

        vk_pipeline pipeline = set_rows->type == GGML_TYPE_F16 ?
            ctx->device->pipeline_rms_norm_set_rows_f32_f16 : ctx->device->pipeline_rms_norm_set_rows_f32_f32;
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
            {
                ggml_vk_tensor_subbuffer(ctx, src0, true),
                ggml_vk_tensor_subbuffer(ctx, src0, true),
                ggml_vk_tensor_subbuffer(ctx, set_rows, true),
                ggml_vk_tensor_subbuffer(ctx, indices),
            }, pc, {
                (uint32_t)src0->ne[1],
                std::min((uint32_t)src0->ne[2], ctx->device->properties.limits.max_Compute_Work_Group_गिनती[1]),
                std::min((uint32_t)src0->ne[3], ctx->device->properties.limits.max_Compute_Work_Group_गिनती[2]),
            });
        ggml_vk_rms_norm_finish(ctx, src0);
        वापस;
    }

    अगर (ctx->fused_rms_norm_mode == RMS_NORM_MUL_ADD || ctx->fused_rms_norm_mode == RMS_NORM_MUL_ADD_MUL) {
        ggml_tensor * mul = cgraph->nodes[गाँठ_idx + 1];
        ggml_tensor * add = cgraph->nodes[गाँठ_idx + 2];
        स्थिर ggml_tensor * weight = mul->src[0] == rms ? mul->src[1] : mul->src[0];
        स्थिर ggml_tensor * residual = add->src[0] == mul ? add->src[1] : add->src[0];
        स्थिर bool do_post_multiply = ctx->fused_rms_norm_mode == RMS_NORM_MUL_ADD_MUL;
        GGML_ASSERT(ctx->अंक_additional_fused_ops == (do_post_multiply ? 3 : 2));
        ggml_tensor * dst = do_post_multiply ? cgraph->nodes[गाँठ_idx + 3] : add;
        स्थिर ggml_tensor * post_scale = do_post_multiply ?
            (dst->src[0] == add ? dst->src[1] : dst->src[0]) : src0;

        स्थिर uint32_t अंक_partials = ctx->do_add_rms_partials ? ggml_vk_rms_अंक_partials(ctx, dst) : 0;
        vk_op_द्विआधारी_धकेलो_constants pc = ggml_vk_rms_norm_धकेलो_constants(src0, weight, dst, op_params[0], अंक_partials);
        init_pushconst_tensor_offsets(ctx, pc, src0, weight, residual, post_scale, dst);

        vk_pipeline pipeline;
        अगर (ctx->do_add_rms_partials) {
            pipeline = do_post_multiply ?
                ctx->device->pipeline_rms_norm_mul_add_mul_partials_f32 : ctx->device->pipeline_rms_norm_mul_add_partials_f32;
        } अन्यथा {
            pipeline = do_post_multiply ?
                ctx->device->pipeline_rms_norm_mul_add_mul_f32 : ctx->device->pipeline_rms_norm_mul_add_f32;
        }
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
        अगर (ctx->do_add_rms_partials) {
            ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
                {
                    ggml_vk_tensor_subbuffer(ctx, src0, true),
                    ggml_vk_tensor_subbuffer(ctx, weight, true),
                    ggml_vk_tensor_subbuffer(ctx, dst, true),
                    ggml_vk_subbuffer(ctx, ctx->prealloc_add_rms_partials, ctx->prealloc_आकार_add_rms_partials_offset),
                    ggml_vk_tensor_subbuffer(ctx, residual),
                    ggml_vk_tensor_subbuffer(ctx, post_scale),
                }, pc, { (uint32_t)CEIL_DIV(src0->ne[0], 128), 1, 1 });
        } अन्यथा {
            ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
                {
                    ggml_vk_tensor_subbuffer(ctx, src0, true),
                    ggml_vk_tensor_subbuffer(ctx, weight, true),
                    ggml_vk_tensor_subbuffer(ctx, dst, true),
                    ggml_vk_tensor_subbuffer(ctx, residual),
                    ggml_vk_tensor_subbuffer(ctx, post_scale),
                }, pc, {
                    (uint32_t)src0->ne[1],
                    std::min((uint32_t)src0->ne[2], ctx->device->properties.limits.max_Compute_Work_Group_गिनती[1]),
                    std::min((uint32_t)src0->ne[3], ctx->device->properties.limits.max_Compute_Work_Group_गिनती[2]),
                });
        }
        ggml_vk_rms_norm_finish(ctx, src0);
        वापस;
    }

    ggml_tensor * dst;
    स्थिर ggml_tensor * src1;

    अगर (ctx->fused_rms_norm_mode != RMS_NORM_गिनती) {
        ggml_tensor * mul = cgraph->nodes[गाँठ_idx + 1];
        dst = mul;
        src1 = mul->src[0] == rms ? mul->src[1] : mul->src[0];
    } अन्यथा {
        dst = rms;
        src1 = src0;
    }

    स्थिर uint32_t अंक_partials = ctx->do_add_rms_partials ? ggml_vk_rms_अंक_partials(ctx, dst) : 0;
    vk_op_द्विआधारी_धकेलो_constants bin = ggml_vk_rms_norm_धकेलो_constants(src0, src1, dst, op_params[0], अंक_partials);

    अगर (ctx->fused_rms_norm_mode == RMS_NORM_MUL_ROPE ||
        ctx->fused_rms_norm_mode == RMS_NORM_MUL_ROPE_VIEW_SET_ROWS) {
        अटल constexpr uint32_t max_tensors = 7;
        स्थिर ggml_tensor *tensors[max_tensors] {};

        ggml_tensor *rms = cgraph->nodes[गाँठ_idx + 0];
        ggml_tensor *mul = cgraph->nodes[गाँठ_idx + 1];
        ggml_tensor *rope = cgraph->nodes[गाँठ_idx + 2];

        ggml_tensor *other_src = mul->src[0] == rms ? mul->src[1] : mul->src[0];

        bool do_set_rows = ctx->fused_rms_norm_mode == RMS_NORM_MUL_ROPE_VIEW_SET_ROWS;
        GGML_ASSERT(ctx->अंक_additional_fused_ops == (do_set_rows ? 4 : 2));

        tensors[0] = rms->src[0];
        tensors[1] = other_src;
        tensors[2] = mul;
        tensors[3] = rope->src[1]; // pos
        tensors[4] = rope->src[2]; // ff
        tensors[5] = cgraph->nodes[गाँठ_idx + ctx->अंक_additional_fused_ops]; // dst
        tensors[6] = do_set_rows ? tensors[5]->src[1] : nullptr;
        स्थिर uint32_t set_rows_stride = do_set_rows ? tensors[5]->nb[1] / ggml_type_आकार(tensors[5]->type) : 0;

        vk_op_rms_norm_mul_rope_धकेलो_constants pc;
        pc.bin = bin;
        pc.rope = ggml_vk_make_rope_constants(rope, rope->src[0], tensors[4] != nullptr, false, set_rows_stride);

        vk_pipeline pipeline = tensors[5]->type == GGML_TYPE_F16 ? ctx->device->pipeline_rms_norm_mul_rope_f32_f16 : ctx->device->pipeline_rms_norm_mul_rope_f32_f32;

        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

        ggml_backend_vk_बफर_context * buf_ctx[max_tensors];
        vk_बफर buf[max_tensors];
        माप_प्रकार offset[max_tensors];
        bool uma[max_tensors];

        क्रम (uint32_t i = 0; i < max_tensors; ++i) {
            अगर (!tensors[i]) {
                // If any remaining descriptors are unused, just point them at src[0]
                buf[i] = buf[0];
                offset[i] = 0;
                जारी;
            }
            buf_ctx[i] = (ggml_backend_vk_बफर_context *)tensors[i]->बफर->context;
            buf[i] = nullptr;
            offset[i] = 0;
            uma[i] = false;

            अगर (ctx->device->uma) {
                ggml_vk_host_get(ctx->device, tensors[i]->तथ्य, buf[i], offset[i]);
                uma[i] = buf[i] != nullptr;
            }
            अगर (!uma[i]) {
                buf[i] = buf_ctx[i]->dev_बफर;
                offset[i] = vk_tensor_offset(tensors[i]) + tensors[i]->view_offs;
            }
            GGML_ASSERT(buf[i] != nullptr);
        }

        // a_offset is unused (the fused path reads from shared memory), but the rope/set_rows dst can be misaligned.
        // Round the binding offset down to the storage buffer alignment; the in-element shift goes in pc.rope.d_offset.
        pc.rope.d_offset = get_misalign_bytes(ctx, tensors[5]) / ggml_type_आकार(tensors[5]->type);
        offset[5] &= ~(माप_प्रकार(ctx->device->properties.limits.min_Storage_बफर_Offset_Alignment) - 1);

        std::array<uint32_t, 3> elements;
        elements = { (uint32_t)rms->src[0]->ne[1], (uint32_t)rms->src[0]->ne[2], (uint32_t)rms->src[0]->ne[3] };
        elements[1] = std::min(elements[1], ctx->device->properties.limits.max_Compute_Work_Group_गिनती[1]);
        elements[2] = std::min(elements[2], ctx->device->properties.limits.max_Compute_Work_Group_गिनती[2]);

        static_assert(max_tensors == 7);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
            {
                ggml_vk_subbuffer(ctx, buf[0], offset[0]),
                ggml_vk_subbuffer(ctx, buf[1], offset[1]),
                ggml_vk_subbuffer(ctx, buf[2], offset[2]),
                ggml_vk_subbuffer(ctx, buf[3], offset[3]),
                ggml_vk_subbuffer(ctx, buf[4], offset[4]),
                ggml_vk_subbuffer(ctx, buf[5], offset[5]),
                ggml_vk_subbuffer(ctx, buf[6], offset[6]),
            }, pc, elements);
    } अन्यथा {
        GGML_ASSERT(ctx->fused_rms_norm_mode == RMS_NORM_MUL || ctx->fused_rms_norm_mode == RMS_NORM_गिनती);
        ggml_vk_op_f32<vk_op_द्विआधारी_धकेलो_constants>(ctx, subctx, src0, src1, nullptr, nullptr, dst, GGML_OP_RMS_NORM, std::move(bin));
    }

    ggml_vk_rms_norm_finish(ctx, src0);
}

व्योम ggml_vk_rms_norm_back(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, स्थिर ggml_tensor * src1, ggml_tensor * dst) {
    भग्न * op_params = (भग्न *)dst->op_params;
    ggml_vk_op_f32<vk_op_धकेलो_constants>(ctx, subctx, src0, src1, nullptr, nullptr, dst, GGML_OP_RMS_NORM_BACK, { (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], op_params[0], 0.0f, 0.0f, 0.0f });
}

व्योम ggml_vk_l2_norm(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, ggml_tensor * dst) {
    स्थिर भग्न * op_params = (स्थिर भग्न *)dst->op_params;
    vk_op_unary_धकेलो_constants p = vk_op_unary_धकेलो_constants_init(src0, dst);
    p.param1 = op_params[0];
    ggml_vk_op_f32<vk_op_unary_धकेलो_constants>(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_L2_NORM, std::move(p));
}

व्योम ggml_vk_unary(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, ggml_tensor * dst) {
    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_UNARY, vk_op_unary_धकेलो_constants_init(src0, dst));
}

व्योम ggml_vk_xielu(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, ggml_tensor * dst) {
    भग्न * op_params = (भग्न *)dst->op_params;
    vk_op_unary_धकेलो_constants p = vk_op_unary_धकेलो_constants_init(src0, dst);
    p.param1 = op_params[1];
    p.param2 = op_params[2];
    p.param3 = op_params[3];
    p.param4 = op_params[4];
    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_UNARY, std::move(p));
}

व्योम ggml_vk_glu(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, स्थिर ggml_tensor * src1, ggml_tensor * dst) {
    स्थिर भग्न * op_params_f = (स्थिर भग्न *)dst->op_params;

    स्थिर bool swapped = (bool)dst->op_params[1];
    स्थिर bool split = src1 != nullptr;
    स्थिर भग्न alpha = op_params_f[2];
    स्थिर भग्न limit = op_params_f[3];

    अगर (!split) {
        GGML_ASSERT(src0->ne[0] / 2 == dst->ne[0]);
    } अन्यथा {
        GGML_ASSERT(src0->ne[0] == src1->ne[0]);
        GGML_ASSERT(src0->ne[0] == dst->ne[0]);
        GGML_ASSERT(src0->type == src1->type);
    }

    स्थिर uint32_t mode = split ? 2 : (swapped ? 1 : 0);
    स्थिर uint32_t src0_type_आकार = ggml_type_आकार(src0->type);
    स्थिर uint32_t src1_type_आकार = split ? ggml_type_आकार(src1->type) : src0_type_आकार;
    स्थिर uint32_t dst_type_आकार  = ggml_type_आकार(dst->type);

    ggml_vk_op_f32<vk_op_glu_धकेलो_constants>(ctx, subctx, src0, src1, nullptr, nullptr, dst, GGML_OP_GLU,
        {
            (uint32_t)ggml_nelements(dst),
            (uint32_t)src0->ne[0],
            (uint32_t)dst->ne[0],
            mode,
            alpha,
            limit,
            (uint32_t)(src0->nb[0] / src0_type_आकार),
            (uint32_t)(src0->nb[1] / src0_type_आकार),
            (uint32_t)(src0->nb[2] / src0_type_आकार),
            (uint32_t)(src0->nb[3] / src0_type_आकार),
            (uint32_t)((split ? src1->nb[0] : src0->nb[0]) / src1_type_आकार),
            (uint32_t)((split ? src1->nb[1] : src0->nb[1]) / src1_type_आकार),
            (uint32_t)((split ? src1->nb[2] : src0->nb[2]) / src1_type_आकार),
            (uint32_t)((split ? src1->nb[3] : src0->nb[3]) / src1_type_आकार),
            (uint32_t)(dst->nb[0] / dst_type_आकार),
            (uint32_t)(dst->nb[1] / dst_type_आकार),
            (uint32_t)(dst->nb[2] / dst_type_आकार),
            (uint32_t)(dst->nb[3] / dst_type_आकार),
            (uint32_t)dst->ne[1],
            (uint32_t)dst->ne[2],
            0,
            0, 0, 0, 0, 0, 0,
        });
}

व्योम ggml_vk_diag_mask_inf(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, ggml_tensor * dst) {
    int32_t * op_params = (int32_t *)dst->op_params;
    ggml_vk_op_f32<vk_op_diag_mask_धकेलो_constants>(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_DIAG_MASK_INF, { (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], op_params[0] });
}

व्योम ggml_vk_soft_max(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, स्थिर ggml_tensor * src1, स्थिर ggml_tensor * src2, ggml_tensor * dst) {
    भग्न * op_params = (भग्न *)dst->op_params;

    भग्न scale = op_params[0];
    भग्न max_bias = op_params[1];

    स्थिर uint32_t ncols =   (uint32_t)src0->ne[0];
    स्थिर uint32_t nrows_x = (uint32_t)ggml_nrows(src0);
    स्थिर uint32_t nrows_y = (uint32_t)src0->ne[1];

    स्थिर uint32_t ne12 = src1 ? (uint32_t)(src1->ne[2]) : 0u;
    स्थिर uint32_t ne13 = src1 ? (uint32_t)(src1->ne[3]) : 0u;
    स्थिर uint32_t nb11 = src1 ? (uint32_t)(src1->nb[1] / src1->nb[0]) : 0u;
    स्थिर uint32_t nb12 = src1 ? (uint32_t)(src1->nb[2] / src1->nb[0]) : 0u;
    स्थिर uint32_t nb13 = src1 ? (uint32_t)(src1->nb[3] / src1->nb[0]) : 0u;

    स्थिर uint32_t n_शीर्ष_kv   = src0->ne[2];
    स्थिर uint32_t n_शीर्ष_log2 = 1u << (uint32_t) floorf(log2f((भग्न) n_शीर्ष_kv));

    स्थिर भग्न m0 = powf(2.0f, -(max_bias       ) / n_शीर्ष_log2);
    स्थिर भग्न m1 = powf(2.0f, -(max_bias / 2.0f) / n_शीर्ष_log2);

    vk_op_soft_max_धकेलो_constants pc {
        ncols,
        src1 != nullptr ? nrows_y : (uint32_t)0,
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],
        ne12, ne13,
        nb11, nb12, nb13,
        scale, max_bias,
        m0, m1,
        n_शीर्ष_log2,
        nrows_x,
        src2 != nullptr
    };

    अगर (ncols <= 16384) {
        ggml_vk_op_f32<vk_op_soft_max_धकेलो_constants>(ctx, subctx, src0, src1, src2, nullptr, dst, GGML_OP_SOFT_MAX, std::move(pc));
    } अन्यथा {

        vk_subbuffer buf_a = ggml_vk_tensor_subbuffer(ctx, src0);
        vk_subbuffer buf_b = src1 ? ggml_vk_tensor_subbuffer(ctx, src1) : buf_a;
        vk_subbuffer buf_c = src2 ? ggml_vk_tensor_subbuffer(ctx, src2) : buf_a;
        vk_subbuffer buf_d = ggml_vk_tensor_subbuffer(ctx, dst);

        uint32_t elems_per_wg = 128 * 4;
        uint32_t अंक_wgs = CEIL_DIV(ncols, elems_per_wg);
        माप_प्रकार tmp_आकार = अंक_wgs * nrows_x * माप(भग्न);

        अगर (ctx->prealloc_आकार_x < tmp_आकार) {
            ctx->prealloc_आकार_x = tmp_आकार;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        अगर (ctx->prealloc_आकार_y < tmp_आकार) {
            ctx->prealloc_आकार_y = tmp_आकार;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        अगर (ctx->prealloc_x_need_sync || ctx->prealloc_y_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }

        vk_subbuffer buf_x = { ctx->prealloc_x, 0, tmp_आकार };
        vk_subbuffer buf_y = { ctx->prealloc_y, 0, tmp_आकार };

        // the partial results overwrite a matmul input converted into prealloc_y
        ctx->prealloc_y_last_pipeline_used = nullptr;
        ctx->prealloc_y_last_tensor_used = nullptr;

        std::array<uint32_t, 3> elements = { अंक_wgs, nrows_x, 1 };

        vk_pipeline pipeline1 = src1 && src1->type == GGML_TYPE_F16 ? ctx->device->pipeline_soft_max_large1_f32_f16 : ctx->device->pipeline_soft_max_large1_f32;
        vk_pipeline pipeline2 = src1 && src1->type == GGML_TYPE_F16 ? ctx->device->pipeline_soft_max_large2_f32_f16 : ctx->device->pipeline_soft_max_large2_f32;
        vk_pipeline pipeline3 = src1 && src1->type == GGML_TYPE_F16 ? ctx->device->pipeline_soft_max_large3_f32_f16 : ctx->device->pipeline_soft_max_large3_f32;

        ggml_pipeline_request_descriptor_sets(ctx, pipeline1, 1);
        ggml_pipeline_request_descriptor_sets(ctx, pipeline2, 1);
        ggml_pipeline_request_descriptor_sets(ctx, pipeline3, 1);

        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline1, { buf_a, buf_b, buf_c, buf_d, buf_x, buf_y }, pc, elements);
        ggml_vk_sync_buffers(ctx, subctx);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline2, { buf_a, buf_b, buf_c, buf_d, buf_x, buf_y }, pc, elements);
        ggml_vk_sync_buffers(ctx, subctx);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline3, { buf_a, buf_b, buf_c, buf_d, buf_x, buf_y }, pc, elements);

        ctx->prealloc_x_need_sync = true;
        ctx->prealloc_y_need_sync = true;
    }
}

व्योम ggml_vk_soft_max_back(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, स्थिर ggml_tensor * src1, ggml_tensor * dst) {
    भग्न * op_params = (भग्न *)dst->op_params;
    ggml_vk_op_f32<vk_op_धकेलो_constants>(ctx, subctx, src0, src1, nullptr, nullptr, dst, GGML_OP_SOFT_MAX_BACK, { (uint32_t)src0->ne[0], (uint32_t)ggml_nrows(src0), op_params[0], op_params[1], 0.0f, 0.0f });
}

व्योम ggml_vk_topk_moe(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_cgraph * cgraph, पूर्णांक गाँठ_idx) {
    topk_moe_mode mode = ctx->fused_topk_moe_mode;
    स्थिर bool has_bias = mode == TOPK_MOE_SIGMOID_NORM_BIAS || mode == TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS;
    ggml_tensor * logits = cgraph->nodes[गाँठ_idx + 0]->src[0];
    ggml_tensor * bias = mode == TOPK_MOE_SIGMOID_NORM_BIAS       ? cgraph->nodes[गाँठ_idx + 2]->src[1] :
                         mode == TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS ? cgraph->nodes[गाँठ_idx + 3]->src[1] :
                                                                   logits;
    ggml_tensor * weights = cgraph->nodes[गाँठ_idx + ctx->अंक_additional_fused_ops];
    ggml_tensor * ids = mode == TOPK_MOE_SIGMOID_NORM_BIAS       ? cgraph->nodes[गाँठ_idx + 4] :
                        mode == TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS ? cgraph->nodes[गाँठ_idx + 5] :
                        mode == TOPK_MOE_LATE_SOFTMAX             ? cgraph->nodes[गाँठ_idx + 1] :
                                                                   cgraph->nodes[गाँठ_idx + 3];

    GGML_ASSERT(logits->type == GGML_TYPE_F32);
    GGML_ASSERT(bias->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    स्थिर पूर्णांक n_experts = logits->ne[0];
    स्थिर पूर्णांक n_rows    = logits->ne[1];
    स्थिर पूर्णांक n_expert_used = weights->ne[1];

    GGML_ASSERT(ids->nb[1] / ggml_type_आकार(ids->type) == (माप_प्रकार) n_experts);

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, nullptr, nullptr, nullptr, cgraph->nodes[गाँठ_idx], GGML_OP_SOFT_MAX);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer logits_buf = ggml_vk_tensor_subbuffer(ctx, logits);
    vk_subbuffer bias_buf = ggml_vk_tensor_subbuffer(ctx, bias);
    vk_subbuffer weights_buf = ggml_vk_tensor_subbuffer(ctx, weights);
    vk_subbuffer ids_buf = ggml_vk_tensor_subbuffer(ctx, ids);

    vk_op_topk_moe_धकेलो_constants pc {};
    pc.n_rows = n_rows;
    pc.n_experts_धकेलो = n_experts;
    pc.n_expert_used = n_expert_used;
    pc.clamp_min = -std::numeric_limits<भग्न>::infinity();
    pc.clamp_max = std::numeric_limits<भग्न>::infinity();
    अगर (mode == TOPK_MOE_EARLY_SOFTMAX_NORM) {
        ggml_tensor * clamp = cgraph->nodes[गाँठ_idx + 7];
        GGML_ASSERT(clamp->op == GGML_OP_CLAMP);
        pc.clamp_min = ggml_get_op_params_f32(clamp, 0);
        pc.clamp_max = ggml_get_op_params_f32(clamp, 1);
    }
    अगर (mode == TOPK_MOE_SIGMOID_NORM_BIAS) {
        ggml_tensor * clamp = cgraph->nodes[गाँठ_idx + 8];
        GGML_ASSERT(clamp->op == GGML_OP_CLAMP);
        pc.clamp_min = ggml_get_op_params_f32(clamp, 0);
        pc.clamp_max = ggml_get_op_params_f32(clamp, 1);
    }
    अगर (mode == TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS) {
        ggml_tensor * clamp = cgraph->nodes[गाँठ_idx + 9];
        GGML_ASSERT(clamp->op == GGML_OP_CLAMP);
        pc.clamp_min = ggml_get_op_params_f32(clamp, 0);
        pc.clamp_max = ggml_get_op_params_f32(clamp, 1);
    }

#घोषणा GATING_FUNC_SOFTMAX 0
#घोषणा GATING_FUNC_SIGMOID 1
#घोषणा GATING_FUNC_SOFTMAX_WEIGHT 2
#घोषणा GATING_FUNC_SQRT_SOFTPLUS 3

    pc.gating_func = mode == TOPK_MOE_SIGMOID_NORM_BIAS       ? GATING_FUNC_SIGMOID :
                     mode == TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS ? GATING_FUNC_SQRT_SOFTPLUS :
                     mode == TOPK_MOE_LATE_SOFTMAX             ? GATING_FUNC_SOFTMAX_WEIGHT :
                                                                 GATING_FUNC_SOFTMAX;
    pc.has_bias = has_bias;
    pc.with_norm = mode == TOPK_MOE_EARLY_SOFTMAX_NORM || has_bias;
    अगर (ctx->fused_topk_moe_scale) {
        GGML_ASSERT(weights->op == GGML_OP_SCALE);
        pc.output_scale = ggml_get_op_params_f32(weights, 0);
        pc.output_bias = ggml_get_op_params_f32(weights, 1);
    } अन्यथा {
        pc.output_scale = 1.0f;
        pc.output_bias = 0.0f;
    }

    GGML_ASSERT(n_expert_used <= n_experts);

    स्थिर uint32_t rows_per_block = 4;
    std::array<uint32_t, 3> elements = { CEIL_DIV(n_rows, rows_per_block), 1, 1 };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, {logits_buf, bias_buf, weights_buf, ids_buf}, pc, elements);
}

व्योम ggml_vk_rope(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_cgraph * cgraph, पूर्णांक गाँठ_idx, bool backprop) {
    ggml_tensor * dst = cgraph->nodes[गाँठ_idx];
    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];
    स्थिर ggml_tensor * src2 = dst->src[2];
    स्थिर ggml_tensor * src3 = nullptr;
    स्थिर पूर्णांक n_dims        = ((int32_t *) dst->op_params)[1];
    स्थिर पूर्णांक mode          = ((int32_t *) dst->op_params)[2];
    // const int n_ctx         = ((int32_t *) dst->op_params)[3];
    स्थिर पूर्णांक n_ctx_orig    = ((int32_t *) dst->op_params)[4];
    स्थिर भग्न freq_base   = ((भग्न *)   dst->op_params)[5];
    स्थिर भग्न beta_fast   = ((भग्न *)   dst->op_params)[9];
    स्थिर भग्न beta_slow   = ((भग्न *)   dst->op_params)[10];
    पूर्णांक sections[4] {};
    अगर (mode & GGML_ROPE_TYPE_MROPE) {
        स_नकल(sections, (int32_t *) dst->op_params + 11, माप(पूर्णांक)*4);
    }

    भग्न corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    uint32_t set_rows_stride = 0;
    // Fused rope + view + set_rows passes the set_rows destination stride in set_rows_stride
    // and overrides the dst and sets src3=row_indices
    अगर (ctx->अंक_additional_fused_ops > 0) {
        set_rows_stride = cgraph->nodes[गाँठ_idx + 2]->nb[1] / ggml_type_आकार(cgraph->nodes[गाँठ_idx + 2]->type);
        src3 = cgraph->nodes[गाँठ_idx + 2]->src[1];
        dst = cgraph->nodes[गाँठ_idx + 2];
    }

    ggml_vk_op_f32<vk_op_rope_धकेलो_constants>(ctx, subctx, src0, src1, src2, src3, dst, GGML_OP_ROPE,
        ggml_vk_make_rope_constants(cgraph->nodes[गाँठ_idx], src0, src2 != nullptr, backprop, set_rows_stride));
}

व्योम ggml_vk_argsort(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, ggml_tensor * dst) {
    स्थिर uint32_t * op_params = (स्थिर uint32_t *)dst->op_params;

    uint32_t ncols = src0->ne[0];
    uint32_t nrows = ggml_nrows(src0);

    uint32_t ncols_pad_log2 = (uint32_t)ceilf(log2f(भग्न(ncols)));
    uint32_t ncolsp2 = 1 << ncols_pad_log2;

    vk_op_argsort_धकेलो_constants pc { ncols, ncolsp2, ncols_pad_log2, nrows, op_params[0], 0, 0, 0, 0, };

    // Pick the largest workgroup size <= ncolsp2
    uint32_t pipeline_idx = std::min(ncols_pad_log2, अंक_argsort_pipelines - 1);

    uint32_t max_wg_log2 = std::min(ctx->device->max_workgroup_आकार_log2, अंक_argsort_pipelines - 1);

    // Use the "small" argsort shader if the whole sort can be done by a single workgroup.
    bool use_small = ncols_pad_log2 <= max_wg_log2 &&
                     ctx->device->pipeline_argsort_f32[pipeline_idx] != nullptr;

    vk_pipeline pipeline = use_small ? ctx->device->pipeline_argsort_f32[pipeline_idx]
                                     : ctx->device->pipeline_argsort_large_f32[pipeline_idx];

    vk_subbuffer src0_buf = ggml_vk_tensor_subbuffer(ctx, src0);
    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
    vk_subbuffer subbuf1 = dst_buf;

    // Reserve space for ivec2 per element, with rows padded to a power of two
    अगर (!use_small) {
        स्थिर माप_प्रकार x_sz = माप_प्रकार{ncolsp2} * nrows * 2 * माप(पूर्णांक);

        अगर (ctx->prealloc_आकार_x < x_sz) {
            ctx->prealloc_आकार_x = x_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        अगर (ctx->prealloc_x_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }
        subbuf1 = { ctx->prealloc_x, 0, ctx->prealloc_x->size };
    }

    std::array<uint32_t, 3> elements;

    elements[0] = ncolsp2;
    elements[1] = std::min((uint32_t)ggml_nrows(src0), ctx->device->properties.limits.max_Compute_Work_Group_गिनती[1]);
    elements[2] = 1;

    // First dispatch initializes tmp_idx and does the first N passes where
    // there is only communication between threads in the same workgroup.
    {
        vk_op_argsort_धकेलो_constants pc2 = pc;
        pc2.outer_start = 0;
        pc2.outer_end = std::min(ncols_pad_log2, max_wg_log2);
        pc2.inner_start = 0;
        pc2.inner_end = 100;
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, subbuf1, dst_buf }, pc2, elements);
    }
    अगर (!use_small) {
        ggml_vk_sync_buffers(ctx, subctx);
        // Loop over outer/inner passes, synchronizing between each pass.
        क्रम (uint32_t outer = max_wg_log2; outer < ncols_pad_log2; ++outer) {
            क्रम (uint32_t inner = 0; inner < outer + 1; ++inner) {
                vk_op_argsort_धकेलो_constants pc2 = pc;
                pc2.outer_start = outer;
                pc2.outer_end = outer + 1;
                pc2.inner_start = inner;
                pc2.inner_end = inner + 1;
                // When the inner idx is large enough, there's only communication
                // within a workgroup. So the remaining inner iterations can all
                // run in the same dispatch.
                अगर (outer - inner < pipeline_idx) {
                    pc2.inner_end = 100;
                    inner = outer;
                    pipeline = ctx->device->pipeline_argsort_large_f32[pipeline_idx];
                } अन्यथा {
                    // Smaller workgroup empirically seems to perform better
                    pipeline = ctx->device->pipeline_argsort_large_f32[pipeline_idx - 2];
                }
                ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
                ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, subbuf1, dst_buf }, pc2, elements);
                ggml_vk_sync_buffers(ctx, subctx);
            }
        }
        ctx->prealloc_x_need_sync = true;
    }
}

व्योम ggml_vk_topk(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, ggml_tensor * dst) {
    uint32_t ncols = src0->ne[0];
    uint32_t nrows = ggml_nrows(src0);
    uint32_t k = dst->ne[0];

    // tournament path is faster where it fits; use radix-select only past its k limit
    स्थिर uint32_t k_min_pipeline = std::max((uint32_t) log2f(भग्न(k)) + 1, ctx->device->subgroup_आकार_log2);
    अगर (k_min_pipeline >= अंक_topk_pipelines || ctx->device->pipeline_topk_f32[k_min_pipeline] == nullptr) {
        vk_pipeline pipeline = ctx->device->pipeline_topk_radix_f32;
        GGML_ASSERT(pipeline != nullptr);

        अगर (ctx->prealloc_x_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }

        vk_op_topk_radix_धकेलो_constants pc { ncols, k, nrows, 0, 0, 0 };
        std::array<uint32_t, 3> elements {
            pipeline->wg_denoms[0],
            std::min(nrows, ctx->device->properties.limits.max_Compute_Work_Group_गिनती[1]),
            1,
        };
        // the non-QSA path only uses bindings 0/1; bind valid buffers for the unused QSA slots
        vk_subbuffer src0_buf = ggml_vk_tensor_subbuffer(ctx, src0);
        vk_subbuffer dst_buf  = ggml_vk_tensor_subbuffer(ctx, dst);
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
            { src0_buf, dst_buf, src0_buf, src0_buf, src0_buf }, pc, elements);
        वापस;
    }

    vk_op_topk_धकेलो_constants pc { ncols, ncols, ncols, k, nrows, 0, 0 };

    अगर (ctx->prealloc_x_need_sync) {
        ggml_vk_sync_buffers(ctx, subctx);
    }

    std::array<uint32_t, 3> elements;
    elements[1] = std::min(nrows, ctx->device->properties.limits.max_Compute_Work_Group_गिनती[1]);
    elements[2] = 1;

    uint32_t अंक_elements = ncols;

    // Each iteration reduces a workgroup's worth of elements down to the K
    // largest elements. Repeat until we have the top K elements.
    // Need to do at least one iteration to write out the results.
    bool done_one_iter = false;
    uint32_t dbl_buf_सूचकांक = 0;
    माप_प्रकार dbl_buf_आकार;
    जबतक (अंक_elements > k || !done_one_iter) {

        // Prefer going as small as num_topk_pipelines - 3 for perf reasons.
        // But if K is larger, then we need a larger workgroup
        uint32_t max_pipeline = अंक_topk_pipelines - 1;
        uint32_t preferred_pipeline = std::max(अंक_topk_pipelines - 3, (uint32_t)log2f(भग्न(k)) + 2);
        max_pipeline = std::min(preferred_pipeline, max_pipeline);
        uint32_t min_pipeline = (uint32_t)log2f(भग्न(k)) + 1;
        // require full subgroup
        min_pipeline = std::max(min_pipeline, ctx->device->subgroup_आकार_log2);

        uint32_t pipeline_idx = (uint32_t)ceilf(log2f(भग्न(अंक_elements)));
        pipeline_idx = std::min(pipeline_idx, max_pipeline);
        pipeline_idx = std::max(pipeline_idx, min_pipeline);

        अगर (अंक_elements > (1u << pipeline_idx)) {
            // If we could finish on this loop iteration (i.e. a single workgroup)
            // then do so. It's better than the overhead of another pass.
            क्रम (uint32_t i = pipeline_idx; i < अंक_topk_pipelines; ++i) {
                अगर (अंक_elements <= (1u << i)) {
                    pipeline_idx = i;
                    अवरोध;
                }
            }
        }

        vk_pipeline pipeline = ctx->device->pipeline_topk_f32[pipeline_idx];
        // If the device doesn't support a pipeline this large, use smaller
        जबतक (!pipeline) {
            pipeline_idx--;
            GGML_ASSERT(pipeline_idx >= min_pipeline);
            pipeline = ctx->device->pipeline_topk_f32[pipeline_idx];
        }

        vk_op_topk_धकेलो_constants pc2 = pc;
        pc2.ncols_निवेश = अंक_elements;

        // Number of elements remaining after this pass
        uint32_t अंक_dst_elements = (अंक_elements / pipeline->wg_denoms[0]) * k + std::min(k, अंक_elements % pipeline->wg_denoms[0]);

        pc2.ncols_output = अंक_dst_elements;

        अगर (!done_one_iter) {
            // Reserve space for ivec2 per element, double buffered
            // K per workgroup per row
            dbl_buf_आकार = अंक_dst_elements * nrows * 2 * माप(पूर्णांक);
            dbl_buf_आकार = ROUNDUP_POW2(dbl_buf_आकार, ctx->device->properties.limits.min_Storage_बफर_Offset_Alignment);
            स्थिर माप_प्रकार x_sz = dbl_buf_आकार * 2;

            अगर (ctx->prealloc_आकार_x < x_sz) {
                ctx->prealloc_आकार_x = x_sz;
                ggml_vk_preallocate_buffers(ctx, subctx);
            }
        }

        vk_subbuffer src_buf;
        vk_subbuffer dst_buf;

        अगर (अंक_elements == ncols) {
            pc2.first_pass = 1;
            src_buf = ggml_vk_tensor_subbuffer(ctx, src0);
        } अन्यथा {
            src_buf = { ctx->prealloc_x, dbl_buf_सूचकांक * dbl_buf_आकार, dbl_buf_आकार };
        }
        अगर (अंक_dst_elements == k) {
            pc2.last_pass = 1;
            dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
        } अन्यथा {
            dst_buf = { ctx->prealloc_x, (dbl_buf_सूचकांक ^ 1) * dbl_buf_आकार, dbl_buf_आकार };
        }

        elements[0] = अंक_elements;

        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src_buf, dst_buf }, pc2, elements);
        अंक_elements = अंक_dst_elements;
        dbl_buf_सूचकांक ^= 1;
        अगर (अंक_elements > k) {
            ggml_vk_sync_buffers(ctx, subctx);
        }
        done_one_iter = true;
    }
    ctx->prealloc_x_need_sync = true;
}

व्योम ggml_vk_topk_qsa(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_cgraph * cgraph, पूर्णांक गाँठ_idx) {
    स्थिर ggml_tensor * get_rows = cgraph->nodes[गाँठ_idx + 0];
    स्थिर ggml_tensor * add      = cgraph->nodes[गाँठ_idx + ctx->अंक_additional_fused_ops - 1];
    ggml_tensor *       शीर्ष_k    = cgraph->nodes[गाँठ_idx + ctx->अंक_additional_fused_ops];

    स्थिर ggml_tensor * scores   = get_rows->src[0]; // [n_tps, n_blocks, n_stream]
    स्थिर ggml_tensor * cell_blk = get_rows->src[1]; // [n_kv, n_stream]

    // raw f16 mask: follow the reshape/cpy chain back to the materialized input
    स्थिर ggml_tensor * mask = add->src[1];
    जबतक (mask->op == GGML_OP_RESHAPE || mask->op == GGML_OP_CPY) {
        mask = mask->src[0];
    }

    स्थिर uint32_t n_tps    = scores->ne[0];
    स्थिर uint32_t n_blocks = scores->ne[1];
    स्थिर uint32_t n_stream = scores->ne[2];
    स्थिर uint32_t n_kv     = cell_blk->ne[0];
    स्थिर uint32_t width    = शीर्ष_k->ne[0];
    स्थिर uint32_t nrows    = n_tps * n_stream;

    vk_pipeline pipeline = ctx->device->pipeline_topk_radix_qsa;
    GGML_ASSERT(pipeline != nullptr);

    // scratch holds the gathered+masked input, materialized once and reused across passes
    स्थिर माप_प्रकार scratch_आकार = माप_प्रकार{ n_kv } * nrows * माप(भग्न);
    अगर (ctx->prealloc_आकार_x < scratch_आकार) {
        ctx->prealloc_आकार_x = scratch_आकार;
        ggml_vk_preallocate_buffers(ctx, subctx);
    }
    अगर (ctx->prealloc_x_need_sync) {
        ggml_vk_sync_buffers(ctx, subctx);
    }

    vk_op_topk_radix_धकेलो_constants pc { n_kv, width, nrows, n_tps, n_blocks, n_stream };
    std::array<uint32_t, 3> elements {
        pipeline->wg_denoms[0],
        std::min(nrows, ctx->device->properties.limits.max_Compute_Work_Group_गिनती[1]),
        1,
    };
    vk_subbuffer scratch_buf { ctx->prealloc_x, 0, ctx->prealloc_x->size };
    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        { ggml_vk_tensor_subbuffer(ctx, scores), ggml_vk_tensor_subbuffer(ctx, शीर्ष_k),
          ggml_vk_tensor_subbuffer(ctx, cell_blk), ggml_vk_tensor_subbuffer(ctx, mask),
          scratch_buf }, pc, elements);
    ctx->prealloc_x_need_sync = true;
}

व्योम ggml_vk_योग(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_योग_rows_धकेलो_constants p = vk_op_योग_rows_धकेलो_constants_init(src0, dst, ggml_nelements(src0));
    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_योग, p);
}

व्योम ggml_vk_योग_rows(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_योग_rows_धकेलो_constants p = vk_op_योग_rows_धकेलो_constants_init(src0, dst, src0->ne[0]);
    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_योग_ROWS, p);
}

व्योम ggml_vk_mean(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_योग_rows_धकेलो_constants p = vk_op_योग_rows_धकेलो_constants_init(src0, dst, src0->ne[0]);
    p.weight = 1.0f / (भग्न)src0->ne[0];
    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_MEAN, p);
}

व्योम ggml_vk_cumsum(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_योग_rows_धकेलो_constants pc = vk_op_योग_rows_धकेलो_constants_init(src0, dst, src0->ne[0]);
    // Use the single pass shader when the rows are small or there are enough rows to fill the GPU.
    // For fewer, larger rows, use the multipass shader to spread each row across SMs.
    अगर (dst->ne[0] <= 4096 || ggml_nrows(dst) >= ctx->device->shader_core_गिनती) {
        ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_CUMSUM, pc);
        वापस;
    }

    // First pass computes partial sums within a block, and stores the last partial
    // to the temp buffer. Second pass sums the block partials from the temp buffer
    // and adds that to the result of the first pass.
    vk_pipeline pipeline1 = ctx->device->pipeline_cumsum_multipass1_f32;
    vk_pipeline pipeline2 = ctx->device->pipeline_cumsum_multipass2_f32;
    GGML_ASSERT(pipeline1 != nullptr && pipeline2 != nullptr);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline1, 1);
    ggml_pipeline_request_descriptor_sets(ctx, pipeline2, 1);

    std::array<uint32_t, 3> elements;

    elements[0] = dst->ne[0];
    elements[1] = (uint32_t)ggml_nrows(dst);
    elements[2] = 1;

    माप_प्रकार अस्थायी_आकार = माप(भग्न) * elements[0] * ggml_nrows(dst);

    अगर (ctx->prealloc_आकार_split_k < अस्थायी_आकार) {
        ctx->prealloc_आकार_split_k = अस्थायी_आकार;
        ggml_vk_preallocate_buffers(ctx, subctx);
    }

    vk_subbuffer src_buf = ggml_vk_tensor_subbuffer(ctx, src0);
    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
    vk_subbuffer अस्थायी_buf = ggml_vk_subbuffer(ctx, ctx->prealloc_split_k, 0);

    अगर (ctx->prealloc_split_k_need_sync) {
        ggml_vk_sync_buffers(ctx, subctx);
    }

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline1, {src_buf, dst_buf, अस्थायी_buf}, pc, elements);
    ggml_vk_sync_buffers(ctx, subctx);
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline2, {src_buf, dst_buf, अस्थायी_buf}, pc, elements);

    ctx->prealloc_split_k_need_sync = true;
}

अटल std::array<uint32_t, 3> ggml_vk_nrows_elements(uint32_t nr) {
    अगर (nr > 262144) {
        वापस { 512, 512, CEIL_DIV(nr, 262144) };
    }
    अगर (nr > 512) {
        वापस { 512, CEIL_DIV(nr, 512), 1 };
    }
    वापस { nr, 1, 1 };
}

व्योम ggml_vk_cross_entropy_loss(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_scalar(dst));

    स्थिर uint32_t nclasses = (uint32_t)src0->ne[0];
    स्थिर uint32_t nrows    = (uint32_t)ggml_nrows(src0);

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, src0, src1, nullptr, dst, GGML_OP_CROSS_ENTROPY_LOSS);
    GGML_ASSERT(pipeline != nullptr);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    ggml_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_योग_rows_f32, 1);

    vk_subbuffer src0_buf = ggml_vk_tensor_subbuffer(ctx, src0);
    vk_subbuffer src1_buf = ggml_vk_tensor_subbuffer(ctx, src1);
    vk_subbuffer dst_buf  = ggml_vk_tensor_subbuffer(ctx, dst, true);

    स्थिर vk_op_धकेलो_constants pc = { nclasses, nrows, 0.0f, 0.0f, 0.0f, 0.0f };

    स्थिर माप_प्रकार tmp_आकार = (माप_प्रकार)nrows * माप(भग्न);
    अगर (ctx->prealloc_आकार_x < tmp_आकार) {
        ctx->prealloc_आकार_x = tmp_आकार;
        ggml_vk_preallocate_buffers(ctx, subctx);
    }
    अगर (ctx->prealloc_x_need_sync) {
        ggml_vk_sync_buffers(ctx, subctx);
    }

    vk_subbuffer tmp_buf = { ctx->prealloc_x, 0, tmp_आकार };
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, tmp_buf }, pc, ggml_vk_nrows_elements(nrows));
    ggml_vk_sync_buffers(ctx, subctx);

    vk_op_योग_rows_धकेलो_constants sp = {};
    sp.n_cols = nrows;
    sp.ne01 = 1;
    sp.ne02 = 1;
    sp.weight = 1.0f;
    init_pushconst_fastdiv(sp);
    sp.misalign_offsets = get_misalign_bytes(ctx, dst) / ggml_type_आकार(dst->type);

    ggml_vk_dispatch_pipeline(ctx, subctx, ctx->device->pipeline_योग_rows_f32, { tmp_buf, dst_buf }, sp, { 1, 1, 1 });
    ctx->prealloc_x_need_sync = true;
}

व्योम ggml_vk_cross_entropy_loss_back(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    स्थिर ggml_tensor * grad   = dst->src[0];
    स्थिर ggml_tensor * logits = dst->src[1];
    स्थिर ggml_tensor * labels = dst->src[2];

    GGML_ASSERT(grad->type   == GGML_TYPE_F32);
    GGML_ASSERT(logits->type == GGML_TYPE_F32);
    GGML_ASSERT(labels->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type    == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_scalar(grad));
    GGML_ASSERT(ggml_is_contiguous(grad));
    GGML_ASSERT(ggml_is_contiguous(logits));
    GGML_ASSERT(ggml_is_contiguous(labels));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(logits, labels));
    GGML_ASSERT(ggml_are_same_shape(logits, dst));

    स्थिर uint32_t nclasses = (uint32_t)logits->ne[0];
    स्थिर uint32_t nrows    = (uint32_t)ggml_nrows(logits);

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, grad, logits, labels, dst, GGML_OP_CROSS_ENTROPY_LOSS_BACK);
    GGML_ASSERT(pipeline != nullptr);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer grad_buf   = ggml_vk_tensor_subbuffer(ctx, grad);
    vk_subbuffer logits_buf = ggml_vk_tensor_subbuffer(ctx, logits);
    vk_subbuffer labels_buf = ggml_vk_tensor_subbuffer(ctx, labels);
    vk_subbuffer dst_buf    = ggml_vk_tensor_subbuffer(ctx, dst);

    स्थिर vk_op_धकेलो_constants pc = { nclasses, nrows, 0.0f, 0.0f, 0.0f, 0.0f };
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { grad_buf, logits_buf, labels_buf, dst_buf }, pc, ggml_vk_nrows_elements(nrows));
}

व्योम ggml_vk_argmax(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, ggml_tensor * dst) {
    ggml_vk_op_f32<vk_op_धकेलो_constants>(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_ARGMAX, { (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], 0.0f, 0.0f, 0.0f, 0.0f });
}

व्योम ggml_vk_गिनती_equal(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, स्थिर ggml_tensor * src1, ggml_tensor * dst) {
    ggml_vk_op_f32<vk_op_धकेलो_constants>(ctx, subctx, src0, src1, nullptr, nullptr, dst, GGML_OP_गिनती_EQUAL, { (uint32_t)ggml_nelements(src0), 0, 0.0f, 0.0f, 0.0f, 0.0f });
}

व्योम ggml_vk_solve_tri(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, स्थिर ggml_tensor * src1, ggml_tensor * dst) {
    स्थिर uint32_t src0_type_आकार = ggml_type_आकार(src0->type);
    स्थिर uint32_t src1_type_आकार = ggml_type_आकार(src1->type);
    स्थिर uint32_t dst_type_आकार = ggml_type_आकार(dst->type);

    ggml_vk_op_f32<vk_op_द्विआधारी_धकेलो_constants>(ctx, subctx, src0, src1, nullptr, nullptr, dst, GGML_OP_SOLVE_TRI, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_आकार, (uint32_t)src0->nb[1] / src0_type_आकार, (uint32_t)src0->nb[2] / src0_type_आकार, (uint32_t)src0->nb[3] / src0_type_आकार,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_आकार, (uint32_t)src1->nb[1] / src1_type_आकार, (uint32_t)src1->nb[2] / src1_type_आकार, (uint32_t)src1->nb[3] / src1_type_आकार,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_आकार, (uint32_t) dst->nb[1] /  dst_type_आकार, (uint32_t) dst->nb[2] /  dst_type_आकार, (uint32_t) dst->nb[3] /  dst_type_आकार,
        0,
        0.0f, 0.0f, 0,
    });
}

व्योम ggml_vk_im2col(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, स्थिर ggml_tensor * src1, ggml_tensor * dst) {
    स्थिर int32_t s0 = dst->op_params[0];
    स्थिर int32_t s1 = dst->op_params[1];
    स्थिर int32_t p0 = dst->op_params[2];
    स्थिर int32_t p1 = dst->op_params[3];
    स्थिर int32_t d0 = dst->op_params[4];
    स्थिर int32_t d1 = dst->op_params[5];

    स्थिर bool is_2D = dst->op_params[6] == 1;

    स्थिर uint32_t IC = src1->ne[is_2D ? 2 : 1];
    स्थिर uint32_t IH = is_2D ? src1->ne[1] : 1;
    स्थिर uint32_t IW =         src1->ne[0];

    स्थिर uint32_t KH = is_2D ? src0->ne[1] : 1;
    स्थिर uint32_t KW =         src0->ne[0];

    स्थिर uint32_t OH = is_2D ? dst->ne[2] : 1;
    स्थिर uint32_t OW =         dst->ne[1];

    स्थिर uint32_t offset_delta = src1->nb[is_2D ? 2 : 1] / 4; // nb is byte offset, src is type float32
    स्थिर uint32_t batch_offset = src1->nb[is_2D ? 3 : 2] / 4; // nb is byte offset, src is type float32

    स्थिर uint32_t batch = src1->ne[is_2D ? 3 : 2];

    स्थिर ggml_backend_vk_बफर_context * d_buf_ctx = (ggml_backend_vk_बफर_context *)dst->बफर->context;
    स्थिर vk_बफर d_buf = d_buf_ctx->dev_बफर;

    स्थिर vk::DeviceAddress dst_addr = d_buf->bda_addr + vk_tensor_offset(dst) + dst->view_offs;

    ggml_vk_op_f32<vk_op_im2col_धकेलो_constants>(ctx, subctx, src0, src1, nullptr, nullptr, dst, GGML_OP_IM2COL, {
        dst_addr,
        batch_offset, offset_delta,
        IC, IW, IH, OW, OH, KW, KH,
        OH * batch,
        IC * KH * KW,
        s0, s1, p0, p1, d0, d1, batch * IC
    });
}

व्योम ggml_vk_im2col_3d(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, स्थिर ggml_tensor * src1, ggml_tensor * dst) {
    GGML_TENSOR_द्विआधारी_OP_LOCALS

    स्थिर int32_t s0 = ((स्थिर int32_t *)(dst->op_params))[0];
    स्थिर int32_t s1 = ((स्थिर int32_t *)(dst->op_params))[1];
    स्थिर int32_t s2 = ((स्थिर int32_t *)(dst->op_params))[2];
    स्थिर int32_t p0 = ((स्थिर int32_t *)(dst->op_params))[3];
    स्थिर int32_t p1 = ((स्थिर int32_t *)(dst->op_params))[4];
    स्थिर int32_t p2 = ((स्थिर int32_t *)(dst->op_params))[5];
    स्थिर int32_t d0 = ((स्थिर int32_t *)(dst->op_params))[6];
    स्थिर int32_t d1 = ((स्थिर int32_t *)(dst->op_params))[7];
    स्थिर int32_t d2 = ((स्थिर int32_t *)(dst->op_params))[8];
    स्थिर int32_t IC = ((स्थिर int32_t *)(dst->op_params))[9];

    स्थिर int64_t N  = ne13 / IC;
    स्थिर int64_t ID = ne12;
    स्थिर int64_t IH = ne11;
    स्थिर int64_t IW = ne10;

    स्थिर int64_t KD = ne02;
    स्थिर int64_t KH = ne01;
    स्थिर int64_t KW = ne00;

    स्थिर int64_t OD = ne3 / N;
    स्थिर int64_t OH = ne2;
    स्थिर int64_t OW = ne1;

    स्थिर ggml_backend_vk_बफर_context * d_buf_ctx = (ggml_backend_vk_बफर_context *)dst->बफर->context;
    स्थिर vk_बफर d_buf = d_buf_ctx->dev_बफर;

    स्थिर vk::DeviceAddress dst_addr = d_buf->bda_addr + vk_tensor_offset(dst) + dst->view_offs;

    vk_op_im2col_3d_धकेलो_constants pc {};

    pc.dst_addr = dst_addr;
    pc.nb10 = nb10 / ggml_type_आकार(src1->type);
    pc.nb11 = nb11 / ggml_type_आकार(src1->type);
    pc.nb12 = nb12 / ggml_type_आकार(src1->type);
    pc.nb13 = nb13 / ggml_type_आकार(src1->type);
    pc.s0 = s0;
    pc.s1 = s1;
    pc.s2 = s2;
    pc.p0 = p0;
    pc.p1 = p1;
    pc.p2 = p2;
    pc.d0 = d0;
    pc.d1 = d1;
    pc.d2 = d2;
    pc.IW = IW;
    pc.IH = IH;
    pc.ID = ID;
    pc.IC = IC;
    pc.KW = KW;
    pc.OH = OH;
    pc.KD_KH_KW = KD*KH*KW;
    pc.KH_KW = KH*KW;
    pc.IC_KD_KH_KW = IC*KD*KH*KW;
    pc.N_OD_OH = N*OD*OH;
    pc.OD_OH = OD*OH;
    pc.OD_OH_OW_IC_KD_KH_KW = OD*OH*OW*IC*KD*KH*KW;
    pc.OH_OW_IC_KD_KH_KW = OH*OW*IC*KD*KH*KW;
    pc.OW_IC_KD_KH_KW = OW*IC*KD*KH*KW;

    ggml_vk_op_f32<vk_op_im2col_3d_धकेलो_constants>(ctx, subctx, src0, src1, nullptr, nullptr, dst, GGML_OP_IM2COL_3D, std::move(pc));
}

व्योम ggml_vk_timestep_embedding(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, ggml_tensor * dst) {
    स्थिर uint32_t dim = dst->op_params[0];
    स्थिर uint32_t max_period = dst->op_params[1];
    स्थिर uint32_t nb1 = dst->nb[1] / ggml_type_आकार(dst->type);

    ggml_vk_op_f32<vk_op_timestep_embedding_धकेलो_constants>(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_TIMESTEP_EMBEDDING, {
        nb1, dim, max_period,
    });
}

व्योम ggml_vk_conv_transpose_1d(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, स्थिर ggml_tensor * src1, ggml_tensor * dst) {
    // src0: (K, Cout, Cin, 1) -- kernel
    // src1: (L, Cin, 1, 1) -- input
    // dst: (*, Cout, 1, 1)

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_द्विआधारी_OP_LOCALS

    GGML_ASSERT(nb00 == माप(भग्न));
    GGML_ASSERT(nb10 == माप(भग्न));

    स्थिर int32_t s0 = dst->op_params[0];

    vk_op_conv_transpose_1d_धकेलो_constants p{};
    p.Cout = static_cast<uint32_t>(ne01);
    p.Cin = static_cast<uint32_t>(ne02);
    p.K = static_cast<uint32_t>(ne00);
    p.L = static_cast<uint32_t>(ne10);
    p.KL = static_cast<uint32_t>(ne0);
    p.nb01 = static_cast<uint32_t>(nb01 / nb00);
    p.nb02 = static_cast<uint32_t>(nb02 / nb00);
    p.nb11 = static_cast<uint32_t>(nb11 / nb10);
    p.nb1 = static_cast<uint32_t>(nb1 / nb0);
    p.s0 = static_cast<uint32_t>(s0);

    ggml_vk_op_f32(ctx, subctx, src0, src1, nullptr, nullptr, dst, GGML_OP_CONV_TRANSPOSE_1D, std::move(p));
}

व्योम ggml_vk_col2im_1d(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, ggml_tensor * dst) {
    // src0: [K_OC, T_in] columns from matmul
    // dst:  [T_out, OC]

    स्थिर int32_t stride = dst->op_params[0];
    स्थिर int32_t oc     = dst->op_params[1];
    स्थिर int32_t p0     = dst->op_params[2];

    स्थिर uint32_t K_OC  = static_cast<uint32_t>(src0->ne[0]);
    स्थिर uint32_t T_in  = static_cast<uint32_t>(src0->ne[1]);
    स्थिर uint32_t T_out = static_cast<uint32_t>(dst->ne[0]);
    स्थिर uint32_t OC    = static_cast<uint32_t>(oc);
    स्थिर uint32_t K     = K_OC / OC;

    vk_op_col2im_1d_धकेलो_constants p{};
    p.T_out  = T_out;
    p.OC     = OC;
    p.K_OC   = K_OC;
    p.T_in   = T_in;
    p.K      = K;
    p.stride = stride;
    p.p0     = p0;

    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_COL2IM_1D, std::move(p));
}

व्योम ggml_vk_snake_dispatch_fused(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_cgraph * cgraph, पूर्णांक गाँठ_idx) {
    स्थिर ggml_tensor * mul0 = cgraph->nodes[गाँठ_idx + 0];
    स्थिर ggml_tensor * sqr  = cgraph->nodes[गाँठ_idx + 2];
    स्थिर ggml_tensor * mul1 = cgraph->nodes[गाँठ_idx + 3];
    ggml_tensor *       add  = cgraph->nodes[गाँठ_idx + 4];

    // x carries the full activation shape, a is the broadcast operand
    स्थिर ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
    स्थिर ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];

    // mul1 reads sqr and inv_b in either operand order
    स्थिर ggml_tensor * inv_b = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];

    vk_pipeline pipeline = nullptr;
    चयन (x->type) {
        हाल GGML_TYPE_F32:  pipeline = ctx->device->pipeline_snake_f32;  अवरोध;
        हाल GGML_TYPE_F16:  pipeline = ctx->device->pipeline_snake_f16;  अवरोध;
        हाल GGML_TYPE_BF16: pipeline = ctx->device->pipeline_snake_bf16; अवरोध;
        शेष:             GGML_ABORT("unsupported type");
    }
    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer x_buf     = ggml_vk_tensor_subbuffer(ctx, x);
    vk_subbuffer a_buf     = ggml_vk_tensor_subbuffer(ctx, a);
    vk_subbuffer inv_b_buf = ggml_vk_tensor_subbuffer(ctx, inv_b);
    vk_subbuffer dst_buf   = ggml_vk_tensor_subbuffer(ctx, add);

    vk_op_snake_धकेलो_constants pc{};
    pc.ne0 = static_cast<uint32_t>(x->ne[0]);
    pc.ne1 = static_cast<uint32_t>(x->ne[1]);

    std::array<uint32_t, 3> elements = { pc.ne0, pc.ne1, 1 };
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { x_buf, a_buf, inv_b_buf, dst_buf }, pc, elements);
}

व्योम ggml_vk_pool_1d(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, ggml_tensor * dst) {
    uint32_t op = static_cast<uint32_t>(dst->op_params[0]);
    स्थिर int32_t k0 = dst->op_params[1];
    स्थिर int32_t s0 = dst->op_params[2];
    स्थिर int32_t p0 = dst->op_params[3];

    स्थिर uint32_t IL = src0->ne[0];

    स्थिर uint32_t N = dst->ne[3] * dst->ne[2];

    स्थिर uint32_t OC = dst->ne[1];
    स्थिर uint32_t OL = dst->ne[0];

    स्थिर uint32_t parallel_elements = N * OC * OL;

    ggml_vk_op_f32<vk_op_pool1d_धकेलो_constants>(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_POOL_1D, {
        IL, OL, OC,
        parallel_elements,
        op,
        k0, s0, p0,
    });
}

व्योम ggml_vk_pool_2d(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, ggml_tensor * dst) {
    uint32_t op = static_cast<uint32_t>(dst->op_params[0]);
    स्थिर int32_t k1 = dst->op_params[1];
    स्थिर int32_t k0 = dst->op_params[2];
    स्थिर int32_t s1 = dst->op_params[3];
    स्थिर int32_t s0 = dst->op_params[4];
    स्थिर int32_t p1 = dst->op_params[5];
    स्थिर int32_t p0 = dst->op_params[6];

    स्थिर uint32_t IH = src0->ne[1];
    स्थिर uint32_t IW = src0->ne[0];

    स्थिर uint32_t N = dst->ne[3];

    स्थिर uint32_t OC = dst->ne[2];
    स्थिर uint32_t OH = dst->ne[1];
    स्थिर uint32_t OW = dst->ne[0];

    स्थिर uint32_t parallel_elements = N * OC * OH * OW;

    ggml_vk_op_f32<vk_op_pool2d_धकेलो_constants>(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_POOL_2D, {
        IW, IH, OW, OH, OC,
        parallel_elements,
        op,
        k0, k1, s0, s1, p0, p1,
    });
}

व्योम ggml_vk_conv_2d(ggml_backend_vk_context * ctx, vk_context & subctx, स्थिर ggml_tensor * src0,
                            स्थिर ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_द्विआधारी_OP_LOCALS
    GGML_ASSERT(nb00 == माप(भग्न) || nb00 == माप(ggml_fp16_t));
    GGML_ASSERT(nb10 == माप(भग्न));
    GGML_ASSERT(nb0 == माप(भग्न));

    bool transpose = dst->op == GGML_OP_CONV_TRANSPOSE_2D;

    vk_op_conv2d_धकेलो_constants p{};
    p.Cout = static_cast<uint32_t>(!transpose ? ne03 : ne02);
    p.Cin  = static_cast<uint32_t>(!transpose ? ne02 : ne03);
    p.N    = static_cast<uint32_t>(ne13);
    GGML_ASSERT(p.Cout == ne2);
    GGML_ASSERT(p.Cin == ne12);

    p.W  = static_cast<uint32_t>(ne10);
    p.H  = static_cast<uint32_t>(ne11);
    p.OW = static_cast<uint32_t>(ne0);
    p.OH = static_cast<uint32_t>(ne1);

    p.nb01 = static_cast<uint32_t>(nb01 / nb00);
    p.nb02 = static_cast<uint32_t>(nb02 / nb00);
    p.nb03 = static_cast<uint32_t>(nb03 / nb00);

    p.nb11 = static_cast<uint32_t>(nb11 / nb10);
    p.nb12 = static_cast<uint32_t>(nb12 / nb10);
    p.nb13 = static_cast<uint32_t>(nb13 / nb10);

    p.nb1 = static_cast<uint32_t>(nb1 / nb0);
    p.nb2 = static_cast<uint32_t>(nb2 / nb0);
    p.nb3 = static_cast<uint32_t>(nb3 / nb0);

    ggml_vk_op_f32(ctx, subctx, src0, src1, nullptr, nullptr, dst, dst->op, std::move(p));
}

व्योम ggml_vk_conv_3d(ggml_backend_vk_context * ctx, vk_context & subctx, स्थिर ggml_tensor * src0,
                            स्थिर ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_द्विआधारी_OP_LOCALS
    GGML_ASSERT(nb00 == माप(भग्न) || nb00 == माप(ggml_fp16_t));
    GGML_ASSERT(nb10 == माप(भग्न));
    GGML_ASSERT(nb0 == माप(भग्न));

    vk_op_conv3d_धकेलो_constants p{};
    p.IC = static_cast<uint32_t>(ggml_get_op_params_i32(dst, 9));
    p.N  = static_cast<uint32_t>(ggml_get_op_params_i32(dst, 10));
    p.OC = static_cast<uint32_t>(ggml_get_op_params_i32(dst, 11));
    GGML_ASSERT(src0->ne[3] == (int64_t)p.IC * p.OC);
    GGML_ASSERT(src1->ne[3] == (int64_t)p.IC * p.N);
    GGML_ASSERT(dst->ne[3] == (int64_t)p.OC * p.N);

    p.IW = static_cast<uint32_t>(ne10);
    p.IH = static_cast<uint32_t>(ne11);
    p.ID = static_cast<uint32_t>(ne12);
    p.OW = static_cast<uint32_t>(ne0);
    p.OH = static_cast<uint32_t>(ne1);
    p.OD = static_cast<uint32_t>(ne2);

    // the shader clamps src addresses to p.IC * p.N * p.IW * p.IH * p.ID - 1 in uint32, so the
    // total input element count must fit in a uint32.
    GGML_ASSERT((uint64_t)p.IC * p.N * p.IW * p.IH * p.ID <= 0xFFFFFFFFull);

    p.nb01 = static_cast<uint32_t>(nb01 / nb00);
    p.nb02 = static_cast<uint32_t>(nb02 / nb00);
    p.nb03 = static_cast<uint32_t>(nb03 / nb00);

    p.nb11 = static_cast<uint32_t>(nb11 / nb10);
    p.nb12 = static_cast<uint32_t>(nb12 / nb10);
    p.nb13 = static_cast<uint32_t>(nb13 / nb10);

    p.nb1 = static_cast<uint32_t>(nb1 / nb0);
    p.nb2 = static_cast<uint32_t>(nb2 / nb0);
    p.nb3 = static_cast<uint32_t>(nb3 / nb0);

    ggml_vk_op_f32(ctx, subctx, src0, src1, nullptr, nullptr, dst, GGML_OP_CONV_3D, std::move(p));
}

व्योम ggml_vk_conv_2d_dw(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, स्थिर ggml_tensor * src1, ggml_tensor * dst) {
    vk_op_conv2d_dw_धकेलो_constants p{};
    p.ne = ggml_nelements(dst);
    p.channels = dst->ne[2];
    p.batches = dst->ne[3];
    p.dst_w = dst->ne[0];
    p.dst_h = dst->ne[1];
    p.src_w = src1->ne[0];
    p.src_h = src1->ne[1];
    p.knl_w = src0->ne[0];
    p.knl_h = src0->ne[1];
    p.stride_x = dst->op_params[0];
    p.stride_y = dst->op_params[1];
    p.pad_x = dst->op_params[2];
    p.pad_y = dst->op_params[3];
    p.dilation_x = dst->op_params[4];
    p.dilation_y = dst->op_params[5];

    GGML_ASSERT(src0->ne[3] == p.channels);
    GGML_ASSERT(src1->ne[3] == p.batches);

    ggml_vk_op_f32(ctx, subctx, src0, src1, nullptr, nullptr, dst, GGML_OP_CONV_2D_DW, std::move(p));
}

व्योम ggml_vk_leaky_relu(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, ggml_tensor * dst) {
    स्थिर भग्न * op_params = (स्थिर भग्न *)dst->op_params;
    vk_op_unary_धकेलो_constants p = vk_op_unary_धकेलो_constants_init(src0, dst);
    p.param1 = op_params[0];

    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_LEAKY_RELU, std::move(p));
}

व्योम ggml_vk_preallocate_buffers(ggml_backend_vk_context * ctx, vk_context subctx) {
#अगर defined(GGML_VULKAN_RUN_TESTS)
    स्थिर std::vector<माप_प्रकार> vals {
        512, 512, 128,
        128, 512, 512,
        4096, 512, 4096,
        11008, 512, 4096,
        4096, 512, 11008,
        32000, 512, 4096,
        8, 8, 8,
        100, 46, 576,
        623, 111, 128,
        100, 46, 558,
        512, 1, 256,
        128, 110, 622,
        511, 511, 127,
        511, 511, 7,
        511, 511, 17,
        49, 49, 128,
        128, 49, 49,
        4096, 49, 4096,
    };
    स्थिर माप_प्रकार अंक_it = 100;

    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, अंक_it, 1, 0, GGML_TYPE_Q4_0);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, अंक_it, 1, 1, GGML_TYPE_Q4_0);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, अंक_it, 1, 2, GGML_TYPE_Q4_0);

    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, अंक_it, 1, 0, GGML_TYPE_Q4_0, true);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, अंक_it, 1, 1, GGML_TYPE_Q4_0, true);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, अंक_it, 1, 2, GGML_TYPE_Q4_0, true);

    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, अंक_it, 1, 0, GGML_TYPE_Q8_0);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, अंक_it, 1, 1, GGML_TYPE_Q8_0);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, अंक_it, 1, 2, GGML_TYPE_Q8_0);

    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, अंक_it, 1, 0, GGML_TYPE_Q8_0, true);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, अंक_it, 1, 1, GGML_TYPE_Q8_0, true);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, अंक_it, 1, 2, GGML_TYPE_Q8_0, true);

    पात();

    क्रम (माप_प्रकार i = 0; i < vals.size(); i += 3) {
        ggml_vk_test_matmul<ggml_fp16_t, भग्न>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, अंक_it, 1, 0);
        ggml_vk_test_matmul<ggml_fp16_t, भग्न>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, अंक_it, 1, 1);
        ggml_vk_test_matmul<ggml_fp16_t, भग्न>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, अंक_it, 1, 2);
        std::अ_त्रुटि << '\n';
        ggml_vk_test_matmul<ggml_fp16_t, भग्न>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, अंक_it, 2, 0);
        ggml_vk_test_matmul<ggml_fp16_t, भग्न>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, अंक_it, 2, 1);
        ggml_vk_test_matmul<ggml_fp16_t, भग्न>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, अंक_it, 2, 2);
        std::अ_त्रुटि << '\n';
        ggml_vk_test_matmul<ggml_fp16_t, भग्न>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, अंक_it, 4, 0);
        ggml_vk_test_matmul<ggml_fp16_t, भग्न>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, अंक_it, 4, 1);
        ggml_vk_test_matmul<ggml_fp16_t, भग्न>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, अंक_it, 4, 2);
        std::अ_त्रुटि << '\n' << std::प_पूर्ण;

        अगर (vals[i + 2] % 32 == 0) {
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, अंक_it, 1, 0, GGML_TYPE_Q4_0);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, अंक_it, 1, 1, GGML_TYPE_Q4_0);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, अंक_it, 1, 2, GGML_TYPE_Q4_0);
            std::अ_त्रुटि << '\n';
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, अंक_it, 2, 0, GGML_TYPE_Q4_0);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, अंक_it, 2, 1, GGML_TYPE_Q4_0);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, अंक_it, 2, 2, GGML_TYPE_Q4_0);
            std::अ_त्रुटि << '\n';
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, अंक_it, 4, 0, GGML_TYPE_Q4_0);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, अंक_it, 4, 1, GGML_TYPE_Q4_0);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, अंक_it, 4, 2, GGML_TYPE_Q4_0);
            std::अ_त्रुटि << '\n' << std::प_पूर्ण;
        }

        अगर (vals[i + 2] % 256 == 0) {
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, अंक_it, 1, 0, GGML_TYPE_Q4_K);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, अंक_it, 1, 1, GGML_TYPE_Q4_K);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, अंक_it, 1, 2, GGML_TYPE_Q4_K);
            std::अ_त्रुटि << '\n';
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, अंक_it, 2, 0, GGML_TYPE_Q4_K);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, अंक_it, 2, 1, GGML_TYPE_Q4_K);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, अंक_it, 2, 2, GGML_TYPE_Q4_K);
            std::अ_त्रुटि << '\n';
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, अंक_it, 4, 0, GGML_TYPE_Q4_K);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, अंक_it, 4, 1, GGML_TYPE_Q4_K);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, अंक_it, 4, 2, GGML_TYPE_Q4_K);
            std::अ_त्रुटि << '\n' << std::प_पूर्ण;
        }
    }

    GGML_ABORT("fatal error");
#पूर्ण_अगर

    अगर (subctx) {
        // Submit and wait for any pending work before reallocating the buffers
        ggml_vk_ctx_end(subctx);
        ggml_vk_submit(subctx, {});
        ctx->submit_pending = true;
        ggml_vk_synchronize(ctx);
        GGML_ASSERT(ctx->compute_ctx.expired());
        ggml_vk_ctx_begin(ctx->device, subctx);
        ctx->compute_ctx = subctx;
    }

    अगर (ctx->prealloc_x == nullptr || (ctx->prealloc_आकार_x > 0 && ctx->prealloc_x->size < ctx->prealloc_आकार_x)) {
        VK_LOG_MEMORY("ggml_vk_preallocate_buffers(x_size: " << ctx->prealloc_आकार_x << ")");
        // Resize buffer
        अगर (ctx->prealloc_x != nullptr) {
            ggml_vk_destroy_बफर(ctx->prealloc_x);
        }
        ctx->prealloc_x = ggml_vk_create_बफर_device(ctx->device, ctx->prealloc_आकार_x);
    }
    अगर (ctx->prealloc_y == nullptr || (ctx->prealloc_आकार_y > 0 && ctx->prealloc_y->size < ctx->prealloc_आकार_y)) {
        VK_LOG_MEMORY("ggml_vk_preallocate_buffers(y_size: " << ctx->prealloc_आकार_y << ")");
        // Resize buffer
        अगर (ctx->prealloc_y != nullptr) {
            ggml_vk_destroy_बफर(ctx->prealloc_y);
        }
        ctx->prealloc_y = ggml_vk_create_बफर_device(ctx->device, ctx->prealloc_आकार_y);
        ctx->prealloc_y_last_pipeline_used = nullptr;
        ctx->prealloc_y_last_tensor_used = nullptr;
        ctx->prealloc_y_last_k_padded = false;
    }
    अगर (ctx->prealloc_split_k == nullptr || (ctx->prealloc_आकार_split_k > 0 && ctx->prealloc_split_k->size < ctx->prealloc_आकार_split_k)) {
        VK_LOG_MEMORY("ggml_vk_preallocate_buffers(split_k_size: " << ctx->prealloc_आकार_split_k << ")");
        // Resize buffer
        अगर (ctx->prealloc_split_k != nullptr) {
            ggml_vk_destroy_बफर(ctx->prealloc_split_k);
        }
        ctx->prealloc_split_k = ggml_vk_create_बफर_device(ctx->device, ctx->prealloc_आकार_split_k);
    }
    अगर (ctx->prealloc_add_rms_partials == nullptr || (ctx->prealloc_आकार_add_rms_partials > 0 && ctx->prealloc_add_rms_partials->size < ctx->prealloc_आकार_add_rms_partials)) {
        VK_LOG_MEMORY("ggml_vk_preallocate_buffers(add_partials_size: " << ctx->prealloc_add_rms_partials << ")");
        // Resize buffer
        अगर (ctx->prealloc_add_rms_partials != nullptr) {
            ggml_vk_destroy_बफर(ctx->prealloc_add_rms_partials);
        }
        ctx->prealloc_add_rms_partials = ggml_vk_create_बफर_device(ctx->device, ctx->prealloc_आकार_add_rms_partials);
    }
}

bool ggml_vk_build_graph(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, पूर्णांक गाँठ_idx, ggml_tensor *गाँठ_begin, पूर्णांक गाँठ_idx_begin, bool last_गाँठ, bool almost_ready, bool submit){
    ggml_tensor * गाँठ = cgraph->nodes[गाँठ_idx];
    अगर (ggml_is_empty(गाँठ) || ggml_op_is_empty(गाँठ->op) || !गाँठ->बफर) {
        वापस false;
    }
    अगर ((गाँठ->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
        वापस false;
    }

    VK_LOG_DEBUG("ggml_vk_build_graph(" << गाँठ << ", " << ggml_op_name(गाँठ->op) << ")");
    ctx->semaphore_idx = 0;

    ggml_tensor * src0 = गाँठ->src[0];
    ggml_tensor * src1 = गाँठ->src[1];
    ggml_tensor * src2 = गाँठ->src[2];
    ggml_tensor * src3 = गाँठ->src[3];

    अगर (गाँठ->op == GGML_OP_ADD) {
        पूर्णांक अग्र_गाँठ_idx = गाँठ_idx + 1 + ctx->अंक_additional_fused_ops;
        अगर (अग्र_गाँठ_idx < cgraph->n_nodes &&
            cgraph->nodes[अग्र_गाँठ_idx]->op == GGML_OP_RMS_NORM &&
            cgraph->nodes[अग्र_गाँठ_idx]->src[0] == cgraph->nodes[अग्र_गाँठ_idx - 1] &&
            ggml_nrows(cgraph->nodes[अग्र_गाँठ_idx]) == 1 &&
            ctx->device->add_rms_fusion) {
            uint32_t size = ggml_vk_rms_partials_आकार(ctx, cgraph->nodes[गाँठ_idx]);
            ctx->do_add_rms_partials_offset_calculation = true;
            अगर (ctx->prealloc_आकार_add_rms_partials_offset + size <= ctx->prealloc_आकार_add_rms_partials) {
                ctx->do_add_rms_partials = true;
            }
        }
    }

    vk_context compute_ctx = ggml_vk_get_compute_ctx(ctx);

    {
        // This logic detects dependencies between modes in the graph and calls ggml_vk_sync_buffers
        // to synchronize them. This handles most "normal" synchronization when computing the graph, and when
        // there is no auxiliary memory use, it shouldn't be necessary to call ggml_vk_sync_buffers
        // outside of this logic. When a node uses one of the prealloc buffers for something like
        // dequantization or split_k, additional synchronization is needed between those passes.
        bool need_sync = false;

        // Check whether "node" requires synchronization. The node requires synchronization if it
        // overlaps in memory with another unsynchronized node and at least one of them is a write.
        // Destination nodes are checked against both the written/read lists. Source nodes are only
        // checked against the written list. Two nodes overlap in memory if they come from the same
        // buffer and the tensor or view ranges overlap.
        स्वतः स्थिर &overlaps_unsynced = [&](स्थिर ggml_tensor *गाँठ, स्थिर std::vector<स्थिर ggml_tensor *> &unsynced_nodes) -> bool {
            अगर (unsynced_nodes.size() == 0) {
                वापस false;
            }
            स्वतः n_base = vk_tensor_offset(गाँठ) + गाँठ->view_offs;
            स्वतः n_आकार = ggml_nbytes(गाँठ);
            ggml_backend_vk_बफर_context * a_buf_ctx = (ggml_backend_vk_बफर_context *)गाँठ->बफर->context;
            vk_बफर a_buf = a_buf_ctx->dev_बफर;
            क्रम (स्वतः &other : unsynced_nodes) {
                ggml_backend_vk_बफर_context * o_buf_ctx = (ggml_backend_vk_बफर_context *)other->बफर->context;
                vk_बफर o_buf = o_buf_ctx->dev_बफर;
                अगर (a_buf == o_buf) {
                    स्वतः o_base = vk_tensor_offset(other) + other->view_offs;
                    स्वतः o_आकार = ggml_nbytes(other);

                    अगर ((o_base <= n_base && n_base < o_base + o_आकार) ||
                        (n_base <= o_base && o_base < n_base + n_आकार)) {
                        वापस true;
                    }
                }
            }
            वापस false;
        };

        // For all fused ops, check if the destination node or any of the source
        // nodes require synchronization.
        क्रम (int32_t i = 0; i < ctx->अंक_additional_fused_ops + 1 && !need_sync; ++i) {
            स्थिर ggml_tensor *cur_गाँठ = cgraph->nodes[गाँठ_idx + i];
            // If the node actually writes to memory, then check if it needs to sync
            अगर (ctx->fused_ops_write_mask & (1 << i)) {
                अगर (overlaps_unsynced(cur_गाँठ, ctx->unsynced_nodes_read) || overlaps_unsynced(cur_गाँठ, ctx->unsynced_nodes_written)) {
                    need_sync = true;
                    अवरोध;
                }
            }
            क्रम (uint32_t j = 0; j < GGML_MAX_SRC; ++j) {
                अगर (!cur_गाँठ->src[j]) {
                    जारी;
                }
                अगर (overlaps_unsynced(cur_गाँठ->src[j], ctx->unsynced_nodes_written)) {
                    need_sync = true;
                    अवरोध;
                }
            }
        }

        अगर (need_sync) {
            अगर (vk_enable_sync_logger) {
                std::अ_त्रुटि <<  "sync" << std::प_पूर्ण;
            }
            ctx->unsynced_nodes_written.सफा();
            ctx->unsynced_nodes_read.सफा();
            ggml_vk_sync_buffers(ctx, compute_ctx);

            अगर (vk_perf_logger_enabled && vk_perf_logger_concurrent) {
                ctx->query_गाँठ_idx[ctx->query_idx] = गाँठ_idx;
                compute_ctx->s->बफर->buf.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands, ctx->query_pool, ctx->query_idx++);
                ggml_vk_sync_buffers(ctx, compute_ctx);
            }
        }
        // Add all fused nodes to the unsynchronized lists.
        क्रम (int32_t i = 0; i < ctx->अंक_additional_fused_ops + 1; ++i) {
            स्थिर ggml_tensor *cur_गाँठ = cgraph->nodes[गाँठ_idx + i];
            // Multiple outputs could be written, e.g. in topk_moe. Add them all to the list.
            अगर (ctx->fused_ops_write_mask & (1 << i)) {
                ctx->unsynced_nodes_written.push_back(cur_गाँठ);
            }
            क्रम (uint32_t j = 0; j < GGML_MAX_SRC; ++j) {
                अगर (!cur_गाँठ->src[j]) {
                    जारी;
                }
                ctx->unsynced_nodes_read.push_back(cur_गाँठ->src[j]);
            }
        }
    }
    अगर (vk_enable_sync_logger) {
        क्रम (पूर्णांक i = 0; i < ctx->अंक_additional_fused_ops + 1; ++i) {
            स्वतः *n = cgraph->nodes[गाँठ_idx + i];
            std::अ_त्रुटि << गाँठ_idx + i << " " << ggml_op_name(n->op) << " " <<  n->name;
            अगर (n->op == GGML_OP_GLU) {
                std::अ_त्रुटि << " " << ggml_glu_op_name(ggml_get_glu_op(n)) << " " << (n->src[1] ? "split" : "single") << " ";
            }
            अगर (n->op == GGML_OP_ROPE) {
                स्थिर पूर्णांक mode = ((स्थिर int32_t *) n->op_params)[2];
                std::अ_त्रुटि << " rope mode: " << mode;
            }
            std::अ_त्रुटि << std::प_पूर्ण;
        }
    }

    // closed explicitly below, and by the destructor on the early returns
    ggml_vk_debug_label dbg(compute_ctx, cgraph, गाँठ_idx, ctx->अंक_additional_fused_ops);

    चयन (गाँठ->op) {
    हाल GGML_OP_REPEAT:
        ggml_vk_repeat(ctx, compute_ctx, src0, गाँठ);

        अवरोध;
    हाल GGML_OP_REPEAT_BACK:
        ggml_vk_repeat_back(ctx, compute_ctx, src0, गाँठ);

        अवरोध;
    हाल GGML_OP_ACC:
    हाल GGML_OP_SET:
        ggml_vk_acc(ctx, compute_ctx, src0, src1, गाँठ);

        अवरोध;
    हाल GGML_OP_GET_ROWS:
        अगर (ctx->fused_topk_qsa) {
            ggml_vk_topk_qsa(ctx, compute_ctx, cgraph, गाँठ_idx);
        } अन्यथा {
            ggml_vk_get_rows(ctx, compute_ctx, src0, src1, गाँठ);
        }

        अवरोध;
    हाल GGML_OP_GET_ROWS_BACK:
        ggml_vk_get_rows_back(ctx, compute_ctx, src0, src1, गाँठ);

        अवरोध;
    हाल GGML_OP_ADD:
        अगर (ctx->अंक_additional_fused_ops) {
            ggml_vk_multi_add(ctx, compute_ctx, cgraph, गाँठ_idx);
        } अन्यथा {
            ggml_vk_add(ctx, compute_ctx, src0, src1, गाँठ);
        }
        अवरोध;
    हाल GGML_OP_OUT_PROD:
        ggml_vk_out_prod(ctx, compute_ctx, src0, src1, गाँठ);
        अवरोध;
    हाल GGML_OP_SUB:
        ggml_vk_sub(ctx, compute_ctx, src0, src1, गाँठ);

        अवरोध;
    हाल GGML_OP_MUL:
        अगर (ctx->अंक_additional_fused_ops) {
            ggml_vk_snake_dispatch_fused(ctx, compute_ctx, cgraph, गाँठ_idx);
        } अन्यथा {
            ggml_vk_mul(ctx, compute_ctx, src0, src1, गाँठ);
        }

        अवरोध;
    हाल GGML_OP_DIV:
        ggml_vk_div(ctx, compute_ctx, src0, src1, गाँठ);

        अवरोध;
    हाल GGML_OP_ADD_ID:
        ggml_vk_add_id(ctx, compute_ctx, src0, src1, src2, गाँठ);

        अवरोध;
    हाल GGML_OP_CONCAT:
        ggml_vk_concat(ctx, compute_ctx, src0, src1, गाँठ);

        अवरोध;
    हाल GGML_OP_UPSCALE:
        ggml_vk_upscale(ctx, compute_ctx, src0, गाँठ);

        अवरोध;
    हाल GGML_OP_ADD1:
        ggml_vk_add1(ctx, compute_ctx, src0, src1, गाँठ);

        अवरोध;
    हाल GGML_OP_ARANGE:
        ggml_vk_arange(ctx, compute_ctx, गाँठ);

        अवरोध;
    हाल GGML_OP_FILL:
        ggml_vk_fill(ctx, compute_ctx, गाँठ);

        अवरोध;
    हाल GGML_OP_SCALE:
        अगर (ctx->fused_hc_post_gate) {
            ggml_tensor * hc_post = cgraph->nodes[गाँठ_idx + ctx->अंक_additional_fused_ops];
            ggml_vk_dsv4_hc_post(ctx, compute_ctx, hc_post->src[0], hc_post->src[1], hc_post->src[2], hc_post->src[3], hc_post, गाँठ);
        } अन्यथा {
            ggml_vk_scale(ctx, compute_ctx, src0, गाँठ);
        }

        अवरोध;
    हाल GGML_OP_SQR:
        ggml_vk_sqr(ctx, compute_ctx, src0, गाँठ);

        अवरोध;
    हाल GGML_OP_SQRT:
        ggml_vk_sqrt(ctx, compute_ctx, src0, गाँठ);

        अवरोध;
    हाल GGML_OP_SIN:
        ggml_vk_sin(ctx, compute_ctx, src0, गाँठ);

        अवरोध;
    हाल GGML_OP_COS:
        ggml_vk_cos(ctx, compute_ctx, src0, गाँठ);

        अवरोध;
    हाल GGML_OP_LOG:
        ggml_vk_log(ctx, compute_ctx, src0, गाँठ);

        अवरोध;
    हाल GGML_OP_TRI:
        ggml_vk_tri(ctx, compute_ctx, src0, गाँठ);

        अवरोध;
    हाल GGML_OP_DIAG:
        ggml_vk_diag(ctx, compute_ctx, src0, गाँठ);

        अवरोध;
    हाल GGML_OP_CLAMP:
        ggml_vk_clamp(ctx, compute_ctx, src0, गाँठ);

        अवरोध;
    हाल GGML_OP_PAD:
        ggml_vk_pad(ctx, compute_ctx, src0, गाँठ);

        अवरोध;
    हाल GGML_OP_PAD_REFLECT_1D:
        ggml_vk_pad_reflect_1d(ctx, compute_ctx, src0, गाँठ);

        अवरोध;
    हाल GGML_OP_ROLL:
        ggml_vk_roll(ctx, compute_ctx, src0, गाँठ);

        अवरोध;
    हाल GGML_OP_CPY:
    हाल GGML_OP_CONT:
    हाल GGML_OP_DUP:
        ggml_vk_cpy(ctx, compute_ctx, src0, गाँठ);

        अवरोध;
    हाल GGML_OP_SET_ROWS:
        ggml_vk_set_rows(ctx, compute_ctx, src0, src1, गाँठ);

        अवरोध;
    हाल GGML_OP_SILU_BACK:
        ggml_vk_silu_back(ctx, compute_ctx, src0, src1, गाँठ);

        अवरोध;
    हाल GGML_OP_NORM:
        ggml_vk_norm(ctx, compute_ctx, src0, गाँठ);

        अवरोध;
    हाल GGML_OP_GROUP_NORM:
        ggml_vk_group_norm(ctx, compute_ctx, src0, गाँठ);

        अवरोध;
    हाल GGML_OP_RMS_NORM:
        ggml_vk_rms_norm(ctx, compute_ctx, cgraph, गाँठ_idx, (भग्न *)गाँठ->op_params);
        अवरोध;
    हाल GGML_OP_RMS_NORM_BACK:
        ggml_vk_rms_norm_back(ctx, compute_ctx, src0, src1, गाँठ);

        अवरोध;
    हाल GGML_OP_L2_NORM:
        ggml_vk_l2_norm(ctx, compute_ctx, src0, गाँठ);

        अवरोध;
    हाल GGML_OP_UNARY:
        अगर (ctx->fused_topk_moe_mode != TOPK_MOE_गिनती) {
            ggml_vk_topk_moe(ctx, compute_ctx, cgraph, गाँठ_idx);
            अवरोध;
        }
        अगर (ctx->अंक_additional_fused_ops) {
            ggml_vk_unary_mul(ctx, compute_ctx, cgraph, गाँठ_idx);
            अवरोध;
        }

        चयन (ggml_get_unary_op(गाँठ)) {
        हाल GGML_UNARY_OP_ELU:
        हाल GGML_UNARY_OP_EXP:
        हाल GGML_UNARY_OP_EXPM1:
        हाल GGML_UNARY_OP_SILU:
        हाल GGML_UNARY_OP_GELU:
        हाल GGML_UNARY_OP_GELU_ERF:
        हाल GGML_UNARY_OP_GELU_QUICK:
        हाल GGML_UNARY_OP_RELU:
        हाल GGML_UNARY_OP_NEG:
        हाल GGML_UNARY_OP_TANH:
        हाल GGML_UNARY_OP_SIGMOID:
        हाल GGML_UNARY_OP_HARDSIGMOID:
        हाल GGML_UNARY_OP_HARDSWISH:
        हाल GGML_UNARY_OP_ABS:
        हाल GGML_UNARY_OP_SOFTPLUS:
        हाल GGML_UNARY_OP_STEP:
        हाल GGML_UNARY_OP_ROUND:
        हाल GGML_UNARY_OP_CEIL:
        हाल GGML_UNARY_OP_FLOOR:
        हाल GGML_UNARY_OP_TRUNC:
        हाल GGML_UNARY_OP_SGN:
            ggml_vk_unary(ctx, compute_ctx, src0, गाँठ);
            अवरोध;
        हाल GGML_UNARY_OP_XIELU:
            ggml_vk_xielu(ctx, compute_ctx, src0, गाँठ);
            अवरोध;
        शेष:
            वापस false;
        }
        अवरोध;
    हाल GGML_OP_GLU:
        चयन (ggml_get_glu_op(गाँठ)) {
        हाल GGML_GLU_OP_GEGLU:
        हाल GGML_GLU_OP_REGLU:
        हाल GGML_GLU_OP_SWIGLU:
        हाल GGML_GLU_OP_SWIGLU_OAI:
        हाल GGML_GLU_OP_GEGLU_ERF:
        हाल GGML_GLU_OP_GEGLU_QUICK:
        हाल GGML_GLU_OP_SWIGLU_CLAMP:
            ggml_vk_glu(ctx, compute_ctx, src0, src1, गाँठ);
            अवरोध;
        शेष:
            वापस false;
        }
        अवरोध;
    हाल GGML_OP_DIAG_MASK_INF:
        ggml_vk_diag_mask_inf(ctx, compute_ctx, src0, गाँठ);

        अवरोध;
    हाल GGML_OP_SOFT_MAX:
        अगर (ctx->fused_topk_moe_mode != TOPK_MOE_गिनती) {
            ggml_vk_topk_moe(ctx, compute_ctx, cgraph, गाँठ_idx);
        } अन्यथा {
            ggml_vk_soft_max(ctx, compute_ctx, src0, src1, src2, गाँठ);
        }

        अवरोध;
    हाल GGML_OP_SOFT_MAX_BACK:
        ggml_vk_soft_max_back(ctx, compute_ctx, src0, src1, गाँठ);

        अवरोध;
    हाल GGML_OP_ROPE:
        ggml_vk_rope(ctx, compute_ctx, cgraph, गाँठ_idx, false);

        अवरोध;
    हाल GGML_OP_ROPE_BACK:
        ggml_vk_rope(ctx, compute_ctx, cgraph, गाँठ_idx, true);

        अवरोध;
    हाल GGML_OP_ARGSORT:
        अगर (ctx->fused_topk_moe_mode != TOPK_MOE_गिनती) {
            ggml_vk_topk_moe(ctx, compute_ctx, cgraph, गाँठ_idx);
        } अन्यथा {
            ggml_vk_argsort(ctx, compute_ctx, src0, गाँठ);
        }

        अवरोध;
    हाल GGML_OP_शीर्ष_K:
        ggml_vk_topk(ctx, compute_ctx, src0, गाँठ);

        अवरोध;
    हाल GGML_OP_योग:
        ggml_vk_योग(ctx, compute_ctx, src0, गाँठ);

        अवरोध;
    हाल GGML_OP_योग_ROWS:
        ggml_vk_योग_rows(ctx, compute_ctx, src0, गाँठ);

        अवरोध;
    हाल GGML_OP_CUMSUM:
        ggml_vk_cumsum(ctx, compute_ctx, src0, गाँठ);

        अवरोध;
    हाल GGML_OP_DSV4_HC_COMB:
        ggml_vk_dsv4_hc_comb(ctx, compute_ctx, src0, src1, src2, गाँठ);

        अवरोध;
    हाल GGML_OP_DSV4_HC_PRE:
        ggml_vk_dsv4_hc_pre(ctx, compute_ctx, src0, src1, गाँठ);

        अवरोध;
    हाल GGML_OP_DSV4_HC_POST:
        ggml_vk_dsv4_hc_post(ctx, compute_ctx, src0, src1, src2, src3, गाँठ);

        अवरोध;
    हाल GGML_OP_MEAN:
        ggml_vk_mean(ctx, compute_ctx, src0, गाँठ);

        अवरोध;
    हाल GGML_OP_ARGMAX:
        ggml_vk_argmax(ctx, compute_ctx, src0, गाँठ);

        अवरोध;
    हाल GGML_OP_CROSS_ENTROPY_LOSS:
        ggml_vk_cross_entropy_loss(ctx, compute_ctx, गाँठ);

        अवरोध;
    हाल GGML_OP_CROSS_ENTROPY_LOSS_BACK:
        ggml_vk_cross_entropy_loss_back(ctx, compute_ctx, गाँठ);

        अवरोध;
    हाल GGML_OP_गिनती_EQUAL:
        ggml_vk_गिनती_equal(ctx, compute_ctx, src0, src1, गाँठ);

        अवरोध;
    हाल GGML_OP_SOLVE_TRI:
        ggml_vk_solve_tri(ctx, compute_ctx, src0, src1, गाँठ);

        अवरोध;
    हाल GGML_OP_IM2COL:
        ggml_vk_im2col(ctx, compute_ctx, src0, src1, गाँठ);

        अवरोध;
    हाल GGML_OP_IM2COL_3D:
        ggml_vk_im2col_3d(ctx, compute_ctx, src0, src1, गाँठ);

        अवरोध;
    हाल GGML_OP_TIMESTEP_EMBEDDING:
        ggml_vk_timestep_embedding(ctx, compute_ctx, src0, गाँठ);

        अवरोध;
    हाल GGML_OP_COL2IM_1D:
        ggml_vk_col2im_1d(ctx, compute_ctx, src0, गाँठ);

        अवरोध;
    हाल GGML_OP_CONV_TRANSPOSE_1D:
        ggml_vk_conv_transpose_1d(ctx, compute_ctx, src0, src1, गाँठ);

        अवरोध;
    हाल GGML_OP_POOL_1D:
        ggml_vk_pool_1d(ctx, compute_ctx, src0, गाँठ);

        अवरोध;
    हाल GGML_OP_POOL_2D:
        ggml_vk_pool_2d(ctx, compute_ctx, src0, गाँठ);

        अवरोध;
    हाल GGML_OP_CONV_2D:
    हाल GGML_OP_CONV_TRANSPOSE_2D:
        ggml_vk_conv_2d(ctx, compute_ctx, src0, src1, गाँठ);

        अवरोध;
    हाल GGML_OP_CONV_3D:
        ggml_vk_conv_3d(ctx, compute_ctx, src0, src1, गाँठ);

        अवरोध;
    हाल GGML_OP_CONV_2D_DW:
        ggml_vk_conv_2d_dw(ctx, compute_ctx, src0, src1, गाँठ);

        अवरोध;
    हाल GGML_OP_LEAKY_RELU:
        ggml_vk_leaky_relu(ctx, compute_ctx, src0, गाँठ);

        अवरोध;
    हाल GGML_OP_MUL_MAT:
        ggml_vk_mul_mat(ctx, compute_ctx, cgraph, गाँठ_idx);

        अवरोध;
    हाल GGML_OP_MUL_MAT_ID:
        ggml_vk_mul_mat_id(ctx, compute_ctx, cgraph, गाँठ_idx);

        अवरोध;

    हाल GGML_OP_FLASH_ATTN_EXT:
        ggml_vk_flash_attn(ctx, compute_ctx, src0, src1, src2, src3, गाँठ->src[4], गाँठ);

        अवरोध;

    हाल GGML_OP_RWKV_WKV6:
        ggml_vk_rwkv_wkv6(ctx, compute_ctx, गाँठ);

        अवरोध;

    हाल GGML_OP_RWKV_WKV7:
        ggml_vk_rwkv_wkv7(ctx, compute_ctx, गाँठ);

        अवरोध;

    हाल GGML_OP_GATED_LINEAR_ATTN:
        ggml_vk_gated_linear_attn(ctx, compute_ctx, गाँठ);

        अवरोध;

    हाल GGML_OP_LIGHTNING_INDEXER:
        ggml_vk_lightning_indexer(ctx, compute_ctx, गाँठ);

        अवरोध;

    हाल GGML_OP_GATED_DELTA_NET:
        ggml_vk_gated_delta_net(ctx, compute_ctx, गाँठ);

        अवरोध;

    हाल GGML_OP_SSM_SCAN:
        ggml_vk_ssm_scan(ctx, compute_ctx, गाँठ);

        अवरोध;

    हाल GGML_OP_SSM_CONV:
        ggml_vk_ssm_conv(ctx, compute_ctx, cgraph, गाँठ_idx);

        अवरोध;

    हाल GGML_OP_OPT_STEP_ADAMW:
        ggml_vk_opt_step_adamw(ctx, compute_ctx, गाँठ);

        अवरोध;

    हाल GGML_OP_OPT_STEP_SGD:
        ggml_vk_opt_step_sgd(ctx, compute_ctx, src0, src1, src2, गाँठ);

        अवरोध;
    शेष:
        वापस false;
    }

    // the submit path below can end the command buffer, so close the region first
    dbg.बंद();

    ctx->tensor_ctxs[गाँठ_idx] = compute_ctx;

#अगर defined(GGML_VULKAN_CHECK_RESULTS)
    // Force context reset on each node so that each tensor ends up in its own context
    // and can be run and compared to its CPU equivalent separately
    last_गाँठ = true;
#पूर्ण_अगर

    अगर (submit || last_गाँठ) {
        ggml_vk_ctx_end(compute_ctx);

        // TODO probably it'd be better to pass a exit_node flag to ggml_vk_compute_forward
        अगर (last_गाँठ) {
            compute_ctx->exit_tensor_idx = गाँठ_idx_begin;
        }
        अन्यथा {
            compute_ctx->exit_tensor_idx = -1;
        }

        ctx->compute_ctx.reset();

        ggml_vk_compute_forward(ctx, cgraph, गाँठ_begin, गाँठ_idx_begin, almost_ready);
    }
    वापस true;
}

व्योम ggml_vk_compute_forward(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, ggml_tensor * tensor, पूर्णांक tensor_idx, bool almost_ready) {
    GGML_UNUSED(cgraph);
    GGML_UNUSED(tensor);

    VK_LOG_DEBUG("ggml_vk_compute_forward(" << tensor << ", name=" << tensor->name << ", op=" << ggml_op_name(tensor->op) << ", type=" << tensor->type << ", ne0=" << tensor->ne[0] << ", ne1=" << tensor->ne[1] << ", ne2=" << tensor->ne[2] << ", ne3=" << tensor->ne[3] << ", nb0=" << tensor->nb[0] << ", nb1=" << tensor->nb[1] << ", nb2=" << tensor->nb[2] << ", nb3=" << tensor->nb[3] << ", view_src=" << tensor->view_src << ", view_offs=" << tensor->view_offs << ")");

    vk_context subctx = ctx->tensor_ctxs[tensor_idx].lock();

    // Only run if ctx hasn't been submitted yet
    अगर (!subctx->seqs.empty()) {
#अगर_घोषित GGML_VULKAN_CHECK_RESULTS
        ggml_vk_check_results_0(ctx, cgraph, tensor_idx);
#पूर्ण_अगर

        // Do staging buffer copies
        क्रम (स्वतः& cpy : subctx->in_memcpys) {
            स_नकल(cpy.dst, cpy.src, cpy.n);
        }

        क्रम (स्वतः& mset : subctx->memsets) {
            स_रखो(mset.dst, mset.val, mset.n);
        }

        अगर (ctx->device->serialize_submissions) {
            ggml_vk_submit(subctx, ctx->fence);
        } अन्यथा अगर (almost_ready && !ctx->almost_ready_fence_pending) {
            ggml_vk_submit(subctx, ctx->almost_ready_fence);
            ctx->almost_ready_fence_pending = true;
        } अन्यथा {
            ggml_vk_submit(subctx, {});
        }
        ctx->submit_pending = true;

#अगर_घोषित GGML_VULKAN_CHECK_RESULTS
        ggml_vk_synchronize(ctx);
        ggml_vk_check_results_1(ctx, cgraph, tensor_idx);
#पूर्ण_अगर
    }

    अगर (tensor_idx == subctx->exit_tensor_idx) {
        // Do staging buffer copies
        क्रम (स्वतः& cpy : subctx->out_memcpys) {
            स_नकल(cpy.dst, cpy.src, cpy.n);
        }
        subctx->in_memcpys.सफा();
        subctx->out_memcpys.सफा();
        subctx->memsets.सफा();
    }
}

व्योम ggml_vk_graph_cleanup(ggml_backend_vk_context * ctx) {
    VK_LOG_DEBUG("ggml_vk_graph_cleanup()");
    ctx->prealloc_y_last_pipeline_used = {};
    ctx->prealloc_y_last_tensor_used = nullptr;
    ctx->prealloc_y_last_k_padded = false;

    ctx->unsynced_nodes_written.सफा();
    ctx->unsynced_nodes_read.सफा();
    ctx->prealloc_x_need_sync = ctx->prealloc_y_need_sync = ctx->prealloc_split_k_need_sync = false;

    ggml_vk_command_pool_cleanup(ctx->device, ctx->compute_cmd_pool);
    अगर (ctx->device->async_use_transfer_queue) {
        ggml_vk_command_pool_cleanup(ctx->device, ctx->transfer_cmd_pool);
    }

    क्रम (माप_प्रकार i = 0; i < ctx->gc.semaphores.size(); i++) {
        ctx->device->device.destroySemaphore({ ctx->gc.semaphores[i].s });
    }
    ctx->gc.semaphores.सफा();

    क्रम (माप_प्रकार i = 0; i < ctx->gc.tl_semaphores.size(); i++) {
        ctx->device->device.destroySemaphore({ ctx->gc.tl_semaphores[i].s });
    }
    ctx->gc.tl_semaphores.सफा();
    ctx->semaphore_idx = 0;

    ctx->event_idx = 0;

    क्रम (स्वतः& event : ctx->gc.events) {
        ctx->device->device.resetEvent(event);
    }

    ctx->tensor_ctxs.सफा();
    ctx->gc.contexts.सफा();
    ctx->pipeline_descriptor_set_requirements = 0;
    ctx->descriptor_set_idx = 0;
}

व्योम ggml_vk_cleanup(ggml_backend_vk_context * ctx) {
    VK_LOG_DEBUG("ggml_vk_cleanup(" << ctx->name << ")");
    // discard any unsubmitted command buffers
    ctx->compute_ctx.reset();
    // wait for any pending command buffers to finish
    ggml_vk_synchronize(ctx);

    ggml_vk_graph_cleanup(ctx);

    ggml_vk_destroy_बफर(ctx->prealloc_x);
    ggml_vk_destroy_बफर(ctx->prealloc_y);
    ggml_vk_destroy_बफर(ctx->prealloc_split_k);
    ggml_vk_destroy_बफर(ctx->prealloc_add_rms_partials);
    ggml_vk_destroy_बफर(ctx->sync_staging);

    ctx->prealloc_y_last_pipeline_used = nullptr;
    ctx->prealloc_y_last_tensor_used = nullptr;
    ctx->prealloc_y_last_k_padded = false;

    ctx->prealloc_आकार_x = 0;
    ctx->prealloc_आकार_y = 0;
    ctx->prealloc_आकार_split_k = 0;

    क्रम (स्वतः& event : ctx->gc.events) {
        ctx->device->device.destroyEvent(event);
    }
    ctx->gc.events.सफा();

    ctx->device->device.destroyFence(ctx->fence);
    ctx->device->device.destroyFence(ctx->almost_ready_fence);

    क्रम (स्वतः& pool : ctx->descriptor_pools) {
        ctx->device->device.destroyDescriptorPool(pool);
    }
    ctx->descriptor_pools.सफा();
    ctx->descriptor_sets.सफा();
    ctx->descriptor_set_bindings.सफा();

    ctx->compute_cmd_pool.destroy(ctx->device->device);
    अगर (ctx->device->async_use_transfer_queue) {
        ctx->device->device.destroySemaphore(ctx->transfer_semaphore.s);

        ctx->transfer_cmd_pool.destroy(ctx->device->device);
    }
    अगर (vk_perf_logger_enabled) {
        ctx->perf_logger->print_timings(true);
    }
}

पूर्णांक ggml_vk_get_device_गिनती() {
    ggml_vk_instance_init();

    वापस vk_instance.device_indices.size();
}

व्योम ggml_vk_get_device_description(पूर्णांक device, अक्षर * description, माप_प्रकार description_आकार) {
    ggml_vk_instance_init();

    std::vector<vk::PhysicalDevice> devices = vk_instance.instance.enumeratePhysicalDevices();

    vk::PhysicalDeviceProperties props;
    devices[device].getProperties(&props);

    snprintf(description, description_आकार, "%s", props.deviceName.तथ्य());
}

bool ggml_backend_बफर_is_vk(ggml_backend_बफर_t बफर) {
    वापस बफर->buft->iface.get_name == ggml_backend_vk_बफर_type_name;
}

व्योम ggml_backend_vk_बफर_free_बफर(ggml_backend_बफर_t बफर) {
    VK_LOG_MEMORY("ggml_backend_vk_buffer_free_buffer()");
    ggml_backend_vk_बफर_context * ctx = (ggml_backend_vk_बफर_context *)बफर->context;
    ggml_vk_destroy_बफर(ctx->dev_बफर);
    मिटाओ ctx;
}

व्योम * ggml_backend_vk_बफर_get_base(ggml_backend_बफर_t बफर) {
    वापस vk_ptr_base;

    UNUSED(बफर);
}

क्रमागत ggml_status ggml_backend_vk_बफर_init_tensor(ggml_backend_बफर_t बफर, ggml_tensor * tensor) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_init_tensor(" << बफर << " (" << बफर->context << "), " << tensor << ")");
    अगर (tensor->view_src != nullptr) {
        GGML_ASSERT(tensor->view_src->बफर->buft == बफर->buft);
    }
    वापस GGML_STATUS_SUCCESS;
}

व्योम ggml_backend_vk_बफर_memset_tensor(ggml_backend_बफर_t बफर, ggml_tensor * tensor, uint8_t value, माप_प्रकार offset, माप_प्रकार size) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_memset_tensor(" << बफर << ", " << tensor << ", " << value << ", " << offset << ", " << size << ")");
    ggml_backend_vk_बफर_context * buf_ctx = (ggml_backend_vk_बफर_context *)बफर->context;
    vk_बफर buf = buf_ctx->dev_बफर;

    अगर (size == 0) {
        वापस;
    }

    uint32_t val32 = (uint32_t)value * 0x01010101;
    ggml_vk_बफर_memset(buf, vk_tensor_offset(tensor) + tensor->view_offs + offset, val32, size);
}

व्योम ggml_backend_vk_बफर_set_tensor(ggml_backend_बफर_t बफर, ggml_tensor * tensor, स्थिर व्योम * तथ्य, माप_प्रकार offset, माप_प्रकार size) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_set_tensor(" << बफर << ", " << tensor << ", " << तथ्य << ", " << offset << ", " << size << ")");
    ggml_backend_vk_बफर_context * buf_ctx = (ggml_backend_vk_बफर_context *)बफर->context;
    vk_बफर buf = buf_ctx->dev_बफर;

    अगर (size == 0) {
        वापस;
    }

    ggml_vk_बफर_write(buf, vk_tensor_offset(tensor) + tensor->view_offs + offset, तथ्य, size);
}

व्योम ggml_backend_vk_बफर_set_tensor_2d(ggml_backend_बफर_t बफर, ggml_tensor * tensor, स्थिर व्योम * तथ्य, माप_प्रकार offset,
                                                 माप_प्रकार size, माप_प्रकार n_copies, माप_प्रकार stride_tensor, माप_प्रकार stride_तथ्य) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_set_tensor_2d(" << बफर << ", " << tensor << ", " << तथ्य << ", " << offset << ", " << size << ", " <<
                 n_copies << ", " << stride_tensor << ", " << stride_तथ्य << ")");
    ggml_backend_vk_बफर_context * buf_ctx = (ggml_backend_vk_बफर_context *)बफर->context;
    vk_बफर buf = buf_ctx->dev_बफर;

    अगर (size == 0) {
        वापस;
    }

    ggml_vk_बफर_write_2d(buf, vk_tensor_offset(tensor) + tensor->view_offs + offset, तथ्य, stride_तथ्य, stride_tensor, size, n_copies);
}

व्योम ggml_backend_vk_बफर_get_tensor(ggml_backend_बफर_t बफर, स्थिर ggml_tensor * tensor, व्योम * तथ्य, माप_प्रकार offset, माप_प्रकार size) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_get_tensor(" << बफर << ", " << tensor << ", " << तथ्य << ", " << offset << ", " << size << ")");
    ggml_backend_vk_बफर_context * buf_ctx = (ggml_backend_vk_बफर_context *)बफर->context;

    अगर (size == 0) {
        वापस;
    }

    vk_बफर buf = buf_ctx->dev_बफर;

    ggml_vk_बफर_read(buf, vk_tensor_offset(tensor) + tensor->view_offs + offset, तथ्य, size);
}

व्योम ggml_backend_vk_बफर_get_tensor_2d(ggml_backend_बफर_t बफर, स्थिर ggml_tensor * tensor, व्योम * तथ्य, माप_प्रकार offset,
                                                 माप_प्रकार size, माप_प्रकार n_copies, माप_प्रकार stride_tensor, माप_प्रकार stride_तथ्य) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_get_tensor_2d(" << बफर << ", " << tensor << ", " << तथ्य << ", " << offset << ", " << size << ", " <<
                 n_copies << ", " << stride_tensor << ", " << stride_तथ्य << ")");
    ggml_backend_vk_बफर_context * buf_ctx = (ggml_backend_vk_बफर_context *)बफर->context;

    अगर (size == 0) {
        वापस;
    }

    vk_बफर buf = buf_ctx->dev_बफर;

    ggml_vk_बफर_read_2d(buf, vk_tensor_offset(tensor) + tensor->view_offs + offset, तथ्य, stride_tensor, stride_तथ्य, size, n_copies);
}

bool ggml_backend_vk_बफर_cpy_tensor(ggml_backend_बफर_t बफर, स्थिर ggml_tensor * src, ggml_tensor * dst) {
    अगर (ggml_nbytes(src) == 0) {
        वापस true;
    }

    अगर (ggml_backend_बफर_is_vk(src->बफर)) {
        ggml_backend_vk_बफर_context * src_buf_ctx = (ggml_backend_vk_बफर_context *)src->बफर->context;
        ggml_backend_vk_बफर_context * dst_buf_ctx = (ggml_backend_vk_बफर_context *)dst->बफर->context;

        vk_बफर src_buf = src_buf_ctx->dev_बफर;
        vk_बफर dst_buf = dst_buf_ctx->dev_बफर;

        ggml_vk_बफर_copy(dst_buf, vk_tensor_offset(dst) + dst->view_offs, src_buf, vk_tensor_offset(src) + src->view_offs, ggml_nbytes(src));

        वापस true;
    }
    वापस false;

    UNUSED(बफर);
}

व्योम ggml_backend_vk_बफर_clear(ggml_backend_बफर_t बफर, uint8_t value) {
    ggml_backend_vk_बफर_context * ctx = (ggml_backend_vk_बफर_context *)बफर->context;

    ggml_vk_बफर_memset(ctx->dev_बफर, 0, value, बफर->size);
}

स्थिर अक्षर * ggml_backend_vk_बफर_type_name(ggml_backend_बफर_type_t buft) {
    ggml_backend_vk_बफर_type_context * ctx = (ggml_backend_vk_बफर_type_context *)buft->context;

    वापस ctx->name.c_str();
}

ggml_backend_बफर_t ggml_backend_vk_बफर_type_alloc_बफर(ggml_backend_बफर_type_t buft, माप_प्रकार size) {
    VK_LOG_MEMORY("ggml_backend_vk_buffer_type_alloc_buffer(" << size << ")");
    ggml_backend_vk_बफर_type_context * ctx = (ggml_backend_vk_बफर_type_context *) buft->context;

    vk_बफर dev_बफर = nullptr;
    प्रयास {
        dev_बफर = ggml_vk_create_बफर_device(ctx->device, size);
    } पकड़ो (स्थिर vk::SystemError& e) {
        वापस nullptr;
    }

    ggml_backend_vk_बफर_context * bufctx = नया ggml_backend_vk_बफर_context(ctx->device, std::move(dev_बफर), ctx->name);

    वापस ggml_backend_बफर_init(buft, ggml_backend_vk_बफर_interface, bufctx, size);
}

माप_प्रकार ggml_backend_vk_बफर_type_get_alignment(ggml_backend_बफर_type_t buft) {
    ggml_backend_vk_बफर_type_context * ctx = (ggml_backend_vk_बफर_type_context *) buft->context;
    वापस ctx->device->properties.limits.min_Storage_बफर_Offset_Alignment;
}

माप_प्रकार ggml_backend_vk_बफर_type_get_max_आकार(ggml_backend_बफर_type_t buft) {
    ggml_backend_vk_बफर_type_context * ctx = (ggml_backend_vk_बफर_type_context *) buft->context;
    वापस ctx->device->suballocation_block_आकार;
}

माप_प्रकार ggml_backend_vk_बफर_type_get_alloc_आकार(ggml_backend_बफर_type_t buft, स्थिर ggml_tensor * tensor) {
    वापस ggml_nbytes(tensor);

    UNUSED(buft);
}

ggml_backend_बफर_type_t ggml_backend_vk_बफर_type(माप_प्रकार dev_अंक) {
    ggml_vk_instance_init();

    VK_LOG_DEBUG("ggml_backend_vk_buffer_type(" << dev_अंक << ")");

    vk_device dev = ggml_vk_get_device(dev_अंक);

    वापस &dev->बफर_type;
}

अटल स्थिर अक्षर * ggml_backend_vk_host_बफर_type_name(ggml_backend_बफर_type_t buft) {
    वापस GGML_VK_NAME "_Host";

    UNUSED(buft);
}

अटल व्योम ggml_backend_vk_host_बफर_free_बफर(ggml_backend_बफर_t बफर) {
    VK_LOG_MEMORY("ggml_backend_vk_host_buffer_free_buffer()");
    ggml_vk_host_free(बफर->context);
}

अटल ggml_backend_बफर_t ggml_backend_vk_host_बफर_type_alloc_बफर(ggml_backend_बफर_type_t buft, माप_प्रकार size) {
    VK_LOG_MEMORY("ggml_backend_vk_host_buffer_type_alloc_buffer(" << size << ")");

    size += 32;  // Behave like the CPU buffer type
    व्योम * ptr = ggml_vk_host_malloc(size);
    अगर (ptr == nullptr) {
        GGML_LOG_WARN("ggml_vulkan: Failed to allocate pinned memory\n");
        // fallback to cpu buffer
        वापस ggml_backend_buft_alloc_बफर(ggml_backend_cpu_बफर_type(), size);
    }

    ggml_backend_बफर_t बफर = ggml_backend_cpu_बफर_from_ptr(ptr, size);
    बफर->buft = buft;
    बफर->iface.free_बफर = ggml_backend_vk_host_बफर_free_बफर;

    वापस बफर;

    UNUSED(buft);
}

अटल माप_प्रकार ggml_backend_vk_host_बफर_type_get_alignment(ggml_backend_बफर_type_t buft) {
    वापस vk_instance.devices[0]->properties.limits.minMemoryMapAlignment;

    UNUSED(buft);
}

अटल माप_प्रकार ggml_backend_vk_host_बफर_type_get_max_आकार(ggml_backend_बफर_type_t buft) {
    वापस vk_instance.devices[0]->suballocation_block_आकार;

    UNUSED(buft);
}

ggml_backend_बफर_type_t ggml_backend_vk_host_बफर_type() {
    अटल काष्ठा ggml_backend_बफर_type ggml_backend_vk_बफर_type_host = {
        /* .iface    = */ {
            /* .get_name            = */ ggml_backend_vk_host_बफर_type_name,
            /* .alloc_buffer        = */ ggml_backend_vk_host_बफर_type_alloc_बफर,
            /* .alloc_buffer_n      = */ nullptr,
            /* .get_alignment       = */ ggml_backend_vk_host_बफर_type_get_alignment,
            /* .get_max_size        = */ ggml_backend_vk_host_बफर_type_get_max_आकार,
            /* .get_alloc_size      = */ ggml_backend_cpu_बफर_type()->iface.get_alloc_आकार,
            /* .get_alloc_size_n    = */ शून्य,
            /* .is_host             = */ ggml_backend_cpu_बफर_type()->iface.is_host,
        },
        /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_vk_reg(), 0),
        /* .context  = */ nullptr,
    };

    // Make sure device 0 is initialized
    ggml_vk_instance_init();
    ggml_vk_get_device(0);

    वापस &ggml_backend_vk_बफर_type_host;
}

अटल स्थिर अक्षर * ggml_backend_vk_name(ggml_backend_t backend) {
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;

    वापस ctx->name.c_str();
}

व्योम ggml_backend_vk_free(ggml_backend_t backend) {
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    VK_LOG_DEBUG("ggml_backend_vk_free(" << ctx->name << ")");

    ggml_vk_cleanup(ctx);

    मिटाओ ctx;
    मिटाओ backend;
}

अटल ggml_backend_बफर_type_t ggml_backend_vk_get_default_बफर_type(ggml_backend_t backend) {
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;

    वापस &ctx->device->बफर_type;
}

अटल व्योम ggml_backend_vk_set_tensor_2d_async(ggml_backend_t backend, ggml_tensor * tensor, स्थिर व्योम * तथ्य, माप_प्रकार offset,
                                                माप_प्रकार size, माप_प्रकार n_copies, माप_प्रकार stride_tensor, माप_प्रकार stride_तथ्य) {
    VK_LOG_DEBUG("ggml_backend_vk_set_tensor_2d_async(" << size << ", " << n_copies << ")");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    GGML_ASSERT((tensor->बफर->buft == ggml_backend_vk_get_default_बफर_type(backend) || tensor->बफर->buft == ggml_backend_vk_host_बफर_type()) && "unsupported buffer type");

    अगर (size == 0) {
        वापस;
    }

    ggml_backend_vk_बफर_context * buf_ctx = (ggml_backend_vk_बफर_context *)tensor->बफर->context;

    vk_context cpy_ctx;

    अगर (ctx->device->async_use_transfer_queue) {
        cpy_ctx = ggml_vk_get_transfer_ctx(ctx);
    } अन्यथा {
        cpy_ctx = ggml_vk_get_compute_ctx(ctx);
    }

    vk_बफर buf = buf_ctx->dev_बफर;

    स्वतः dst_offset = vk_tensor_offset(tensor) + tensor->view_offs + offset;

    bool ret = ggml_vk_बफर_write_2d_async(cpy_ctx, buf, dst_offset, तथ्य, stride_तथ्य, stride_tensor, size, n_copies);

    अगर (!ret) {
        स्थिर माप_प्रकार staging_आकार = size * n_copies;
        ggml_vk_ensure_sync_staging_बफर(ctx, staging_आकार);
        ggml_vk_sync_buffers(nullptr, cpy_ctx);

        std::vector<vk::बफर_Copy> slices(1);
        अगर (size == stride_tensor) {
            slices[0].srcOffset = 0;
            slices[0].dstOffset = dst_offset;
            slices[0].size = staging_आकार;
        } अन्यथा {
            slices.resize(n_copies);
            क्रम (माप_प्रकार i = 0; i < n_copies; i++) {
                slices[i].srcOffset = i * size;
                slices[i].dstOffset = dst_offset + i * stride_tensor;
                slices[i].size = size;
            }
        }

        cpy_ctx->s->बफर->buf.copy_बफर(ctx->sync_staging->बफर, buf->बफर, slices);

        अगर (size == stride_तथ्य) {
            deferred_memcpy(ctx->sync_staging->ptr, तथ्य, staging_आकार, &cpy_ctx->in_memcpys);
        } अन्यथा {
            क्रम (माप_प्रकार i = 0; i < n_copies; i++) {
                deferred_memcpy((uint8_t *)ctx->sync_staging->ptr + i * size, (स्थिर uint8_t *)तथ्य + i * stride_तथ्य, size, &cpy_ctx->in_memcpys);
            }
        }
        ggml_vk_synchronize(ctx);
    }
}

अटल व्योम ggml_backend_vk_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, स्थिर व्योम * तथ्य, माप_प्रकार offset, माप_प्रकार size) {
    VK_LOG_DEBUG("ggml_backend_vk_set_tensor_async(" << size << ")");
    ggml_backend_vk_set_tensor_2d_async(backend, tensor, तथ्य, offset, size, 1, size, size);
}

अटल व्योम ggml_backend_vk_get_tensor_2d_async(ggml_backend_t backend, स्थिर ggml_tensor * tensor, व्योम * तथ्य, माप_प्रकार offset,
                                                माप_प्रकार size, माप_प्रकार n_copies, माप_प्रकार stride_tensor, माप_प्रकार stride_तथ्य) {
    VK_LOG_DEBUG("ggml_backend_vk_get_tensor_2d_async(" << size << ", " << n_copies << ")");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    GGML_ASSERT((tensor->बफर->buft == ggml_backend_vk_get_default_बफर_type(backend) || tensor->बफर->buft == ggml_backend_vk_host_बफर_type()) && "unsupported buffer type");

    अगर (size == 0) {
        वापस;
    }

    ggml_backend_vk_बफर_context * buf_ctx = (ggml_backend_vk_बफर_context *)tensor->बफर->context;

    vk_context compute_ctx = ggml_vk_get_compute_ctx(ctx);

    vk_बफर buf = buf_ctx->dev_बफर;

    स्वतः src_offset = vk_tensor_offset(tensor) + tensor->view_offs + offset;
    bool ret = ggml_vk_बफर_read_2d_async(compute_ctx, buf, src_offset, तथ्य, stride_tensor, stride_तथ्य, size, n_copies);

    अगर (!ret) {
        स्थिर माप_प्रकार staging_आकार = size * n_copies;
        ggml_vk_ensure_sync_staging_बफर(ctx, staging_आकार);
        ggml_vk_sync_buffers(nullptr, compute_ctx);

        std::vector<vk::बफर_Copy> slices(1);
        अगर (size == stride_tensor) {
            slices[0].srcOffset = src_offset;
            slices[0].dstOffset = 0;
            slices[0].size = staging_आकार;
        } अन्यथा {
            slices.resize(n_copies);
            क्रम (माप_प्रकार i = 0; i < n_copies; i++) {
                slices[i].srcOffset = src_offset + i * stride_tensor;
                slices[i].dstOffset = i * size;
                slices[i].size = size;
            }
        }

        compute_ctx->s->बफर->buf.copy_बफर(buf->बफर, ctx->sync_staging->बफर, slices);

        अगर (size == stride_तथ्य) {
            deferred_memcpy(तथ्य, ctx->sync_staging->ptr, staging_आकार, &compute_ctx->out_memcpys);
        } अन्यथा {
            क्रम (माप_प्रकार i = 0; i < n_copies; i++) {
                deferred_memcpy((uint8_t *)तथ्य + i * stride_तथ्य, (स्थिर uint8_t *)ctx->sync_staging->ptr + i * size, size, &compute_ctx->out_memcpys);
            }
        }
        ggml_vk_synchronize(ctx);
    }
}

अटल व्योम ggml_backend_vk_get_tensor_async(ggml_backend_t backend, स्थिर ggml_tensor * tensor, व्योम * तथ्य, माप_प्रकार offset, माप_प्रकार size) {
    VK_LOG_DEBUG("ggml_backend_vk_get_tensor_async(" << size << ")");
    ggml_backend_vk_get_tensor_2d_async(backend, tensor, तथ्य, offset, size, 1, size, size);
}

अटल bool ggml_backend_vk_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, स्थिर ggml_tensor * src, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_backend_vk_cpy_tensor_async(" << src << " -> " << dst << ", size=" << ggml_nbytes(src) << ")");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend_dst->context;

    // Skip zero-size tensors
    अगर (ggml_nbytes(src) == 0) {
        वापस true;
    }

    अगर (dst->बफर->buft != ggml_backend_vk_get_default_बफर_type(backend_dst)) {
        वापस false;
    }

    ggml_backend_vk_बफर_context * dst_buf_ctx = (ggml_backend_vk_बफर_context *)dst->बफर->context;
    vk_बफर dst_buf = dst_buf_ctx->dev_बफर;

    अगर (ggml_backend_बफर_is_vk(src->बफर)) {
        ggml_backend_vk_बफर_context * src_buf_ctx = (ggml_backend_vk_बफर_context *)src->बफर->context;

        // Async copy only works within the same device
        अगर (src_buf_ctx->dev_बफर->device != dst_buf->device) {
            वापस false;
        }

        vk_context compute_ctx = ggml_vk_get_compute_ctx(ctx);

        ggml_vk_बफर_copy_async(compute_ctx, dst_buf, vk_tensor_offset(dst) + dst->view_offs,
                                   src_buf_ctx->dev_बफर, vk_tensor_offset(src) + src->view_offs,
                                   ggml_nbytes(src));
        वापस true;
    }

    अगर (ggml_backend_बफर_is_host(src->बफर)) {
        vk_बफर pinned_buf = nullptr;
        माप_प्रकार pinned_offset = 0;
        ggml_vk_host_get(ctx->device, src->तथ्य, pinned_buf, pinned_offset);
        अगर (pinned_buf == nullptr) {
            वापस false;
        }

        // If the backend is idle, use a CPU copy to avoid GPU synchronization overhead.
        अटल constexpr माप_प्रकार max_cpu_copy_आकार = 128 * 1024;
        स्थिर bool src_backend_synchronous = backend_src->iface.synchronize == nullptr;
        स्थिर bool transfer_idle = !ctx->device->async_use_transfer_queue ||
                                   ctx->transfer_semaphore_last_submitted == ctx->transfer_semaphore.value;
        स्थिर bool backend_idle = ctx->compute_ctx.expired() && ctx->transfer_ctx.expired() &&
                                  !ctx->submit_pending && !ctx->almost_ready_fence_pending && transfer_idle;
        स्थिर bool dst_host_coherent =
            (dst_buf->memory_property_flags & (vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent)) ==
            (vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

        अगर ((backend_src == backend_dst || src_backend_synchronous) && backend_idle && dst_host_coherent && ggml_nbytes(src) <= max_cpu_copy_आकार) {
            ggml_vk_बफर_write(dst_buf, vk_tensor_offset(dst) + dst->view_offs, src->तथ्य, ggml_nbytes(src));
            वापस true;
        }

        vk_context cpy_ctx;
        अगर (ctx->device->async_use_transfer_queue) {
            cpy_ctx = ggml_vk_get_transfer_ctx(ctx);
        } अन्यथा {
            cpy_ctx = ggml_vk_get_compute_ctx(ctx);
        }

        वापस ggml_vk_बफर_write_async(cpy_ctx, dst_buf,
                                          vk_tensor_offset(dst) + dst->view_offs,
                                          src->तथ्य, ggml_nbytes(src));
    }

    वापस false;
}

व्योम ggml_vk_synchronize(ggml_backend_vk_context * ctx) {
    VK_LOG_DEBUG("ggml_vk_synchronize()");

    bool do_transfer = !ctx->compute_ctx.expired();

    अगर (ggml_vk_submit_transfer_ctx(ctx)) {
        ctx->submit_pending = true;
    }

    vk_context compute_ctx;
    vk_command_बफर* cmd_buf = nullptr;
    अगर (do_transfer) {
        compute_ctx = ctx->compute_ctx.lock();
        अगर (compute_ctx->s) {
            cmd_buf = compute_ctx->s->बफर;
        }

        ggml_vk_ctx_end(compute_ctx);

        क्रम (स्वतः& cpy : compute_ctx->in_memcpys) {
            स_नकल(cpy.dst, cpy.src, cpy.n);
        }

        अगर (ctx->device->serialize_submissions) {
            ggml_vk_submit(compute_ctx, ctx->fence);
            VK_CHECK(ctx->device->device.waitForFences({ ctx->fence }, true, UINT64_MAX), "synchronize waitForFences", ctx->device);
            ctx->device->device.resetFences({ ctx->fence });
        } अन्यथा {
            ggml_vk_submit(compute_ctx, {});
        }
        ctx->submit_pending = true;
    }

    अगर (ctx->submit_pending) {
        अगर (ctx->device->serialize_submissions) {
            ctx->submit_pending = false;
        } अन्यथा अगर (ctx->device->async_use_transfer_queue && ctx->transfer_semaphore_last_submitted < ctx->transfer_semaphore.value) {
            vk::TimelineSemaphoreSubmitInfo tl_info{
                1, &ctx->transfer_semaphore.value,
                0, nullptr,
            };
            vk::PipelineStageFlags stage = ctx->device->transfer_queue->stage_flags;
            vk::SubmitInfo si{
                1, &ctx->transfer_semaphore.s, &stage,
                0, nullptr,
                0, nullptr,
            };
            si.setPNext(&tl_info);
            ctx->device->compute_queue->handle->submit({ si }, ctx->fence);
            ctx->transfer_semaphore_last_submitted = ctx->transfer_semaphore.value;
        } अन्यथा {
            ctx->device->compute_queue->handle->submit({}, ctx->fence);
        }
        अगर (!ctx->device->serialize_submissions) {
            ggml_vk_wait_for_fence(ctx);
        }
        ctx->submit_pending = false;
        अगर (cmd_buf) {
            cmd_buf->in_use = false;
            cmd_buf->buf.reset();
        }
    }

    अगर (do_transfer) {
        क्रम (स्वतः& cpy : compute_ctx->out_memcpys) {
            स_नकल(cpy.dst, cpy.src, cpy.n);
        }
        ctx->compute_ctx.reset();
    }
}

अटल व्योम ggml_backend_vk_synchronize(ggml_backend_t backend) {
    VK_LOG_DEBUG("ggml_backend_vk_synchronize()");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;

    ggml_vk_synchronize(ctx);

    ggml_vk_graph_cleanup(ctx);
}

bool ggml_vk_is_empty(ggml_tensor * गाँठ) {
    वापस ggml_is_empty(गाँठ) || गाँठ->op == GGML_OP_NONE || गाँठ->op == GGML_OP_RESHAPE || गाँठ->op == GGML_OP_TRANSPOSE || गाँठ->op == GGML_OP_VIEW || गाँठ->op == GGML_OP_PERMUTE;
}

अटल bool ggml_vk_can_fuse_unary_mul(स्थिर काष्ठा ggml_cgraph * cgraph, पूर्णांक unary_idx, पूर्णांक mul_idx) {
    स्थिर ggml_tensor * unary = cgraph->nodes[unary_idx];
    स्थिर ggml_tensor * mul = cgraph->nodes[mul_idx];

    अगर (ggml_vk_unary_mul_op_सूचकांक(ggml_get_unary_op(unary)) < 0) {
        वापस false;
    }
    अगर (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16) {
        वापस false;
    }
    अगर (unary->type != mul->type) {
        वापस false;
    }
    अगर (mul->src[0] != unary && mul->src[1] != unary) {
        वापस false;
    }
    स्थिर ggml_tensor * other = (mul->src[0] == unary) ? mul->src[1] : mul->src[0];
    अगर (other == nullptr || other->type != unary->type) {
        वापस false;
    }
    अगर (!ggml_is_contiguous_1(other) || !ggml_is_contiguous_1(unary->src[0])) {
        वापस false;
    }
    // fastmod needs src to tile into dst
    अगर (mul->src[0] == unary) {
        वापस ggml_can_repeat(other, unary);
    }
    वापस ggml_can_repeat(unary, mul->src[0]);
}

अटल bool ggml_vk_can_fuse_unary_mul_pair(स्थिर काष्ठा ggml_cgraph * cgraph, पूर्णांक गाँठ_idx) {
    स्थिर क्रमागत ggml_op ops[]    = { GGML_OP_UNARY, GGML_OP_MUL };
    स्थिर पूर्णांक           outputs[] = { गाँठ_idx + 1 };
    वापस ggml_can_fuse_subgraph(cgraph, गाँठ_idx, 2, ops, outputs, 1) &&
           ggml_vk_can_fuse_unary_mul(cgraph, गाँठ_idx, गाँठ_idx + 1);
}

अटल bool ggml_vk_can_fuse_hc_post_gate(स्थिर काष्ठा ggml_cgraph * cgraph, पूर्णांक गाँठ_idx) {
    स्थिर ggml_tensor * scale_in  = cgraph->nodes[गाँठ_idx];
    स्थिर ggml_tensor * sigmoid   = cgraph->nodes[गाँठ_idx + 1];
    स्थिर ggml_tensor * scale_out = cgraph->nodes[गाँठ_idx + 2];

    // the shader folds scale -> sigmoid -> scale; a bias on either scale is not handled
    वापस ggml_get_unary_op(sigmoid) == GGML_UNARY_OP_SIGMOID &&
           ggml_get_op_params_f32(scale_in, 1) == 0.0f &&
           ggml_get_op_params_f32(scale_out, 1) == 0.0f &&
           scale_in->src[0]->type == GGML_TYPE_F32 &&
           ggml_are_same_shape(scale_in->src[0], scale_out);
}

bool ggml_vk_can_fuse(स्थिर ggml_backend_vk_context * ctx, स्थिर काष्ठा ggml_cgraph * cgraph, पूर्णांक गाँठ_idx, std::initializer_list<क्रमागत ggml_op> ops) {
    अगर (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_MUL) {
        वापस ggml_vk_can_fuse_unary_mul_pair(cgraph, गाँठ_idx);
    }

    अगर (!ggml_can_fuse(cgraph, गाँठ_idx, ops)) {
        वापस false;
    }

    अगर ((ops.size() == 2 || ops.size() == 3 || ops.size() == 4) &&
        ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL) {
        // additional constraints specific to this fusion
        स्थिर ggml_tensor *rms_norm = cgraph->nodes[गाँठ_idx];
        स्थिर ggml_tensor *mul = cgraph->nodes[गाँठ_idx + 1];

        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);
        // rms_norm only supports f32
        अगर (mul->src[0]->type != GGML_TYPE_F32 ||
            mul->src[1]->type != GGML_TYPE_F32 ||
            mul->type != GGML_TYPE_F32) {
            वापस false;
        }
        // if rms_norm is the B operand, then we don't handle broadcast
        अगर (rms_norm == mul->src[1] &&
            !ggml_are_same_shape(mul->src[0], rms_norm)) {
            वापस false;
        }
        // rms_norm shader assumes contiguous rows
        अगर (!ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
            वापस false;
        }

        अगर (ops.size() >= 3 && ops.begin()[2] == GGML_OP_ADD) {
            स्थिर ggml_tensor *add = cgraph->nodes[गाँठ_idx + 2];
            स्थिर ggml_tensor *residual = add->src[0] == mul ? add->src[1] : add->src[0];
            अगर (add->src[0] != mul && add->src[1] != mul) {
                वापस false;
            }
            अगर (residual->type != GGML_TYPE_F32 || add->type != GGML_TYPE_F32 ||
                !ggml_are_same_shape(add, residual) || !ggml_is_contiguous(residual) ||
                !ggml_is_contiguous(add) || get_misalign_bytes(ctx, residual) != 0) {
                वापस false;
            }

            स्थिर ggml_tensor *dst = add;
            अगर (ops.size() == 4) {
                अगर (ops.begin()[3] != GGML_OP_MUL) {
                    वापस false;
                }

                स्थिर ggml_tensor *post_mul = cgraph->nodes[गाँठ_idx + 3];
                स्थिर ggml_tensor *scale = post_mul->src[0] == add ? post_mul->src[1] : post_mul->src[0];
                अगर (post_mul->src[0] != add && post_mul->src[1] != add) {
                    वापस false;
                }
                // The shader reads data_e[0], so the final multiply must use a scalar.
                अगर (scale->type != GGML_TYPE_F32 || post_mul->type != GGML_TYPE_F32 ||
                    ggml_nelements(scale) != 1 || !ggml_is_contiguous(post_mul) ||
                    get_misalign_bytes(ctx, scale) != 0) {
                    वापस false;
                }
                dst = post_mul;
            }

            अगर (get_misalign_bytes(ctx, dst) != 0) {
                वापस false;
            }
        }
    }

    स्वतः स्थिर &mm_add_ok = [&](स्थिर ggml_tensor *mul, स्थिर ggml_tensor *add) {
        स्थिर ggml_tensor *bias = add->src[0] == mul ? add->src[1] : add->src[0];

        // mat-vec only
        अगर (ggml_nrows(mul) != 1) {
            वापस false;
        }
        // shaders assume the types match
        अगर (mul->type != bias->type) {
            वापस false;
        }
        // shaders reuse the D shape for bias
        अगर (!ggml_are_same_shape(mul, bias) ||
            !ggml_are_same_stride(mul, bias)) {
            वापस false;
        }
        // unaligned bias isn't handled
        अगर (get_misalign_bytes(ctx, bias) != 0) {
            वापस false;
        }
        वापस true;
    };

    अगर ((ops.size() == 2 || ops.size() == 3) && ops.begin()[0] == GGML_OP_MUL_MAT && ops.begin()[1] == GGML_OP_ADD) {
        // additional constraints specific to this fusion
        स्थिर ggml_tensor *mul = cgraph->nodes[गाँठ_idx];
        स्थिर ggml_tensor *add = cgraph->nodes[गाँठ_idx + 1];

        अगर (!mm_add_ok(mul, add)) {
            वापस false;
        }
        अगर (ops.size() == 3) {
            अगर (ops.begin()[2] != GGML_OP_ADD) {
                वापस false;
            }
            अगर (!mm_add_ok(add, cgraph->nodes[गाँठ_idx + 2])) {
                वापस false;
            }
        }
    }

    स्वतः स्थिर &mmid_mul_ok = [&](स्थिर ggml_tensor *mmid, स्थिर ggml_tensor *mul) {
        स्थिर ggml_tensor *scale = mul->src[1];

        अगर (mmid != mul->src[0]) {
            वापस false;
        }
        // mat-vec only
        अगर (!ggml_vk_use_mul_mat_vec_id(cgraph, गाँठ_idx)) {
            वापस false;
        }
        // shaders assume the types match
        अगर (mmid->type != scale->type) {
            वापस false;
        }
        // shaders assume the bias is contiguous
        अगर (!ggml_is_contiguous(scale)) {
            वापस false;
        }
        // unaligned bias isn't handled
        अगर (get_misalign_bytes(ctx, scale) != 0) {
            वापस false;
        }
        // shader only indexes by expert index
        अगर (scale->ne[0] != 1 ||
            scale->ne[1] != mul->ne[1] ||
            scale->ne[2] != 1 ||
            scale->ne[3] != 1) {
            वापस false;
        }
        वापस true;
    };

    अगर ((ops.size() == 2 || ops.size() == 3) && ops.begin()[0] == GGML_OP_MUL_MAT_ID && ops.begin()[1] == GGML_OP_ADD_ID) {
        // additional constraints specific to this fusion
        स्थिर ggml_tensor *mul = cgraph->nodes[गाँठ_idx];
        स्थिर ggml_tensor *add = cgraph->nodes[गाँठ_idx + 1];
        स्थिर ggml_tensor *bias = add->src[1];

        अगर (mul != add->src[0]) {
            वापस false;
        }
        // mat-vec only
        अगर (!ggml_vk_use_mul_mat_vec_id(cgraph, गाँठ_idx)) {
            वापस false;
        }
        // shaders assume the types match
        अगर (mul->type != bias->type) {
            वापस false;
        }
        // shaders assume the bias is contiguous
        अगर (!ggml_is_contiguous(bias)) {
            वापस false;
        }
        // the ID tensor must be the same for mul_mat_id and add_id
        अगर (mul->src[2] != add->src[2]) {
            वापस false;
        }
        // unaligned bias isn't handled
        अगर (get_misalign_bytes(ctx, bias) != 0) {
            वापस false;
        }

        अगर (ops.size() == 3) {
            अगर (ops.begin()[2] != GGML_OP_MUL) {
                वापस false;
            }
            स्थिर ggml_tensor *mul = cgraph->nodes[गाँठ_idx + 2];
            वापस mmid_mul_ok(add, mul);
        }
    }

    अगर (ops.size() == 2 && ops.begin()[0] == GGML_OP_MUL_MAT_ID && ops.begin()[1] == GGML_OP_MUL) {
        // additional constraints specific to this fusion
        स्थिर ggml_tensor *mmid = cgraph->nodes[गाँठ_idx];
        स्थिर ggml_tensor *mul = cgraph->nodes[गाँठ_idx + 1];

        अगर (!mmid_mul_ok(mmid, mul)) {
            वापस false;
        }
    }

    वापस true;
}

bool ggml_vk_can_fuse_ssm_conv(स्थिर ggml_backend_vk_context * ctx, स्थिर काष्ठा ggml_cgraph * cgraph,
                                      पूर्णांक गाँठ_idx, पूर्णांक अंक_extra) {
    स्थिर ggml_tensor * conv = cgraph->nodes[गाँठ_idx];
    अगर (conv->op != GGML_OP_SSM_CONV) {
        वापस false;
    }

    स्थिर ggml_tensor * silu = nullptr;
    स्थिर ggml_tensor * bias = nullptr;

    अगर (अंक_extra == 1) {
        अगर (!ggml_can_fuse(cgraph, गाँठ_idx, { GGML_OP_SSM_CONV, GGML_OP_UNARY })) {
            वापस false;
        }
        silu = cgraph->nodes[गाँठ_idx + 1];
    } अन्यथा अगर (अंक_extra == 2) {
        अगर (!ggml_can_fuse(cgraph, गाँठ_idx, { GGML_OP_SSM_CONV, GGML_OP_ADD, GGML_OP_UNARY })) {
            वापस false;
        }
        स्थिर ggml_tensor * add = cgraph->nodes[गाँठ_idx + 1];
        silu = cgraph->nodes[गाँठ_idx + 2];
        bias = (add->src[0] == conv) ? add->src[1] : add->src[0];

        अगर (bias->type != GGML_TYPE_F32 || !ggml_is_contiguous(bias)) {
            वापस false;
        }
        // bias must be channel-wise (one element per channel of the conv output)
        अगर (ggml_nelements(bias) != conv->ne[0] || bias->ne[0] != conv->ne[0]) {
            वापस false;
        }
        अगर (add->type != GGML_TYPE_F32) {
            वापस false;
        }
        // The shader doesn't apply per-tensor offsets, so reject misaligned bias.
        अगर (get_misalign_bytes(ctx, bias) != 0) {
            वापस false;
        }
    } अन्यथा {
        वापस false;
    }

    अगर (ggml_get_unary_op(silu) != GGML_UNARY_OP_SILU) {
        वापस false;
    }
    अगर (conv->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
        वापस false;
    }
    // The shader writes to the fused dst using its own strides, but the push constants don't
    // carry a per-tensor offset, so the binding must be naturally aligned.
    अगर (get_misalign_bytes(ctx, silu) != 0) {
        वापस false;
    }
    वापस true;
}

bool ggml_vk_can_fuse_topk_moe(ggml_backend_vk_context * ctx, स्थिर काष्ठा ggml_cgraph * cgraph,
                                      पूर्णांक गाँठ_idx, topk_moe_mode mode) {

    स्थिर ggml_tensor * softmax;
    स्थिर ggml_tensor * weights;
    स्थिर ggml_tensor * get_rows;
    स्थिर ggml_tensor * argsort;

    चयन (mode) {
    हाल TOPK_MOE_EARLY_SOFTMAX_NORM:
        softmax = cgraph->nodes[गाँठ_idx + 0];
        weights = cgraph->nodes[गाँठ_idx + 9];
        get_rows = cgraph->nodes[गाँठ_idx + 4];
        argsort = cgraph->nodes[गाँठ_idx + 2];
        अवरोध;
    हाल TOPK_MOE_SIGMOID_NORM_BIAS:
        softmax = cgraph->nodes[गाँठ_idx + 0]; // really sigmoid
        weights = cgraph->nodes[गाँठ_idx + 10];
        get_rows = cgraph->nodes[गाँठ_idx + 5];
        argsort = cgraph->nodes[गाँठ_idx + 3];
        अगर (ggml_get_unary_op(softmax) != GGML_UNARY_OP_SIGMOID) {
            वापस false;
        }
        // bias is expected to be 1D
        अगर (ggml_nrows(cgraph->nodes[गाँठ_idx + 2]->src[1]) != 1 ||
            !ggml_is_contiguous(cgraph->nodes[गाँठ_idx + 2]->src[1])) {
            वापस false;
        }
        // sigmoid fusion seems to generate infinities on moltenvk
        अगर (ctx->device->driver_id == vk::DriverId::eMoltenvk) {
            वापस false;
        }
        अवरोध;
    हाल TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS:
        softmax = cgraph->nodes[गाँठ_idx + 0]; // really softplus
        weights = cgraph->nodes[गाँठ_idx + 11];
        get_rows = cgraph->nodes[गाँठ_idx + 6];
        argsort = cgraph->nodes[गाँठ_idx + 4];
        अगर (ggml_get_unary_op(softmax) != GGML_UNARY_OP_SOFTPLUS) {
            वापस false;
        }
        // bias is expected to be 1D
        अगर (ggml_nrows(cgraph->nodes[गाँठ_idx + 3]->src[1]) != 1 ||
            !ggml_is_contiguous(cgraph->nodes[गाँठ_idx + 3]->src[1])) {
            वापस false;
        }
        अवरोध;
    हाल TOPK_MOE_EARLY_SOFTMAX:
        softmax = cgraph->nodes[गाँठ_idx + 0];
        weights = cgraph->nodes[गाँठ_idx + 4];
        get_rows = cgraph->nodes[गाँठ_idx + 4];
        argsort = cgraph->nodes[गाँठ_idx + 2];
        अवरोध;
    हाल TOPK_MOE_LATE_SOFTMAX:
        softmax = cgraph->nodes[गाँठ_idx + 4];
        weights = cgraph->nodes[गाँठ_idx + 5];
        get_rows = cgraph->nodes[गाँठ_idx + 2];
        argsort = cgraph->nodes[गाँठ_idx + 0];
        अवरोध;
    शेष:
        वापस false;
    }

    ggml_tensor * probs = get_rows->src[0];
    अगर (probs->op != GGML_OP_RESHAPE) {
        वापस false;
    }
    probs = probs->src[0];
    ggml_tensor * selection_probs = argsort->src[0];

    अगर (probs != selection_probs &&
        mode != TOPK_MOE_SIGMOID_NORM_BIAS &&
        mode != TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS) {
        वापस false;
    }

    अगर (!ggml_is_contiguous(softmax->src[0]) || !ggml_is_contiguous(weights)) {
        वापस false;
    }

    अगर (softmax->op == GGML_OP_SOFT_MAX) {
        स्थिर भग्न * op_params = (स्थिर भग्न *)softmax->op_params;

        भग्न scale = op_params[0];
        भग्न max_bias = op_params[1];

        अगर (scale != 1.0f || max_bias != 0.0f) {
            वापस false;
        }

        // don't fuse when masks or sinks are present
        अगर (softmax->src[1] || softmax->src[2]) {
            वापस false;
        }
    }

    स्थिर पूर्णांक n_expert = softmax->ne[0];
    अगर (n_expert > (1 << (अंक_topk_moe_pipelines-1))) {
        वापस false;
    }

    अगर (!ctx->device->subgroup_arithmetic ||
        !ctx->device->subgroup_shuffle ||
        !ctx->device->subgroup_require_full_support ||
        ctx->device->disable_fusion) {
        वापस false;
    }

    वापस true;
}

अटल bool ggml_vk_match_ops(स्थिर काष्ठा ggml_cgraph * cgraph, पूर्णांक गाँठ_idx,
                              स्थिर std::initializer_list<ggml_op> & ops) {
    अगर (गाँठ_idx + (पूर्णांक) ops.size() > cgraph->n_nodes) {
        वापस false;
    }
    क्रम (माप_प्रकार j = 0; j < ops.size(); ++j) {
        स्थिर ggml_tensor * गाँठ = cgraph->nodes[गाँठ_idx + j];
        अगर (गाँठ->op != ops.begin()[j] ||
            (गाँठ->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 ||
            (गाँठ->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
            वापस false;
        }
    }
    वापस true;
}

bool ggml_vk_can_fuse_topk_qsa(ggml_backend_vk_context * ctx, स्थिर काष्ठा ggml_cgraph * cgraph, पूर्णांक गाँठ_idx) {
    अगर (ctx->device->disable_fusion || !ctx->device->pipeline_topk_radix_qsa) {
        वापस false;
    }

    स्थिर पूर्णांक n_ops = topk_qsa_pattern.size();
    अगर (!ggml_vk_match_ops(cgraph, गाँठ_idx, topk_qsa_pattern) ||
        !ggml_check_edges(cgraph, गाँठ_idx, topk_qsa_edges)) {
        वापस false;
    }

    // elided nodes must be single-use (cpy counts its own src[1] self-reference)
    क्रम (पूर्णांक j = 0; j < n_ops - 1; ++j) {
        स्थिर ggml_tensor * गाँठ = cgraph->nodes[गाँठ_idx + j];
        स्थिर int32_t want = गाँठ->op == GGML_OP_CPY ? 2 : 1;
        अगर (ggml_गाँठ_get_use_गिनती(cgraph, गाँठ_idx + j) != want) {
            वापस false;
        }
    }

    स्थिर ggml_tensor * get_rows = cgraph->nodes[गाँठ_idx + 0];
    स्थिर ggml_tensor * add      = cgraph->nodes[गाँठ_idx + n_ops - 2];
    स्थिर ggml_tensor * शीर्ष_k    = cgraph->nodes[गाँठ_idx + n_ops - 1];

    स्थिर ggml_tensor * scores   = get_rows->src[0]; // [n_tps, n_blocks, n_stream]
    स्थिर ggml_tensor * cell_blk = get_rows->src[1]; // [n_kv, n_stream]
    स्थिर ggml_tensor * expanded = add->src[0];      // [n_kv, n_tps, n_stream]

    // raw mask: follow the reshape/cpy chain back to the materialized f16 input
    स्थिर ggml_tensor * mask = add->src[1];
    जबतक (mask && (mask->op == GGML_OP_RESHAPE || mask->op == GGML_OP_CPY)) {
        mask = mask->src[0];
    }
    अगर (!mask || mask->type != GGML_TYPE_F16) {
        वापस false;
    }

    अगर (scores->type != GGML_TYPE_F32 || cell_blk->type != GGML_TYPE_I32 || शीर्ष_k->type != GGML_TYPE_I32) {
        वापस false;
    }
    अगर (!ggml_is_contiguous(scores) || !ggml_is_contiguous(cell_blk) || !ggml_is_contiguous(mask) ||
        !ggml_is_contiguous(expanded) || !ggml_is_contiguous(शीर्ष_k)) {
        वापस false;
    }

    स्थिर int64_t n_tps    = scores->ne[0];
    स्थिर int64_t n_blocks = scores->ne[1];
    स्थिर int64_t n_stream = scores->ne[2];
    स्थिर int64_t n_kv     = cell_blk->ne[0];
    स्थिर int64_t width    = शीर्ष_k->ne[0];

    // pin the indexer layout the shader's addressing assumes
    अगर (scores->ne[3] != 1 || cell_blk->ne[1] != n_stream || ggml_nrows(cell_blk) != n_stream ||
        ggml_nelements(mask) != n_kv * n_tps * n_stream ||
        expanded->ne[0] != n_kv || expanded->ne[1] != n_tps || expanded->ne[2] != n_stream ||
        शीर्ष_k->ne[1] != n_tps || शीर्ष_k->ne[2] != n_stream || शीर्ष_k->ne[3] != 1 ||
        n_blocks <= 0 || n_kv <= 0 || width <= 0 || width > n_kv) {
        वापस false;
    }

    // only worth it in the radix regime; small k uses the faster tournament unfused
    स्थिर uint32_t k_min_pipeline = std::max((uint32_t) log2f(भग्न(width)) + 1, ctx->device->subgroup_आकार_log2);
    अगर (k_min_pipeline < अंक_topk_pipelines && ctx->device->pipeline_topk_f32[k_min_pipeline]) {
        वापस false;
    }
    वापस true;
}

bool ggml_vk_can_fuse_rope_set_rows(ggml_backend_vk_context * ctx, स्थिर काष्ठा ggml_cgraph * cgraph,
                                           पूर्णांक गाँठ_idx) {
    स्थिर ggml_tensor *rope = cgraph->nodes[गाँठ_idx + 0];
    स्थिर ggml_tensor *view = cgraph->nodes[गाँठ_idx + 1];
    स्थिर ggml_tensor *set_rows = cgraph->nodes[गाँठ_idx + 2];

    // The set_rows epilogue uses one index per ne2 slice and does not encode ne3.
    अगर (rope->src[0]->ne[3] != 1) {
        वापस false;
    }

    अगर (set_rows->type != GGML_TYPE_F32 && set_rows->type != GGML_TYPE_F16) {
        वापस false;
    }

    // The shader reads each aligned I64 index as a uvec2 and uses its low 32 bits.
    अगर (set_rows->src[1]->type != GGML_TYPE_I64 || !ggml_is_contiguous(set_rows->src[1]) ||
        set_rows->nb[0] != ggml_type_आकार(set_rows->type) || get_misalign_bytes(ctx, set_rows->src[1]) != 0) {
        वापस false;
    }

    // SET_ROWS consumes one flattened [ne0*ne1] row for each ne2 slice.
    अगर (!ggml_is_contiguous(view) ||
        view->ne[0] != rope->ne[0] * rope->ne[1] || view->ne[1] != rope->ne[2] ||
        view->ne[2] != 1 || view->ne[3] != 1 ||
        ggml_nelements(set_rows->src[1]) != rope->ne[2]) {
        वापस false;
    }

    // Only norm/neox/mrope/imrope shaders have the fusion code
    स्थिर पूर्णांक mode = ((स्थिर int32_t *) rope->op_params)[2];
    अगर (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX &&
        mode != GGML_ROPE_TYPE_MROPE && mode != GGML_ROPE_TYPE_IMROPE) {
        वापस false;
    }

    वापस true;
}

bool ggml_vk_can_fuse_rms_norm_set_rows(ggml_backend_vk_context * ctx, स्थिर काष्ठा ggml_cgraph * cgraph,
                                               पूर्णांक गाँठ_idx) {
    स्थिर ggml_tensor * rms = cgraph->nodes[गाँठ_idx];
    स्थिर ggml_tensor * view = cgraph->nodes[गाँठ_idx + 1];
    स्थिर ggml_tensor * set_rows = cgraph->nodes[गाँठ_idx + 2];

    // The RMS kernel reads F32 and writes directly to the F32 or F16 SET_ROWS destination.
    अगर (rms->src[0]->type != GGML_TYPE_F32 || rms->type != GGML_TYPE_F32 ||
        (set_rows->type != GGML_TYPE_F32 && set_rows->type != GGML_TYPE_F16) ||
        set_rows->src[1]->type != GGML_TYPE_I64 || !ggml_is_contiguous(set_rows->src[1]) ||
        set_rows->nb[0] != ggml_type_आकार(set_rows->type) || get_misalign_bytes(ctx, set_rows->src[1]) != 0) {
        वापस false;
    }
    // As with the ROPE epilogue, each ne2 slice supplies one flattened row and ne3 is not encoded.
    अगर (rms->ne[3] != 1 || !ggml_is_contiguous(rms->src[0]) || !ggml_is_contiguous(view)) {
        वापस false;
    }
    अगर (view->ne[0] != rms->ne[0] * rms->ne[1] || view->ne[1] != rms->ne[2] ||
        view->ne[2] != 1 || view->ne[3] != 1 ||
        ggml_nelements(set_rows->src[1]) != rms->ne[2]) {
        वापस false;
    }

    वापस true;
}

bool ggml_vk_can_fuse_snake(ggml_backend_vk_context * ctx, स्थिर काष्ठा ggml_cgraph * cgraph, पूर्णांक गाँठ_idx) {
    GGML_UNUSED(ctx);
    अगर (!ggml_can_fuse(cgraph, गाँठ_idx, snake_pattern)) {
        वापस false;
    }

    स्थिर ggml_tensor * mul0     = cgraph->nodes[गाँठ_idx + 0];
    स्थिर ggml_tensor * sin_गाँठ = cgraph->nodes[गाँठ_idx + 1];
    स्थिर ggml_tensor * sqr      = cgraph->nodes[गाँठ_idx + 2];
    स्थिर ggml_tensor * mul1     = cgraph->nodes[गाँठ_idx + 3];
    स्थिर ggml_tensor * add      = cgraph->nodes[गाँठ_idx + 4];

    स्थिर ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
    स्थिर ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];

    स्थिर ggml_tensor * inv_b    = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];
    स्थिर ggml_tensor * x_in_add = (add->src[0] == mul1) ? add->src[1] : add->src[0];

    अगर (x_in_add != x) {
        वापस false;
    }
    अगर (x->type != GGML_TYPE_F32 && x->type != GGML_TYPE_F16 && x->type != GGML_TYPE_BF16) {
        वापस false;
    }
    // Shader bindings: data_a is A_TYPE so it follows x's precision, while
    // data_b and data_c are hardcoded float, so the broadcast operands must
    // be F32 regardless of x's type.
    अगर (a->type     != GGML_TYPE_F32) वापस false;
    अगर (inv_b->type != GGML_TYPE_F32) वापस false;
    // Chain intermediates and output share x's precision (single A_TYPE / D_TYPE pipeline).
    अगर (mul0->type     != x->type) वापस false;
    अगर (sin_गाँठ->type != x->type) वापस false;
    अगर (sqr->type      != x->type) वापस false;
    अगर (mul1->type     != x->type) वापस false;
    अगर (add->type      != x->type) वापस false;
    अगर (!ggml_are_same_shape(a, inv_b)) {
        वापस false;
    }
    अगर (a->ne[0] != 1 || a->ne[1] != x->ne[1]) {
        वापस false;
    }
    // Dispatch is 2D over (ne0, ne1), so x and add must be 2D and a / inv_b
    // must collapse to [1, C, 1, 1]. Higher dims are not handled by the shader.
    अगर (x->ne[2]     != 1 || x->ne[3]     != 1) वापस false;
    अगर (add->ne[2]   != 1 || add->ne[3]   != 1) वापस false;
    अगर (a->ne[2]     != 1 || a->ne[3]     != 1) वापस false;
    अगर (inv_b->ne[2] != 1 || inv_b->ne[3] != 1) वापस false;
    // Shader uses idx = i0 + i1 * ne0 and reads data_b[i1] / data_c[i1],
    // so every operand must be contiguous.
    अगर (!ggml_is_contiguous(x) || !ggml_is_contiguous(add) ||
        !ggml_is_contiguous(a) || !ggml_is_contiguous(inv_b)) {
        वापस false;
    }
    वापस true;
}

bool ggml_vk_tensors_overlap(स्थिर ggml_tensor * a, स्थिर ggml_tensor * b, bool elementwise) {
    ggml_backend_vk_बफर_context * a_buf_ctx = (ggml_backend_vk_बफर_context *)a->बफर->context;
    vk_बफर a_buf = a_buf_ctx->dev_बफर;
    ggml_backend_vk_बफर_context * b_buf_ctx = (ggml_backend_vk_बफर_context *)b->बफर->context;
    vk_बफर b_buf = b_buf_ctx->dev_बफर;
    अगर (a_buf == b_buf) {
        स्वतः a_base = vk_tensor_offset(a) + a->view_offs;
        स्वतः a_आकार = ggml_nbytes(a);
        स्वतः b_base = vk_tensor_offset(b) + b->view_offs;
        स्वतः b_आकार = ggml_nbytes(b);

        अगर (elementwise && a_base == b_base && a_आकार == b_आकार) {
            वापस false;
        }

        अगर ((b_base <= a_base && a_base < b_base + b_आकार) ||
            (a_base <= b_base && b_base < a_base + a_आकार)) {
            वापस true;
        }
    }
    वापस false;
}

bool ggml_vk_can_fuse_rms_norm_mul_rope(ggml_backend_vk_context * ctx, स्थिर काष्ठा ggml_cgraph * cgraph,
                                               पूर्णांक गाँठ_idx) {
    स्थिर ggml_tensor *rms = cgraph->nodes[गाँठ_idx + 0];
    स्थिर ggml_tensor *mul = cgraph->nodes[गाँठ_idx + 1];
    स्थिर ggml_tensor *rope = cgraph->nodes[गाँठ_idx + 2];

    स्थिर पूर्णांक mode = ((स्थिर int32_t *) rope->op_params)[2];

    // noncontig tensors aren't tested, and don't seem common in practice
    अगर (!ggml_is_contiguous(rms) ||
        !ggml_is_contiguous(mul) ||
        !ggml_is_contiguous(rope)) {
        वापस false;
    }

    // only norm/neox are handled in the shader
    अगर (mode != GGML_ROPE_TYPE_NEOX && mode != GGML_ROPE_TYPE_NORMAL) {
        वापस false;
    }

    // shared memory size for passing data from mul->rope
    अगर (mul->ne[0] > 1024) {
        वापस false;
    }

    // conditions for pipeline creation
    अगर (माप(vk_op_rms_norm_mul_rope_धकेलो_constants) > ctx->device->properties.limits.max_धकेलो_Constants_आकार) {
        वापस false;
    }

    वापस true;
}

uint32_t ggml_vk_fuse_multi_add(ggml_backend_vk_context * ctx, स्थिर काष्ठा ggml_cgraph * cgraph, पूर्णांक गाँठ_idx) {

    स्थिर ggml_tensor *first_गाँठ = cgraph->nodes[गाँठ_idx];
    अगर (first_गाँठ->op != GGML_OP_ADD) {
        वापस 0;
    }

    अगर (!ctx->device->multi_add) {
        वापस 0;
    }

    int32_t अंक_adds = 1;
    जबतक (गाँठ_idx + अंक_adds < cgraph->n_nodes &&
           cgraph->nodes[गाँठ_idx + अंक_adds]->op == GGML_OP_ADD &&
           अंक_adds < MAX_FUSED_ADDS) {
        अंक_adds++;
    }

    // The shader currently requires same shapes (but different strides are allowed),
    // everything f32, and no misalignment
    क्रम (int32_t i = 0; i < अंक_adds; ++i) {
        स्थिर ggml_tensor *अग्र_गाँठ = cgraph->nodes[गाँठ_idx + i];
        अगर (!ggml_are_same_shape(first_गाँठ, अग्र_गाँठ->src[0]) ||
            !ggml_are_same_shape(first_गाँठ, अग्र_गाँठ->src[1]) ||
            अग्र_गाँठ->type != GGML_TYPE_F32 ||
            अग्र_गाँठ->src[0]->type != GGML_TYPE_F32 ||
            अग्र_गाँठ->src[1]->type != GGML_TYPE_F32 ||
            get_misalign_bytes(ctx, अग्र_गाँठ) ||
            get_misalign_bytes(ctx, अग्र_गाँठ->src[0]) ||
            get_misalign_bytes(ctx, अग्र_गाँठ->src[1])) {
            अंक_adds = i;
        }
    }

    // Verify we can fuse these
    ggml_op adds[MAX_FUSED_ADDS];
    क्रम (int32_t i = 0; i < अंक_adds; ++i) {
        adds[i] = GGML_OP_ADD;
    }

    // decrease num_adds if they can't all be fused
    जबतक (अंक_adds > 1 && !ggml_can_fuse(cgraph, गाँठ_idx, adds, अंक_adds)) {
        अंक_adds--;
    }

    // a single add is not "fused", so just return zero
    अगर (अंक_adds == 1) {
        वापस 0;
    }
    वापस अंक_adds;
}

अटल int32_t find_first_set(uint32_t x) {
    int32_t ret = 0;
    अगर (!x) {
        वापस -1;
    }
    जबतक (!(x & 1)) {
        x >>= 1;
        ret++;
    }
    वापस ret;
}

अटल ggml_status ggml_backend_vk_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    VK_LOG_DEBUG("ggml_backend_vk_graph_compute(" << cgraph->n_nodes << " nodes)");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;

    ctx->device->diag_cgraph = nullptr;
    ctx->device->diag_पूर्व_start = -1;
    ctx->device->diag_पूर्व_end = -1;

    अगर (vk_instance.debug_utils_support) {
        ctx->device->debug_cmdbuf_idx = 0;
    }

    // queue scope, so it encloses every submit this evaluation makes.
    // closed when the function returns
    ggml_vk_debug_label queue_dbg(ctx->device->compute_queue->handle.लो(), "ggml_backend_vk_graph_compute");

    ctx->prealloc_आकार_add_rms_partials_offset = 0;
    ctx->do_add_rms_partials = false;
    ctx->do_add_rms_partials_offset_calculation = false;

    पूर्णांक last_गाँठ = cgraph->n_nodes - 1;

    // If the last op in the cgraph isn't backend GPU, the command buffer doesn't get closed properly
    जबतक (last_गाँठ > 0 && (ggml_vk_is_empty(cgraph->nodes[last_गाँठ]) || ((cgraph->nodes[last_गाँठ]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0))) {
        last_गाँठ -= 1;
    }

    // Reserve tensor context space for all nodes
    ctx->tensor_ctxs.resize(cgraph->n_nodes);

    bool first_गाँठ_in_batch = true; // true if next node will be first node in a batch
    पूर्णांक submit_गाँठ_idx = 0; // index to first node in a batch

    ggml_vk_submit_transfer_ctx(ctx);

    vk_context compute_ctx;
    अगर (vk_perf_logger_enabled) {
        // allocate/resize the query pool
        अगर (ctx->अंक_queries < cgraph->n_nodes + 1) {
            अगर (ctx->query_pool) {
                ctx->device->device.destroyQueryPool(ctx->query_pool);
            }
            vk::QueryPoolCreateInfo query_create_info;
            query_create_info.queryType = vk::QueryType::eTimestamp;
            query_create_info.query_गिनती = cgraph->n_nodes + 100;
            ctx->query_pool = ctx->device->device.createQueryPool(query_create_info);
            ctx->अंक_queries = query_create_info.query_गिनती;
            ctx->query_fusion_names.resize(ctx->अंक_queries);
            ctx->query_fusion_गाँठ_गिनती.resize(ctx->अंक_queries);
            ctx->query_nodes.resize(ctx->अंक_queries);
            ctx->query_गाँठ_idx.resize(ctx->अंक_queries);
        }

        ctx->device->device.resetQueryPool(ctx->query_pool, 0, cgraph->n_nodes+1);
        std::fill(ctx->query_fusion_names.begin(), ctx->query_fusion_names.end(), nullptr);
        std::fill(ctx->query_fusion_गाँठ_गिनती.begin(), ctx->query_fusion_गाँठ_गिनती.end(), 0);
        std::fill(ctx->query_nodes.begin(), ctx->query_nodes.end(), nullptr);
        std::fill(ctx->query_गाँठ_idx.begin(), ctx->query_गाँठ_idx.end(), 0);

        GGML_ASSERT(ctx->compute_ctx.expired());
        compute_ctx = ggml_vk_get_compute_ctx(ctx);
        ctx->query_idx = 0;
        compute_ctx->s->बफर->buf.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands, ctx->query_pool, ctx->query_idx++);
        ggml_vk_sync_buffers(ctx, compute_ctx);
    }

    ctx->prealloc_y_last_pipeline_used = nullptr;
    ctx->prealloc_y_last_tensor_used = nullptr;
    ctx->prealloc_y_last_k_padded = false;

    अगर (ctx->prealloc_आकार_add_rms_partials) {
        ggml_vk_preallocate_buffers(ctx, nullptr);
        compute_ctx = ggml_vk_get_compute_ctx(ctx);
        // initialize partial sums to zero.
        ggml_vk_बफर_memset_async(compute_ctx, ctx->prealloc_add_rms_partials, 0, 0, ctx->prealloc_आकार_add_rms_partials);
        ggml_vk_sync_buffers(ctx, compute_ctx);
    }

    // Submit after enough work has accumulated, to overlap CPU cmdbuffer generation with GPU execution.
    // Estimate the amount of compute work using flops, and submit every 200 GFLOP
    // (and scaled down based on total graph flops, so smaller models submit earlier).
    // Also submit at least every 100 nodes, in case there are workloads without heavy compute.
    uint32_t submitted_nodes = 0;
    uint32_t submit_गिनती = 0;
    uint64_t batch_flops = 0;
    uint64_t योग_flops = 0;
    uint64_t flops_cap = 200'000'000'000ULL;

    // On weaker AMD GPUs larger submissions can hit a driver timeout, submit more often to avoid this
    अगर (ctx->device->vendor_id == VK_VENDOR_ID_AMD && ctx->device->shader_core_गिनती > 0) {
        अगर (ctx->device->architecture == AMD_GCN && ctx->device->shader_core_गिनती < 32) {
            flops_cap = 500'000'000ULL * ctx->device->shader_core_गिनती;
        } अन्यथा अगर (ctx->device->architecture != AMD_GCN && ctx->device->shader_core_गिनती < 24) {
            flops_cap = 2'000'000'000ULL * ctx->device->shader_core_गिनती;
        }
    }
    uint64_t flops_per_submit = std::min(flops_cap, ctx->last_योग_flops / 40u);

    स्वतः स्थिर submit_after = [&](पूर्णांक start, पूर्णांक end) {
        अगर (ctx->device->serialize_submissions) {
            प्रयास {
                स्वतः res = ctx->device->device.waitForFences({ ctx->fence }, true, UINT64_MAX);
                अगर (res != vk::परिणाम_2::eSuccess) {
                    GGML_LOG_ERROR("ggml_vulkan: waitForFences error during serialized submission\n");
                    फेंको vk::SystemError(vk::make_error_code(res), "ggml_vulkan: waitForFences during serialized submission");
                }
            } पकड़ो (vk::DeviceLostError &) {
                ggml_vk_print_device_fault_info(ctx->device);
                GGML_LOG_ERROR("ggml_vulkan: device lost on %s waiting for submission (nodes %d to %d):\n",
                        ctx->device->name.c_str(), start, end);
                ggml_vk_print_गाँठ_list(cgraph, start, end);
                फेंको;
            }
            ctx->device->device.resetFences({ ctx->fence });
            ctx->submit_pending = false;
            ctx->device->diag_cgraph = cgraph;
            ctx->device->diag_पूर्व_start = start;
            ctx->device->diag_पूर्व_end = end;
        }
        first_गाँठ_in_batch = true;
        submitted_nodes = 0;
        batch_flops = 0;
        अगर (submit_गिनती < 3) {
            flops_per_submit *= 2;
        }
        submit_गिनती++;
    };

    क्रम (पूर्णांक i = 0; i < cgraph->n_nodes; i++) {
        अगर (first_गाँठ_in_batch) {
            submit_गाँठ_idx = i;
        }

        {
            स्वतः गाँठ_flops = ggml_vk_get_गाँठ_flops(cgraph->nodes[i]);
            योग_flops += गाँठ_flops;

            // Flush the current batch before recording a node that would push it over the flop threshold
            अगर (flops_per_submit != 0 && submitted_nodes > 0 && batch_flops + गाँठ_flops >= flops_per_submit) {
                vk_context flush_ctx = ggml_vk_get_compute_ctx(ctx);
                ggml_vk_ctx_end(flush_ctx);
                flush_ctx->exit_tensor_idx = -1;
                ctx->compute_ctx.reset();
                ggml_vk_compute_forward(ctx, cgraph, cgraph->nodes[submit_गाँठ_idx], submit_गाँठ_idx, false);
                submit_after(submit_गाँठ_idx, i - 1);
                submit_गाँठ_idx = i;
            }

            batch_flops += गाँठ_flops;
        }

        // op_srcs_fused_elementwise indicates whether an op's srcs all contribute to
        // the fused result in an elementwise-way. This affects whether the memory for
        // the src is allowed to overlap the memory for the destination.
        // The array is sized to handle the largest fusion (asserted later).
        bool op_srcs_fused_elementwise[13];

        ctx->fused_topk_moe_mode = TOPK_MOE_गिनती;
        ctx->fused_topk_moe_scale = false;
        ctx->fused_topk_qsa = false;
        ctx->fused_hc_post_gate = false;
        ctx->fused_rms_norm_mode = RMS_NORM_गिनती;
        स्थिर अक्षर *fusion_string {};
        अगर (!ctx->device->disable_fusion) {
            uint32_t अंक_adds = ggml_vk_fuse_multi_add(ctx, cgraph, i);
            अगर (अंक_adds) {
                ctx->अंक_additional_fused_ops = अंक_adds - 1;
                fusion_string = "MULTI_ADD";
                std::fill_n(op_srcs_fused_elementwise, ctx->अंक_additional_fused_ops + 1, true);
            } अन्यथा अगर (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_MUL_MAT, GGML_OP_ADD, GGML_OP_ADD })) {
                ctx->अंक_additional_fused_ops = 2;
                fusion_string = "MUL_MAT_ADD_ADD";
                op_srcs_fused_elementwise[0] = false;
                op_srcs_fused_elementwise[1] = true;
                op_srcs_fused_elementwise[2] = true;
            } अन्यथा अगर (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_MUL_MAT, GGML_OP_ADD })) {
                ctx->अंक_additional_fused_ops = 1;
                fusion_string = "MUL_MAT_ADD";
                op_srcs_fused_elementwise[0] = false;
                op_srcs_fused_elementwise[1] = true;
            } अन्यथा अगर (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_MUL_MAT_ID, GGML_OP_ADD_ID, GGML_OP_MUL })) {
                ctx->अंक_additional_fused_ops = 2;
                fusion_string = "MUL_MAT_ID_ADD_ID_MUL";
                op_srcs_fused_elementwise[0] = false;
                op_srcs_fused_elementwise[1] = true;
                op_srcs_fused_elementwise[2] = true;
            } अन्यथा अगर (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_MUL_MAT_ID, GGML_OP_ADD_ID })) {
                ctx->अंक_additional_fused_ops = 1;
                fusion_string = "MUL_MAT_ID_ADD_ID";
                op_srcs_fused_elementwise[0] = false;
                op_srcs_fused_elementwise[1] = true;
            } अन्यथा अगर (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_MUL_MAT_ID, GGML_OP_MUL })) {
                ctx->अंक_additional_fused_ops = 1;
                fusion_string = "MUL_MAT_ID_MUL";
                op_srcs_fused_elementwise[0] = false;
                op_srcs_fused_elementwise[1] = true;
            } अन्यथा अगर (ggml_can_fuse_subgraph(cgraph, i, rms_norm_mul_rope_view_set_rows_pattern, { i + 4 }) &&
                       ggml_check_edges(cgraph, i, rms_norm_mul_rope_view_set_rows_edges) &&
                       ggml_vk_can_fuse_rms_norm_mul_rope(ctx, cgraph, i) &&
                       ggml_vk_can_fuse_rope_set_rows(ctx, cgraph, i + 2)) {
                ctx->अंक_additional_fused_ops = 4;
                ctx->fused_rms_norm_mode = RMS_NORM_MUL_ROPE_VIEW_SET_ROWS;
                fusion_string = "RMS_NORM_MUL_ROPE_VIEW_SET_ROWS";
                op_srcs_fused_elementwise[0] = false;
                op_srcs_fused_elementwise[1] = false;
                op_srcs_fused_elementwise[2] = false;
                op_srcs_fused_elementwise[3] = false;
                op_srcs_fused_elementwise[4] = false;
            } अन्यथा अगर (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE }) &&
                       ggml_vk_can_fuse_rms_norm_mul_rope(ctx, cgraph, i)) {
                ctx->अंक_additional_fused_ops = 2;
                ctx->fused_rms_norm_mode = RMS_NORM_MUL_ROPE;
                fusion_string = "RMS_NORM_MUL_ROPE";
                // rope is approximately elementwise - whole rows are done by a single workgroup and it's row-wise
                op_srcs_fused_elementwise[0] = false;
                op_srcs_fused_elementwise[1] = true;
                op_srcs_fused_elementwise[2] = true;
            } अन्यथा अगर (ggml_can_fuse_subgraph(cgraph, i, hc_post_gate_pattern, { i + 3 }) &&
                       ggml_check_edges(cgraph, i, hc_post_gate_edges) &&
                       ggml_vk_can_fuse_hc_post_gate(cgraph, i)) {
                ctx->अंक_additional_fused_ops = hc_post_gate_pattern.size() - 1;
                ctx->fused_hc_post_gate = true;
                fusion_string = "HC_POST_GATE";
                std::fill_n(op_srcs_fused_elementwise, ctx->अंक_additional_fused_ops + 1, false);
            } अन्यथा अगर (ggml_vk_can_fuse(ctx, cgraph, i, rms_norm_mul_add_mul_pattern)) {
                ctx->अंक_additional_fused_ops = 3;
                ctx->fused_rms_norm_mode = RMS_NORM_MUL_ADD_MUL;
                fusion_string = "RMS_NORM_MUL_ADD_MUL";
                std::fill_n(op_srcs_fused_elementwise, 4, true);
            } अन्यथा अगर (ggml_vk_can_fuse(ctx, cgraph, i, rms_norm_mul_add_pattern)) {
                ctx->अंक_additional_fused_ops = 2;
                ctx->fused_rms_norm_mode = RMS_NORM_MUL_ADD;
                fusion_string = "RMS_NORM_MUL_ADD";
                std::fill_n(op_srcs_fused_elementwise, 3, true);
            } अन्यथा अगर (ggml_can_fuse_subgraph(cgraph, i, rms_norm_view_set_rows_pattern, { i + 2 }) &&
                       ggml_check_edges(cgraph, i, rms_norm_view_set_rows_edges) &&
                       ggml_vk_can_fuse_rms_norm_set_rows(ctx, cgraph, i)) {
                ctx->अंक_additional_fused_ops = 2;
                ctx->fused_rms_norm_mode = RMS_NORM_VIEW_SET_ROWS;
                fusion_string = "RMS_NORM_VIEW_SET_ROWS";
                std::fill_n(op_srcs_fused_elementwise, 3, false);
            } अन्यथा अगर (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
                ctx->अंक_additional_fused_ops = 1;
                ctx->fused_rms_norm_mode = RMS_NORM_MUL;
                fusion_string = "RMS_NORM_MUL";
                // rms_norm is not elementwise, but whole rows must be consumed and the scale factor computed before
                // they are overwritten, and one workgroup per row. So close enough.
                op_srcs_fused_elementwise[0] = true;
                op_srcs_fused_elementwise[1] = true;
            } अन्यथा अगर (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL })) {
                ctx->अंक_additional_fused_ops = 1;
                चयन (ggml_get_unary_op(cgraph->nodes[i])) {
                    हाल GGML_UNARY_OP_GELU:     fusion_string = "GELU_MUL";     अवरोध;
                    हाल GGML_UNARY_OP_SIGMOID:  fusion_string = "SIGMOID_MUL";  अवरोध;
                    हाल GGML_UNARY_OP_SILU:     fusion_string = "SILU_MUL";     अवरोध;
                    शेष:                     fusion_string = "SOFTPLUS_MUL"; अवरोध;
                }
                op_srcs_fused_elementwise[0] = true;
                op_srcs_fused_elementwise[1] = true;
            } अन्यथा अगर (ggml_vk_can_fuse_ssm_conv(ctx, cgraph, i, 2)) {
                ctx->अंक_additional_fused_ops = 2;
                fusion_string = "SSM_CONV_BIAS_SILU";
                // ssm_conv reads multiple input tokens per output, so it's not elementwise w.r.t. its srcs.
                // The downstream add and silu are elementwise on the conv output.
                op_srcs_fused_elementwise[0] = false;
                op_srcs_fused_elementwise[1] = true;
                op_srcs_fused_elementwise[2] = true;
            } अन्यथा अगर (ggml_vk_can_fuse_ssm_conv(ctx, cgraph, i, 1)) {
                ctx->अंक_additional_fused_ops = 1;
                fusion_string = "SSM_CONV_SILU";
                op_srcs_fused_elementwise[0] = false;
                op_srcs_fused_elementwise[1] = true;
            } अन्यथा अगर (ggml_can_fuse_subgraph(cgraph, i, rope_view_set_rows_pattern, { i + 2 }) &&
                       ggml_check_edges(cgraph, i, rope_view_set_rows_edges) &&
                       ggml_vk_can_fuse_rope_set_rows(ctx, cgraph, i)) {
                ctx->अंक_additional_fused_ops = 2;
                fusion_string = "ROPE_VIEW_SET_ROWS";
                op_srcs_fused_elementwise[0] = false;
                op_srcs_fused_elementwise[1] = false;
                op_srcs_fused_elementwise[2] = false;
            } अन्यथा अगर (ggml_vk_can_fuse_snake(ctx, cgraph, i)) {
                ctx->अंक_additional_fused_ops = 4;
                fusion_string = "SNAKE";
                // elementwise=true: snake.comp is safe under exact aliasing because each
                // thread reads data_x[idx] into a register before writing data_d[idx]
                // with a data dependency on that register. The overlap check still
                // rejects partial overlaps (different base or size).
                std::fill_n(op_srcs_fused_elementwise, 5, true);
            } अन्यथा अगर (ggml_vk_can_fuse_topk_qsa(ctx, cgraph, i)) {
                ctx->अंक_additional_fused_ops = topk_qsa_pattern.size() - 1;
                ctx->fused_topk_qsa = true;
                fusion_string = "TOPK_QSA";
                std::fill_n(op_srcs_fused_elementwise, ctx->अंक_additional_fused_ops + 1, false);
            } अन्यथा अगर (ggml_can_fuse_subgraph(cgraph, i, topk_moe_early_softmax_norm, { i + 3, i + 9 }) &&
                       ggml_check_edges(cgraph, i, topk_moe_early_softmax_norm_edges) &&
                       ggml_vk_can_fuse_topk_moe(ctx, cgraph, i, TOPK_MOE_EARLY_SOFTMAX_NORM)) {
                ctx->अंक_additional_fused_ops = topk_moe_early_softmax_norm.size() - 1;
                // view of argsort writes to memory
                ctx->fused_ops_write_mask |= 1 << 3;
                ctx->fused_topk_moe_mode = TOPK_MOE_EARLY_SOFTMAX_NORM;
                fusion_string = "TOPK_MOE_EARLY_SOFTMAX_NORM";
                std::fill_n(op_srcs_fused_elementwise, ctx->अंक_additional_fused_ops + 1, false);
            } अन्यथा अगर (ggml_can_fuse_subgraph(cgraph, i, topk_moe_sigmoid_norm_bias, { i + 4, i + 10 }) &&
                       ggml_check_edges(cgraph, i, topk_moe_sigmoid_norm_bias_edges) &&
                       ggml_vk_can_fuse_topk_moe(ctx, cgraph, i, TOPK_MOE_SIGMOID_NORM_BIAS)) {
                ctx->अंक_additional_fused_ops = topk_moe_sigmoid_norm_bias.size() - 1;
                // view of argsort writes to memory
                ctx->fused_ops_write_mask |= 1 << 4;
                ctx->fused_topk_moe_mode = TOPK_MOE_SIGMOID_NORM_BIAS;
                fusion_string = "TOPK_MOE_SIGMOID_NORM_BIAS";
                std::fill_n(op_srcs_fused_elementwise, ctx->अंक_additional_fused_ops + 1, false);
            } अन्यथा अगर (ggml_can_fuse_subgraph(cgraph, i, topk_moe_sqrt_softplus_norm_bias, { i + 5, i + 11 }) &&
                       ggml_check_edges(cgraph, i, topk_moe_sqrt_softplus_norm_bias_edges) &&
                       ggml_vk_can_fuse_topk_moe(ctx, cgraph, i, TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS)) {
                ctx->अंक_additional_fused_ops = topk_moe_sqrt_softplus_norm_bias.size() - 1;
                // view of argsort writes to memory
                ctx->fused_ops_write_mask |= 1 << 5;
                ctx->fused_topk_moe_mode = TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS;
                fusion_string = "TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS";
                std::fill_n(op_srcs_fused_elementwise, ctx->अंक_additional_fused_ops + 1, false);
            } अन्यथा अगर (ggml_can_fuse_subgraph(cgraph, i, topk_moe_early_softmax, { i + 3, i + 4 }) &&
                       ggml_check_edges(cgraph, i, topk_moe_early_softmax_edges) &&
                       ggml_vk_can_fuse_topk_moe(ctx, cgraph, i, TOPK_MOE_EARLY_SOFTMAX)) {
                ctx->अंक_additional_fused_ops = topk_moe_early_softmax.size() - 1;
                // view of argsort writes to memory
                ctx->fused_ops_write_mask |= 1 << 3;
                ctx->fused_topk_moe_mode = TOPK_MOE_EARLY_SOFTMAX;
                fusion_string = "TOPK_MOE_EARLY_SOFTMAX";
                std::fill_n(op_srcs_fused_elementwise, ctx->अंक_additional_fused_ops + 1, false);
            } अन्यथा अगर (ggml_can_fuse_subgraph(cgraph, i, topk_moe_late_softmax, { i + 1, i + 5 }) &&
                       ggml_check_edges(cgraph, i, topk_moe_late_softmax_edges) &&
                       ggml_vk_can_fuse_topk_moe(ctx, cgraph, i, TOPK_MOE_LATE_SOFTMAX)) {
                ctx->अंक_additional_fused_ops = topk_moe_late_softmax.size() - 1;
                // view of argsort writes to memory
                ctx->fused_ops_write_mask |= 1 << 1;
                ctx->fused_topk_moe_mode = TOPK_MOE_LATE_SOFTMAX;
                fusion_string = "TOPK_MOE_LATE_SOFTMAX";
                std::fill_n(op_srcs_fused_elementwise, ctx->अंक_additional_fused_ops + 1, false);
            }
            अगर (ctx->fused_topk_moe_mode != TOPK_MOE_गिनती) {
                // Look for an additional scale op to fuse - occurs in deepseek2 and nemotron3 nano.
                अगर (ggml_can_fuse_subgraph(cgraph, i + ctx->अंक_additional_fused_ops - 1, { GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE }, { i + ctx->अंक_additional_fused_ops + 1 }) ||
                    ggml_can_fuse_subgraph(cgraph, i + ctx->अंक_additional_fused_ops, { GGML_OP_GET_ROWS, GGML_OP_SCALE }, { i + ctx->अंक_additional_fused_ops + 1 })) {
                    ctx->fused_topk_moe_scale = true;
                    ctx->अंक_additional_fused_ops++;
                    op_srcs_fused_elementwise[ctx->अंक_additional_fused_ops] = false;
                }
            }
        }
        GGML_ASSERT(ctx->अंक_additional_fused_ops < (पूर्णांक)(माप(op_srcs_fused_elementwise) / माप(op_srcs_fused_elementwise[0])));
        ctx->fused_ops_write_mask |= 1 << ctx->अंक_additional_fused_ops;

        // Check whether fusion would overwrite src operands while they're still in use.
        // If so, disable fusion.
        अगर (ctx->अंक_additional_fused_ops) {
            // There are up to two output nodes - topk_moe has two.
            uint32_t bits = ctx->fused_ops_write_mask & ~(1 << ctx->अंक_additional_fused_ops);
            ggml_tensor *output_nodes[2] {};
            output_nodes[0] = cgraph->nodes[i + ctx->अंक_additional_fused_ops];
            अगर (bits) {
                पूर्णांक output_idx = find_first_set(bits);
                GGML_ASSERT(bits == (1u << output_idx));
                output_nodes[1] = cgraph->nodes[i + output_idx];
            }

            bool need_disable = false;

            क्रम (पूर्णांक j = 0; j < 2; ++j) {
                ggml_tensor *dst = output_nodes[j];
                अगर (!dst) {
                    जारी;
                }
                // Loop over all srcs of all nodes in the fusion. If the src overlaps
                // the destination and the src is not an intermediate node that's being
                // elided, then disable fusion.
                क्रम (पूर्णांक k = 0; k <= ctx->अंक_additional_fused_ops; ++k) {
                    क्रम (uint32_t s = 0; s < GGML_MAX_SRC; ++s) {
                        ggml_tensor *src = cgraph->nodes[i + k]->src[s];
                        अगर (!src || src->op == GGML_OP_NONE) {
                            जारी;
                        }
                        अगर (ggml_vk_tensors_overlap(src, dst, op_srcs_fused_elementwise[k])) {
                            bool found = false;
                            क्रम (पूर्णांक n = 0; n < k; ++n) {
                                अगर (cgraph->nodes[i + n] == src) {
                                    found = true;
                                    अवरोध;
                                }
                            }
                            अगर (!found) {
                                need_disable = true;
                            }
                        }
                    }
                }
            }
            अगर (need_disable) {
                ctx->अंक_additional_fused_ops = 0;
                ctx->fused_ops_write_mask = 1;
                ctx->fused_topk_moe_mode = TOPK_MOE_गिनती;
                ctx->fused_topk_moe_scale = false;
                ctx->fused_topk_qsa = false;
                ctx->fused_hc_post_gate = false;
                ctx->fused_rms_norm_mode = RMS_NORM_गिनती;
                fusion_string = nullptr;
            }
        }

        // Signal the almost_ready fence when the graph is mostly complete (< 20% remaining)
        bool almost_ready = (cgraph->n_nodes - i) < cgraph->n_nodes / 5;
        bool submit = (submitted_nodes >= ctx->device->max_nodes_per_submit) ||
                      (flops_per_submit != 0 && batch_flops >= flops_per_submit) ||
                      (i + ctx->अंक_additional_fused_ops >= last_गाँठ) ||
                      (almost_ready && !ctx->almost_ready_fence_pending);

        bool enqueued = ggml_vk_build_graph(ctx, cgraph, i, cgraph->nodes[submit_गाँठ_idx], submit_गाँठ_idx, i + ctx->अंक_additional_fused_ops >= last_गाँठ, almost_ready, submit);

        अगर (vk_perf_logger_enabled && enqueued) {
            compute_ctx = ggml_vk_get_compute_ctx(ctx);
            अगर (!vk_perf_logger_concurrent) {
                // track a single node/fusion for the current query
                ctx->query_nodes[ctx->query_idx] = cgraph->nodes[i];
                ctx->query_fusion_names[ctx->query_idx] = fusion_string;
                compute_ctx->s->बफर->buf.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands, ctx->query_pool, ctx->query_idx++);
                ggml_vk_sync_buffers(ctx, compute_ctx);
            } अन्यथा {
                // track a fusion string and number of fused ops for the current node_idx
                ctx->query_fusion_names[i] = fusion_string;
                ctx->query_fusion_गाँठ_गिनती[i] = ctx->अंक_additional_fused_ops;
            }
        }

        अगर (enqueued) {
            ++submitted_nodes;

#अगर_अघोषित GGML_VULKAN_CHECK_RESULTS
            अगर (first_गाँठ_in_batch) {
                first_गाँठ_in_batch = false;
            }
#पूर्ण_अगर
        }

        अगर (submit && enqueued) {
            submit_after(submit_गाँठ_idx, i + (पूर्णांक)ctx->अंक_additional_fused_ops);
        }
        i += ctx->अंक_additional_fused_ops;
        ctx->अंक_additional_fused_ops = 0;
        ctx->fused_ops_write_mask = 0;
    }

    ctx->last_योग_flops = योग_flops;

    अगर (vk_perf_logger_enabled) {
        // End the command buffer and submit/wait
        GGML_ASSERT(!ctx->compute_ctx.expired());
        compute_ctx = ctx->compute_ctx.lock();
        ggml_vk_ctx_end(compute_ctx);

        ggml_vk_submit(compute_ctx, ctx->device->fence);
        VK_CHECK(ctx->device->device.waitForFences({ ctx->device->fence }, true, UINT64_MAX), "GGML_VULKAN_PERF waitForFences", ctx->device);
        ctx->device->device.resetFences({ ctx->device->fence });
        ctx->compute_ctx.reset();

        // Get the results and pass them to the logger
        std::vector<uint64_t> timestamps(cgraph->n_nodes + 1);
        VK_CHECK(ctx->device->device.getQueryPoolResults(ctx->query_pool, 0, ctx->query_idx, (cgraph->n_nodes + 1)*माप(uint64_t), timestamps.तथ्य(), माप(uint64_t), vk::Query_परिणाम_Flag_Bits::e64 | vk::Query_परिणाम_Flag_Bits::eWait), "get timestamp results", ctx->device);
        अगर (!vk_perf_logger_concurrent) {
            // Log each op separately
            क्रम (पूर्णांक i = 1; i < ctx->query_idx; i++) {
                स्वतः गाँठ = ctx->query_nodes[i];
                स्वतः name = ctx->query_fusion_names[i];
                ctx->perf_logger->log_timing(गाँठ, name, uint64_t((timestamps[i] - timestamps[i-1]) * ctx->device->properties.limits.timestampPeriod));
            }
        } अन्यथा {
            // Log each group of nodes
            पूर्णांक पूर्व_गाँठ_idx = 0;
            क्रम (पूर्णांक i = 1; i < ctx->query_idx; i++) {
                स्वतः cur_गाँठ_idx = ctx->query_गाँठ_idx[i];
                std::vector<ggml_tensor *> nodes;
                std::vector<स्थिर अक्षर *> names;
                क्रम (पूर्णांक गाँठ_idx = पूर्व_गाँठ_idx; गाँठ_idx < cur_गाँठ_idx; ++गाँठ_idx) {
                    अगर (ggml_op_is_empty(cgraph->nodes[गाँठ_idx]->op)) {
                        जारी;
                    }
                    nodes.push_back(cgraph->nodes[गाँठ_idx]);
                    names.push_back(ctx->query_fusion_names[गाँठ_idx]);
                    गाँठ_idx += ctx->query_fusion_गाँठ_गिनती[गाँठ_idx];
                }
                पूर्व_गाँठ_idx = cur_गाँठ_idx;
                ctx->perf_logger->log_timing(nodes, names, uint64_t((timestamps[i] - timestamps[i-1]) * ctx->device->properties.limits.timestampPeriod));
            }
        }
        ctx->perf_logger->print_timings();
    }

    अगर (!ctx->device->support_async) {
        ggml_vk_synchronize(ctx);
    }

    वापस GGML_STATUS_SUCCESS;

    UNUSED(backend);
}

व्योम ggml_vk_graph_optimize(ggml_backend_t backend, काष्ठा ggml_cgraph * graph, काष्ठा ggml_backend_graph_optimize_params * params)
{
    VK_LOG_DEBUG("ggml_vk_graph_optimize(" << graph->n_nodes << " nodes)");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;

    अगर (ctx->device->disable_graph_optimize) {
        वापस;
    }

    स्वतः स्थिर &is_empty = [](स्थिर ggml_tensor * गाँठ) -> bool {
        वापस गाँठ->op == GGML_OP_NONE || गाँठ->op == GGML_OP_RESHAPE || गाँठ->op == GGML_OP_TRANSPOSE || गाँठ->op == GGML_OP_VIEW || गाँठ->op == GGML_OP_PERMUTE;
    };

    स्वतः स्थिर &is_src_of = [&is_empty](स्थिर ggml_tensor *dst, स्थिर ggml_tensor *src) -> bool {
        स्वतः स्थिर &base = [](स्थिर ggml_tensor * tensor) {
            वापस tensor->view_src ? tensor->view_src : tensor;
        };
        क्रम (uint32_t s = 0; s < GGML_MAX_SRC; ++s) {
            अगर (dst->src[s] == src) {
                वापस true;
            }
            अगर (is_empty(dst) || is_empty(src)) {
                जारी;
            }
            // A source view of dst may read storage written through a different view by src.
            अगर (dst->src[s] && base(dst->src[s]) == base(src)) {
                वापस true;
            }
            // Moving dst forward may overwrite storage still read through a view by src.
            अगर (src->src[s] && base(dst) == base(src->src[s])) {
                वापस true;
            }
        }
        // implicit dependency if they view the same tensor
        अगर (base(dst) == base(src)) {
            वापस true;
        }
        वापस false;
    };

    std::vector<ggml_tensor *> new_order;
    std::vector<bool> used(graph->n_nodes, false);
    std::set<ggml_tensor *> used_गाँठ_set;

    पूर्णांक first_unused = 0;

    // scheduled or zero-compute nodes in [lo, hi)
    स्वतः स्थिर &empty_or_scheduled_between = [&](पूर्णांक lo, पूर्णांक hi) -> bool {
        क्रम (पूर्णांक v = lo; v < hi; ++v) {
            अगर (!used[v] && !is_empty(graph->nodes[v])) {
                वापस false;
            }
        }
        वापस true;
    };
    जबतक (first_unused < graph->n_nodes) {
        std::vector<पूर्णांक> current_set;

        // Check for fusion patterns and avoid reordering them
        स्वतः स्थिर &match_pattern = [&](स्थिर std::initializer_list<ggml_op> &pattern, पूर्णांक start) -> bool {
            अगर (start + (पूर्णांक)pattern.size() <= graph->n_nodes) {
                bool is_pattern = true;
                क्रम (माप_प्रकार j = 0; j < pattern.size(); ++j) {
                    अगर (graph->nodes[start + j]->op != pattern.begin()[j] || used[start + j]) {
                        is_pattern = false;
                    }
                }
                वापस is_pattern;
            }
            वापस false;
        };

        स्वतः स्थिर &keep_pattern = [&](स्थिर std::initializer_list<ggml_op> &pattern) -> bool {
            अगर (match_pattern(pattern, first_unused)) {
                क्रम (माप_प्रकार j = 0; j < pattern.size(); ++j) {
                    new_order.push_back(graph->nodes[first_unused + j]);
                    used_गाँठ_set.insert(graph->nodes[first_unused + j]);
                    used[first_unused + j] = true;
                }
                जबतक (first_unused < graph->n_nodes && used[first_unused]) {
                    first_unused++;
                }
                वापस true;
            }
            वापस false;
        };

        स्वतः स्थिर &add_pattern_alloc_deps = [&](स्थिर std::initializer_list<ggml_op> &pattern, पूर्णांक last_गाँठ) {
            // Keep external inputs alive through the fused output.
            std::set<ggml_tensor *> seen;
            क्रम (माप_प्रकार j = 0; j < pattern.size(); ++j) {
                ggml_tensor * गाँठ = graph->nodes[first_unused + j];
                क्रम (uint32_t s = 0; s < GGML_MAX_SRC; ++s) {
                    ggml_tensor * src = गाँठ->src[s];
                    अगर (src && seen.insert(src).second) {
                        params->add_alloc_dep(params->user_तथ्य, src, graph->nodes[last_गाँठ]);
                    }
                }
                seen.insert(गाँठ);
            }
        };

        स्वतः स्थिर &keep_topk_moe_pattern = [&](स्थिर std::initializer_list<ggml_op> &pattern) -> bool {
            अगर (!match_pattern(pattern, first_unused)) {
                वापस false;
            }

            पूर्णांक last_गाँठ = first_unused + (पूर्णांक) pattern.size() - 1;
            // Some TOPK_MOE variants fuse a trailing scale.
            अगर (last_गाँठ + 1 < graph->n_nodes && graph->nodes[last_गाँठ + 1]->op == GGML_OP_SCALE) {
                last_गाँठ++;
            }

            add_pattern_alloc_deps(pattern, last_गाँठ);

            वापस keep_pattern(pattern);
        };

        अगर (keep_topk_moe_pattern(topk_moe_early_softmax_norm)) {
            जारी;
        }
        अगर (keep_topk_moe_pattern(topk_moe_sigmoid_norm_bias)) {
            जारी;
        }
        अगर (keep_topk_moe_pattern(topk_moe_sqrt_softplus_norm_bias)) {
            जारी;
        }
        अगर (keep_topk_moe_pattern(topk_moe_early_softmax)) {
            जारी;
        }
        अगर (keep_topk_moe_pattern(topk_moe_late_softmax)) {
            जारी;
        }
        अगर (keep_pattern(snake_pattern)) {
            जारी;
        }
        अगर (keep_pattern(topk_qsa_pattern)) {
            जारी;
        }

        अगर (keep_pattern(rms_norm_mul_add_mul_pattern)) {
            जारी;
        }
        अगर (keep_pattern(rms_norm_mul_add_pattern)) {
            जारी;
        }
        अगर (keep_pattern(rms_norm_mul_rope_view_set_rows_pattern)) {
            जारी;
        }
        अगर (keep_pattern(rms_norm_view_set_rows_pattern)) {
            जारी;
        }
        अगर (keep_pattern(rope_view_set_rows_pattern)) {
            जारी;
        }
        अगर (match_pattern(hc_post_gate_pattern, first_unused)) {
            add_pattern_alloc_deps(hc_post_gate_pattern, first_unused + (पूर्णांक) hc_post_gate_pattern.size() - 1);
            keep_pattern(hc_post_gate_pattern);
            जारी;
        }

        // First, grab the next unused node.
        current_set.push_back(first_unused);

        // Loop through the next N nodes. Grab any that don't depend on other nodes that
        // haven't already been run. Nodes that have already been run have used[i] set
        // to true. Allow nodes that depend on the previous node if it's a fusion pattern
        // that we support (e.g. RMS_NORM + MUL).
        // This first pass only grabs "real" (non-view nodes). Second pass grabs view nodes.
        // The goal is to not interleave real and view nodes in a way that breaks fusion.
        स्थिर पूर्णांक अंक_TO_CHECK = 20;
        क्रम (पूर्णांक j = first_unused+1; j < std::min(first_unused + अंक_TO_CHECK, graph->n_nodes); ++j) {
            अगर (used[j]) {
                जारी;
            }
            अगर (is_empty(graph->nodes[j])) {
                जारी;
            }
            // Protect every interior QSA node (not just the start): the mask branch is
            // independent, so it gets pulled out and breaks keep_pattern otherwise.
            स्वतः स्थिर &in_qsa_pattern = [&](पूर्णांक n) -> bool {
                क्रम (पूर्णांक o = 0; o < (पूर्णांक) topk_qsa_pattern.size(); ++o) {
                    अगर (n - o >= 0 && match_pattern(topk_qsa_pattern, n - o)) {
                        वापस true;
                    }
                }
                वापस false;
            };
            अगर (match_pattern(topk_moe_early_softmax_norm, j) ||
                match_pattern(topk_moe_sigmoid_norm_bias, j) ||
                match_pattern(topk_moe_sqrt_softplus_norm_bias, j) ||
                match_pattern(topk_moe_early_softmax, j) ||
                match_pattern(topk_moe_late_softmax, j) ||
                match_pattern(snake_pattern, j) ||
                in_qsa_pattern(j) ||
                match_pattern(rms_norm_mul_add_mul_pattern, j) ||
                match_pattern(rms_norm_mul_add_pattern, j) ||
                match_pattern(rms_norm_mul_rope_view_set_rows_pattern, j) ||
                match_pattern(rms_norm_view_set_rows_pattern, j) ||
                match_pattern(rope_view_set_rows_pattern, j) ||
                match_pattern(hc_post_gate_pattern, j)) {
                जारी;
            }
            bool ok = true;
            क्रम (पूर्णांक c = first_unused; c < j; ++c) {
                अगर (!used[c] &&
                    is_src_of(graph->nodes[j], graph->nodes[c]) &&
                    !(c == current_set.back() && graph->nodes[c]->op == GGML_OP_RMS_NORM && graph->nodes[j]->op == GGML_OP_MUL && empty_or_scheduled_between(c+1, j)) &&
                    !(c == current_set.back() && graph->nodes[c]->op == GGML_OP_UNARY && graph->nodes[j]->op == GGML_OP_MUL && empty_or_scheduled_between(c+1, j)) &&
                    !(j == c+1 && c == current_set.back() && graph->nodes[c]->op == GGML_OP_MUL_MAT && graph->nodes[j]->op == GGML_OP_ADD) &&
                    !(j == c+1 && c == current_set.back() && graph->nodes[c]->op == GGML_OP_MUL_MAT_ID && graph->nodes[j]->op == GGML_OP_ADD_ID) &&
                    !(j == c+1 && c == current_set.back() && graph->nodes[c]->op == GGML_OP_MUL_MAT_ID && graph->nodes[j]->op == GGML_OP_MUL) &&
                    !(j == c+1 && c == current_set.back() && graph->nodes[c]->op == GGML_OP_ADD && graph->nodes[j]->op == GGML_OP_ADD) &&
                    !(j == c+1 && c == current_set.back() && graph->nodes[c]->op == GGML_OP_SSM_CONV && graph->nodes[j]->op == GGML_OP_ADD) &&
                    !(j == c+1 && c == current_set.back() && graph->nodes[c]->op == GGML_OP_SSM_CONV && graph->nodes[j]->op == GGML_OP_UNARY)) {
                    ok = false;
                    अवरोध;
                }
            }
            अगर (ok) {
                current_set.push_back(j);

                पूर्णांक rope_idx = j;

                // When we've found RMS_NORM + MUL, try to find a ROPE that uses it
                अगर (j > 0 &&
                    graph->nodes[j]->op == GGML_OP_MUL &&
                    graph->nodes[j-1]->op == GGML_OP_RMS_NORM) {
                    क्रम (पूर्णांक k = j + 1; k < std::min(j + 15, graph->n_nodes); ++k) {
                        अगर (graph->nodes[k]->op == GGML_OP_ROPE &&
                            graph->nodes[k]->src[0] == graph->nodes[j] &&
                            // Check that other srcs are already valid
                            graph->nodes[k]->src[1]->op == GGML_OP_NONE &&
                            (graph->nodes[k]->src[2] == nullptr || graph->nodes[k]->src[2]->op == GGML_OP_NONE)) {
                            rope_idx = k;
                            current_set.push_back(rope_idx);
                            used[rope_idx] = true;
                            अवरोध;
                        }
                    }
                }
                // Look for ROPE/RMS_NORM + VIEW + SET_ROWS and make them consecutive
                अगर (graph->nodes[rope_idx]->op == GGML_OP_ROPE || graph->nodes[rope_idx]->op == GGML_OP_RMS_NORM) {
                    पूर्णांक view_idx = -1;
                    पूर्णांक set_rows_idx = -1;
                    क्रम (पूर्णांक k = rope_idx + 1; k < std::min(rope_idx + 15, graph->n_nodes); ++k) {
                        अगर (used[k]) {
                            जारी;
                        }
                        अगर (view_idx == -1 && graph->nodes[k]->op == GGML_OP_VIEW && graph->nodes[k]->src[0] == graph->nodes[rope_idx]) {
                            view_idx = k;
                            जारी;
                        }
                        अगर (view_idx != -1 && graph->nodes[k]->op == GGML_OP_SET_ROWS && graph->nodes[k]->src[0] == graph->nodes[view_idx]) {
                            set_rows_idx = k;
                            अवरोध;
                        }
                    }
                    अगर (set_rows_idx != -1) {
                        स्थिर पूर्णांक गाँठ_idxs[] = { rope_idx, view_idx, set_rows_idx };
                        स्थिर ggml_op ops[] = { graph->nodes[rope_idx]->op, GGML_OP_VIEW, GGML_OP_SET_ROWS };
                        bool can_pull = ggml_can_fuse_subgraph_ext(graph, गाँठ_idxs, 3, ops, &set_rows_idx, 1);

                        क्रम (पूर्णांक c = rope_idx + 1; can_pull && c < set_rows_idx; ++c) {
                            अगर (!used[c] && c != view_idx && !is_empty(graph->nodes[c]) &&
                                is_src_of(graph->nodes[set_rows_idx], graph->nodes[c])) {
                                can_pull = false;
                            }
                        }

                        अगर (can_pull) {
                            current_set.push_back(view_idx);
                            current_set.push_back(set_rows_idx);
                            used[view_idx] = true;
                            used[set_rows_idx] = true;
                        }
                    }
                }
                // Look for MUL_MAT_ID + ADD_ID + MUL
                अगर (j > 0 &&
                    graph->nodes[j]->op == GGML_OP_ADD_ID &&
                    graph->nodes[j-1]->op == GGML_OP_MUL_MAT_ID) {
                    क्रम (पूर्णांक k = j + 1; k < std::min(j + 15, graph->n_nodes); ++k) {
                        अगर (graph->nodes[k]->op == GGML_OP_MUL &&
                            graph->nodes[k]->src[0] == graph->nodes[j] &&
                            // src1 must either be weights or already processed
                            (graph->nodes[k]->src[1]->op == GGML_OP_NONE || used_गाँठ_set.find(graph->nodes[k]->src[1]) != used_गाँठ_set.end())) {
                            current_set.push_back(k);
                            used[k] = true;
                            अवरोध;
                        }
                    }
                }
                // Look for MUL_MAT + ADD + ADD
                अगर (j > 0 &&
                    graph->nodes[j]->op == GGML_OP_ADD &&
                    graph->nodes[j-1]->op == GGML_OP_MUL_MAT) {
                    क्रम (पूर्णांक k = j + 1; k < std::min(j + 15, graph->n_nodes); ++k) {
                        अगर (graph->nodes[k]->op == GGML_OP_ADD &&
                            graph->nodes[k]->src[0] == graph->nodes[j] &&
                            // src1 must either be weights or already processed
                            (graph->nodes[k]->src[1]->op == GGML_OP_NONE || used_गाँठ_set.find(graph->nodes[k]->src[1]) != used_गाँठ_set.end())) {
                            current_set.push_back(k);
                            used[k] = true;
                            अवरोध;
                        }
                    }
                }
                // SSM_CONV + ADD + UNARY: pull the consuming UNARY forward
                अगर (j > 0 &&
                    graph->nodes[j]->op == GGML_OP_ADD &&
                    graph->nodes[j-1]->op == GGML_OP_SSM_CONV) {
                    क्रम (पूर्णांक k = j + 1; k < std::min(j + 15, graph->n_nodes); ++k) {
                        अगर (graph->nodes[k]->op == GGML_OP_UNARY &&
                            graph->nodes[k]->src[0] == graph->nodes[j]) {
                            current_set.push_back(k);
                            used[k] = true;
                            अवरोध;
                        }
                    }
                }
                // UNARY + MUL: pull the consuming MUL forward
                अगर (j > 0 &&
                    graph->nodes[j]->op == GGML_OP_UNARY) {
                    क्रम (पूर्णांक k = j + 1; k < std::min(j + 15, graph->n_nodes); ++k) {
                        ggml_tensor * mul = graph->nodes[k];
                        अगर (mul->op != GGML_OP_MUL || (mul->src[0] != graph->nodes[j] && mul->src[1] != graph->nodes[j])) {
                            जारी;
                        }
                        ggml_tensor * other = (mul->src[0] == graph->nodes[j]) ? mul->src[1] : mul->src[0];
                        // the other src must either be weights or already processed
                        अगर (!(other->op == GGML_OP_NONE || used_गाँठ_set.find(other) != used_गाँठ_set.end())) {
                            जारी;
                        }
                        अगर (!ggml_vk_can_fuse_unary_mul(graph, j, k)) {
                            जारी;
                        }
                        current_set.push_back(k);
                        used[k] = true;
                        अवरोध;
                    }
                }
            }
        }
        // Second pass grabs view nodes.
        // Skip this if it would break a fusion optimization (don't split up add->rms_norm or add->add).
        अगर (graph->nodes[current_set.back()]->op != GGML_OP_ADD) {
            क्रम (पूर्णांक j = first_unused+1; j < std::min(first_unused + अंक_TO_CHECK, graph->n_nodes); ++j) {
                अगर (used[j]) {
                    जारी;
                }
                अगर (!is_empty(graph->nodes[j])) {
                    जारी;
                }
                bool ok = true;
                क्रम (पूर्णांक c = first_unused; c < j; ++c) {
                    bool c_in_current_set = std::find(current_set.begin(), current_set.end(), c) != current_set.end();
                    // skip views whose srcs haven't been processed.
                    अगर (!used[c] &&
                        is_src_of(graph->nodes[j], graph->nodes[c]) &&
                        !c_in_current_set) {
                        ok = false;
                        अवरोध;
                    }
                }
                अगर (ok) {
                    current_set.push_back(j);
                }
            }
        }

        // Push the current set into new_order
        क्रम (स्वतः c : current_set) {
            new_order.push_back(graph->nodes[c]);
            used_गाँठ_set.insert(graph->nodes[c]);
            used[c] = true;
        }
        जबतक (first_unused < graph->n_nodes && used[first_unused]) {
            first_unused++;
        }
    }
    // Replace the graph with the new order.
    क्रम (पूर्णांक i = 0; i < graph->n_nodes; ++i) {
        graph->nodes[i] = new_order[i];
    }
}

अटल व्योम ggml_backend_vk_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    VK_LOG_DEBUG("ggml_backend_vk_event_record(backend=" << backend << ", event=" << event << ")");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    vk_event *vkev = (vk_event *)event->context;

    ggml_vk_submit_transfer_ctx(ctx);

    vk_context compute_ctx = ggml_vk_get_compute_ctx(ctx);
    स्वतः* cmd_buf = compute_ctx->s->बफर; // retrieve pointer before it gets reset

    अगर (vkev->has_event) {
        // Move existing event into submitted
        vkev->events_submitted.push_back(vkev->event);
    }

    // Grab the next event and record it, create one if necessary
    अगर (vkev->events_free.empty()) {
        vkev->event = ctx->device->device.createEvent({});
    } अन्यथा {
        vkev->event = vkev->events_free.back();
        vkev->events_free.pop_back();
    }

    vkev->has_event = true;

    ggml_vk_set_event(compute_ctx, vkev->event);

    vkev->tl_semaphore.value++;
    compute_ctx->s->signal_semaphores.push_back(vkev->tl_semaphore);
    ggml_vk_ctx_end(compute_ctx);

    ggml_vk_submit(compute_ctx, {});
    ctx->submit_pending = true;
    vkev->cmd_बफर = cmd_buf;
    vkev->cmd_बफर_use_गणक = cmd_buf->use_गणक;
    ctx->compute_ctx.reset();
}

अटल व्योम ggml_backend_vk_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    VK_LOG_DEBUG("ggml_backend_vk_event_wait(backend=" << backend << ", event=" << event << ")");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    vk_event *vkev = (vk_event *)event->context;

    vk_context compute_ctx = ggml_vk_get_compute_ctx(ctx);

    अगर (vkev->has_event) {
        // Wait for latest event
        ggml_vk_wait_events(compute_ctx, { vkev->event });

        अगर (ctx->device->async_use_transfer_queue) {
            vk_context transfer_ctx = ggml_vk_get_transfer_ctx(ctx);
            transfer_ctx->s->wait_semaphores.push_back(vkev->tl_semaphore);
        }
    }
}

अटल ggml_backend_i ggml_backend_vk_interface = {
    /* .get_name                = */ ggml_backend_vk_name,
    /* .free                    = */ ggml_backend_vk_free,
    /* .set_tensor_async        = */ ggml_backend_vk_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_vk_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_vk_set_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_vk_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ ggml_backend_vk_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_vk_synchronize,
    /* .graph_plan_create       = */ शून्य,
    /* .graph_plan_free         = */ शून्य,
    /* .graph_plan_update       = */ शून्य,
    /* .graph_plan_compute      = */ शून्य,
    /* .graph_compute           = */ ggml_backend_vk_graph_compute,
    /* .event_record            = */ ggml_backend_vk_event_record,
    /* .event_wait              = */ ggml_backend_vk_event_wait,
    /* .graph_optimize          = */ ggml_vk_graph_optimize,
};

अटल ggml_guid_t ggml_backend_vk_guid() {
    अटल ggml_guid guid = { 0xb8, 0xf7, 0x4f, 0x86, 0x40, 0x3c, 0xe1, 0x02, 0x91, 0xc8, 0xdd, 0xe9, 0x02, 0x3f, 0xc0, 0x2b };
    वापस &guid;
}

ggml_backend_t ggml_backend_vk_init(माप_प्रकार dev_अंक) {
    VK_LOG_DEBUG("ggml_backend_vk_init(" << dev_अंक << ")");

    ggml_backend_vk_context * ctx = नया ggml_backend_vk_context;
    ggml_vk_init(ctx, dev_अंक);

    ggml_backend_t vk_backend = नया ggml_backend {
        /* .guid    = */ ggml_backend_vk_guid(),
        /* .iface   = */ ggml_backend_vk_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_vk_reg(), dev_अंक),
        /* .context = */ ctx,
    };

    अगर (!ctx->device->support_async) {
        vk_backend->iface.get_tensor_async = nullptr;
    }

    वापस vk_backend;
}

bool ggml_backend_is_vk(ggml_backend_t backend) {
    वापस backend != शून्य && ggml_guid_matches(backend->guid, ggml_backend_vk_guid());
}

पूर्णांक ggml_backend_vk_get_device_गिनती() {
    वापस ggml_vk_get_device_गिनती();
}

व्योम ggml_backend_vk_get_device_description(पूर्णांक device, अक्षर * description, माप_प्रकार description_आकार) {
    GGML_ASSERT(device < (पूर्णांक) vk_instance.device_indices.size());
    पूर्णांक dev_idx = vk_instance.device_indices[device];
    ggml_vk_get_device_description(dev_idx, description, description_आकार);
}

व्योम ggml_backend_vk_get_device_memory(पूर्णांक device, माप_प्रकार * मुक्त, माप_प्रकार * योग) {
    GGML_ASSERT(device < (पूर्णांक) vk_instance.device_indices.size());
    GGML_ASSERT(device < (पूर्णांक) vk_instance.device_supports_membudget.size());

    vk::PhysicalDevice vkdev = vk_instance.instance.enumeratePhysicalDevices()[vk_instance.device_indices[device]];
    vk::PhysicalDeviceMemoryBudgetPropertiesEXT budgetprops;
    vk::PhysicalDeviceMemoryProperties2 memprops = {};
    स्थिर bool membudget_supported = vk_instance.device_supports_membudget[device];
    स्थिर bool is_integrated_gpu = vkdev.getProperties().deviceType == vk::PhysicalDeviceType::eIntegratedGpu;

    अगर (membudget_supported) {
        memprops.p_अग्र = &budgetprops;
    }
    vkdev.getMemoryProperties2(&memprops);

    *योग = 0;
    *मुक्त = 0;

    क्रम (uint32_t i = 0; i < memprops.memoryProperties.memory_Heap_गिनती; ++i) {
        स्थिर vk::MemoryHeap & heap = memprops.memoryProperties.memoryHeaps[i];

        अगर (is_integrated_gpu || (heap.flags & vk::MemoryHeapFlagBits::eDeviceLocal)) {
            *योग += heap.size;

            अगर (membudget_supported && i < budgetprops.heapUsage.size()) {
                // Driver may report heapBudget > heap.size so we add a safeguard here
                // See https://github.com/ggml-org/llama.cpp/issues/29277
                स्थिर vk::Device_आकार budget = std::min(budgetprops.heapBudget[i], heap.size);
                *मुक्त += budget > budgetprops.heapUsage[i] ? budget - budgetprops.heapUsage[i] : 0;
            } अन्यथा {
                *मुक्त += heap.size;
            }
        }
    }
}

अटल vk::PhysicalDeviceType ggml_backend_vk_get_device_type(पूर्णांक device_idx) {
    GGML_ASSERT(device_idx >= 0 && device_idx < (पूर्णांक) vk_instance.device_indices.size());

    vk::PhysicalDevice device = vk_instance.instance.enumeratePhysicalDevices()[vk_instance.device_indices[device_idx]];

    vk::PhysicalDeviceProperties2 props = {};
    device.getProperties2(&props);

    वापस props.properties.deviceType;
}

अटल std::string ggml_backend_vk_get_device_pci_id(पूर्णांक device_idx) {
    GGML_ASSERT(device_idx >= 0 && device_idx < (पूर्णांक) vk_instance.device_indices.size());

    vk::PhysicalDevice device = vk_instance.instance.enumeratePhysicalDevices()[vk_instance.device_indices[device_idx]];

    स्थिर std::vector<vk::ExtensionProperties> ext_props = device.enumerateDeviceExtensionProperties();

    bool ext_support = false;

    क्रम (स्थिर स्वतः& properties : ext_props) {
        अगर (म_भेद("VK_EXT_pci_bus_info", properties.extensionName) == 0) {
            ext_support = true;
            अवरोध;
        }
    }

    अगर (!ext_support) {
        वापस "";
    }

    vk::PhysicalDeviceProperties2 props = {};
    vk::PhysicalDevicePCIBusInfoPropertiesEXT pci_bus_info = {};

    props.p_अग्र = &pci_bus_info;

    device.getProperties2(&props);

    स्थिर uint32_t pci_domain = pci_bus_info.pciDomain;
    स्थिर uint32_t pci_bus = pci_bus_info.pciBus;
    स्थिर uint32_t pci_device = pci_bus_info.pciDevice;
    स्थिर uint8_t pci_function = (uint8_t) pci_bus_info.pciFunction; // pci function is between 0 and 7, prevent printf overflow warning

    अक्षर pci_bus_id[16] = {};
    snprintf(pci_bus_id, माप(pci_bus_id), "%04x:%02x:%02x.%x", pci_domain, pci_bus, pci_device, pci_function);

    वापस std::string(pci_bus_id);
}

अटल स्थिर अक्षर * ggml_backend_vk_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    वापस ctx->name.c_str();
}

अटल स्थिर अक्षर * ggml_backend_vk_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    वापस ctx->description.c_str();
}

अटल व्योम ggml_backend_vk_device_get_memory(ggml_backend_dev_t device, माप_प्रकार * मुक्त, माप_प्रकार * योग) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)device->context;
    ggml_backend_vk_get_device_memory(ctx->device, मुक्त, योग);
}

अटल ggml_backend_बफर_type_t ggml_backend_vk_device_get_बफर_type(ggml_backend_dev_t dev) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    वापस ggml_backend_vk_बफर_type(ctx->device);
}

अटल ggml_backend_बफर_type_t ggml_backend_vk_device_get_host_बफर_type(ggml_backend_dev_t dev) {
    UNUSED(dev);
    वापस ggml_backend_vk_host_बफर_type();
}

अटल क्रमागत ggml_backend_dev_type ggml_backend_vk_device_get_type(ggml_backend_dev_t dev) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;

    वापस ctx->is_integrated_gpu ? GGML_BACKEND_DEVICE_TYPE_IGPU : GGML_BACKEND_DEVICE_TYPE_GPU;
}

अटल व्योम ggml_backend_vk_device_get_props(ggml_backend_dev_t dev, काष्ठा ggml_backend_dev_props * props) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;

    props->name        = ggml_backend_vk_device_get_name(dev);
    props->description = ggml_backend_vk_device_get_description(dev);
    props->type        = ggml_backend_vk_device_get_type(dev);
    props->device_id   = ctx->pci_bus_id.empty() ? nullptr : ctx->pci_bus_id.c_str();
    ggml_backend_vk_device_get_memory(dev, &props->memory_free, &props->memory_योग);
    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ true,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ true,
        /* .mmap_support          = */ !ctx->is_integrated_gpu,
    };
}

अटल ggml_backend_t ggml_backend_vk_device_init(ggml_backend_dev_t dev, स्थिर अक्षर * params) {
    UNUSED(params);
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    वापस ggml_backend_vk_init(ctx->device);
}

अटल bool ggml_backend_vk_device_supports_op(ggml_backend_dev_t dev, स्थिर ggml_tensor * op) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    स्थिर vk_device& device = ggml_vk_get_device(ctx->device);

    स्थिर bool uses_bda = (op->op == GGML_OP_IM2COL || op->op == GGML_OP_IM2COL_3D) &&
                          device->shader_int64 && device->बफर_device_address;

    स्वतः स्थिर & tensor_आकार_supported = [&](माप_प्रकार tensor_आकार) {
        अगर (tensor_आकार > device->max_बफर_आकार) {
            वापस false;
        }
        // For im2col shaders using BDA, maxStorageBufferRange limit doesn't apply.
        // If shader64BitIndexing is enabled, maxStorageBufferRange limit doesn't apply.
        अगर (!uses_bda && !device->shader_64b_indexing) {
            अगर (tensor_आकार > device->properties.limits.max_Storage_बफर_Range) {
                वापस false;
            }
        }
        वापस true;
    };
    // reject any tensors larger than the max buffer size
    क्रम (पूर्णांक i = 0; i < GGML_MAX_SRC; i++) {
        अगर (op->src[i] && !tensor_आकार_supported(ggml_nbytes(op->src[i]))) {
            वापस false;
        }
    }
    अगर (!tensor_आकार_supported(ggml_nbytes(op))) {
        वापस false;
    }

    चयन (op->op) {
        हाल GGML_OP_UNARY:
            चयन (ggml_get_unary_op(op)) {
                हाल GGML_UNARY_OP_EXP:
                हाल GGML_UNARY_OP_EXPM1:
                हाल GGML_UNARY_OP_ELU:
                हाल GGML_UNARY_OP_GELU:
                हाल GGML_UNARY_OP_GELU_ERF:
                हाल GGML_UNARY_OP_GELU_QUICK:
                हाल GGML_UNARY_OP_SILU:
                हाल GGML_UNARY_OP_RELU:
                हाल GGML_UNARY_OP_XIELU:
                हाल GGML_UNARY_OP_NEG:
                हाल GGML_UNARY_OP_TANH:
                हाल GGML_UNARY_OP_SIGMOID:
                हाल GGML_UNARY_OP_HARDSIGMOID:
                हाल GGML_UNARY_OP_HARDSWISH:
                हाल GGML_UNARY_OP_ABS:
                हाल GGML_UNARY_OP_SOFTPLUS:
                हाल GGML_UNARY_OP_STEP:
                हाल GGML_UNARY_OP_ROUND:
                हाल GGML_UNARY_OP_CEIL:
                हाल GGML_UNARY_OP_FLOOR:
                हाल GGML_UNARY_OP_TRUNC:
                हाल GGML_UNARY_OP_SGN:
                    वापस (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                           (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                           (op->src[0]->type == op->type);
                शेष:
                    वापस false;
            }
        हाल GGML_OP_GLU:
            चयन (ggml_get_glu_op(op)) {
                हाल GGML_GLU_OP_GEGLU:
                हाल GGML_GLU_OP_REGLU:
                हाल GGML_GLU_OP_SWIGLU:
                हाल GGML_GLU_OP_SWIGLU_OAI:
                हाल GGML_GLU_OP_GEGLU_ERF:
                हाल GGML_GLU_OP_GEGLU_QUICK:
                हाल GGML_GLU_OP_SWIGLU_CLAMP:
                    वापस (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                           (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                           (op->src[0]->type == op->type) &&
                           (!op->src[1] || op->src[1]->type == op->src[0]->type);
                शेष:
                    वापस false;
            }
        हाल GGML_OP_MUL_MAT:
        हाल GGML_OP_MUL_MAT_ID:
            {
                ggml_type src0_type = op->src[0]->type;
                अगर (op->op == GGML_OP_MUL_MAT_ID) {
                    अगर (!ggml_vk_use_mul_mat_vec_id(op)) {
                        स्थिर ggml_tensor * ids = op->src[2];
                        // Shaders store row IDs as uint16_t slot and token indices.
                        अगर (ids->ne[0] > int64_t(UINT16_MAX) + 1 || ids->ne[1] > int64_t(UINT16_MAX) + 1) {
                            वापस false;
                        }
                    }
                    अगर (!device->mul_mat_id_s[src0_type] && !device->mul_mat_id_m[src0_type] && !device->mul_mat_id_l[src0_type]) {
                        // If there's not enough shared memory for row_ids and the result tile, fallback to CPU
                        वापस false;
                    }
                    अगर (ggml_get_op_params_i32(op, 3) == GGML_PREC_F32) {
                        वापस false;
                    }
                }
                चयन (src0_type) {
                    हाल GGML_TYPE_F32:
                    हाल GGML_TYPE_F16:
                    हाल GGML_TYPE_BF16:
                    हाल GGML_TYPE_Q1_0:
                    हाल GGML_TYPE_Q2_0:
                    हाल GGML_TYPE_Q4_0:
                    हाल GGML_TYPE_Q4_1:
                    हाल GGML_TYPE_Q5_0:
                    हाल GGML_TYPE_Q5_1:
                    हाल GGML_TYPE_Q8_0:
                    हाल GGML_TYPE_Q2_K:
                    हाल GGML_TYPE_Q3_K:
                    हाल GGML_TYPE_Q4_K:
                    हाल GGML_TYPE_Q5_K:
                    हाल GGML_TYPE_Q6_K:
                    हाल GGML_TYPE_IQ1_S:
                    हाल GGML_TYPE_IQ1_M:
                    हाल GGML_TYPE_IQ2_XXS:
                    हाल GGML_TYPE_IQ2_XS:
                    हाल GGML_TYPE_IQ2_S:
                    हाल GGML_TYPE_IQ3_XXS:
                    हाल GGML_TYPE_IQ3_S:
                    हाल GGML_TYPE_IQ4_XS:
                    हाल GGML_TYPE_IQ4_NL:
                    हाल GGML_TYPE_MXFP4:
                    हाल GGML_TYPE_NVFP4:
                    हाल GGML_TYPE_TQ1_0:
                    हाल GGML_TYPE_TQ2_0:
                        अवरोध;
                    शेष:
                        वापस false;
                }
                काष्ठा ggml_tensor * a;
                काष्ठा ggml_tensor * b;
                अगर (op->op == GGML_OP_MUL_MAT) {
                    a = op->src[0];
                    b = op->src[1];
                } अन्यथा {
                    a = op->src[2];
                    b = op->src[1];
                }
                अगर (a->ne[3] != b->ne[3]) {
                    वापस false;
                }
                अगर (!(ggml_vk_dim01_contiguous(op->src[0]) || op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_BF16) ||
                    !(ggml_vk_dim01_contiguous(op->src[1]) || op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16)) {
                    वापस false;
                }
                अगर (op->src[0]->type == GGML_TYPE_BF16 && op->src[1]->type == GGML_TYPE_F16) {
                    // We currently don't have a bf16 x f16 shader, or an fp16->bf16 copy shader.
                    // So don't support this combination for now.
                    वापस false;
                }
                अगर (op->src[1]->type == GGML_TYPE_BF16 && op->src[0]->type != GGML_TYPE_BF16 && op->op != GGML_OP_MUL_MAT) {
                    // BF16 in src1 (widened to F32/F16) is only wired up for MUL_MAT
                    वापस false;
                }

                वापस true;
            }
        हाल GGML_OP_FLASH_ATTN_EXT:
            {
                bool coopmat2 = device->coopmat2;
                uint32_t HSK = op->src[1]->ne[0];
                uint32_t HSV = op->src[2]->ne[0];
                अगर ((HSK % 8) != 0 || (HSV % 8) != 0) {
                    वापस false;
                }
                अगर (op->src[4] && op->src[4]->type != GGML_TYPE_F32) {
                    वापस false;
                }
                अगर (op->src[0]->type != GGML_TYPE_F32) {
                    वापस false;
                }
                अगर (op->type != GGML_TYPE_F32) {
                    वापस false;
                }
                अगर (op->src[3] && op->src[3]->type != GGML_TYPE_F16) {
                    वापस false;
                }
                स्वतः fa_kv_ok = [](ggml_type t) {
                    चयन (t) {
                    हाल GGML_TYPE_F32:
                    हाल GGML_TYPE_F16:
                    हाल GGML_TYPE_BF16:
                    हाल GGML_TYPE_Q8_0:
                    हाल GGML_TYPE_Q5_1:
                    हाल GGML_TYPE_Q5_0:
                    हाल GGML_TYPE_Q4_1:
                    हाल GGML_TYPE_Q4_0:
                    हाल GGML_TYPE_IQ4_NL:
                        वापस true;
                    शेष:
                        वापस false;
                    }
                };
                अगर (!fa_kv_ok(op->src[1]->type) || !fa_kv_ok(op->src[2]->type)) {
                    वापस false;
                }
                अगर ((op->src[1]->type == GGML_TYPE_BF16) != (op->src[2]->type == GGML_TYPE_BF16)) {
                    वापस false;
                }
                अगर (!coopmat2 && !(device->subgroup_shuffle && device->subgroup_vote)) {
                    // scalar/coopmat1 FA uses subgroupShuffle/subgroupAll
                    वापस false;
                }
                वापस true;
            }
        हाल GGML_OP_GET_ROWS:
            {
                चयन (op->src[0]->type) {
                    हाल GGML_TYPE_F32:
                    हाल GGML_TYPE_F16:
                    हाल GGML_TYPE_BF16:
                    हाल GGML_TYPE_Q1_0:
                    हाल GGML_TYPE_Q2_0:
                    हाल GGML_TYPE_Q4_0:
                    हाल GGML_TYPE_Q4_1:
                    हाल GGML_TYPE_Q5_0:
                    हाल GGML_TYPE_Q5_1:
                    हाल GGML_TYPE_Q8_0:
                    हाल GGML_TYPE_Q2_K:
                    हाल GGML_TYPE_Q3_K:
                    हाल GGML_TYPE_Q4_K:
                    हाल GGML_TYPE_Q5_K:
                    हाल GGML_TYPE_Q6_K:
                    हाल GGML_TYPE_IQ1_S:
                    हाल GGML_TYPE_IQ1_M:
                    हाल GGML_TYPE_IQ2_XXS:
                    हाल GGML_TYPE_IQ2_XS:
                    हाल GGML_TYPE_IQ2_S:
                    हाल GGML_TYPE_IQ3_XXS:
                    हाल GGML_TYPE_IQ3_S:
                    हाल GGML_TYPE_IQ4_XS:
                    हाल GGML_TYPE_IQ4_NL:
                    हाल GGML_TYPE_MXFP4:
                    हाल GGML_TYPE_NVFP4:
                    हाल GGML_TYPE_TQ1_0:
                    हाल GGML_TYPE_TQ2_0:
                    हाल GGML_TYPE_I32:
                        वापस true;
                    शेष:
                        वापस false;
                }
            }
        हाल GGML_OP_GET_ROWS_BACK:
            वापस op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32;
        हाल GGML_OP_SET_ROWS:
            {
                अगर ((op->src[0]->type != GGML_TYPE_F32 && op->src[0]->type != GGML_TYPE_F16) ||
                    (op->src[1]->type != GGML_TYPE_I32 && op->src[1]->type != GGML_TYPE_I64)) {
                    वापस false;
                }
                चयन (op->type) {
                    हाल GGML_TYPE_F32:
                    हाल GGML_TYPE_F16:
                    हाल GGML_TYPE_BF16:
                    हाल GGML_TYPE_Q1_0:
                    हाल GGML_TYPE_Q2_0:
                    हाल GGML_TYPE_Q4_0:
                    हाल GGML_TYPE_Q4_1:
                    हाल GGML_TYPE_Q5_0:
                    हाल GGML_TYPE_Q5_1:
                    हाल GGML_TYPE_Q8_0:
                    हाल GGML_TYPE_IQ4_NL:
                        वापस true;
                    शेष:
                        वापस false;
                }
            }
        हाल GGML_OP_CONT:
        हाल GGML_OP_CPY:
        हाल GGML_OP_DUP:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1] != nullptr ? op->src[1]->type : src0_type;

                अगर (src0_type == GGML_TYPE_F32) {
                    चयन (src1_type) {
                    हाल GGML_TYPE_F32:
                    हाल GGML_TYPE_F16:
                    हाल GGML_TYPE_BF16:
                    हाल GGML_TYPE_Q1_0:
                    हाल GGML_TYPE_Q2_0:
                    हाल GGML_TYPE_Q4_0:
                    हाल GGML_TYPE_Q4_1:
                    हाल GGML_TYPE_Q5_0:
                    हाल GGML_TYPE_Q5_1:
                    हाल GGML_TYPE_Q8_0:
                    हाल GGML_TYPE_IQ4_NL:
                        वापस true;
                    शेष:
                        अवरोध;
                    }
                }
                अगर (src1_type == GGML_TYPE_F32) {
                    चयन (src0_type) {
                    हाल GGML_TYPE_F16:
                    हाल GGML_TYPE_BF16:
                    हाल GGML_TYPE_Q1_0:
                    हाल GGML_TYPE_Q2_0:
                    हाल GGML_TYPE_Q4_0:
                    हाल GGML_TYPE_Q4_1:
                    हाल GGML_TYPE_Q5_0:
                    हाल GGML_TYPE_Q5_1:
                    हाल GGML_TYPE_Q8_0:
                    हाल GGML_TYPE_IQ4_NL:
                        वापस true;
                    शेष:
                        अवरोध;
                    }
                }

                अगर (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F16) {
                    वापस true;
                }

                अगर (
                    (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_I32) ||
                    (src0_type == GGML_TYPE_I32 && src1_type == GGML_TYPE_F32)
                ) {
                    वापस true;
                }

                // We can handle copying from a type to the same type if it's
                // either not quantized or is quantized and contiguous.
                // We use f16 or f32 shaders to do the copy,
                // so the type/block size must be a multiple of 4.
                अगर (src0_type == src1_type &&
                    (!ggml_is_quantized(src0_type) || (ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op))) &&
                    (ggml_type_आकार(src0_type) % 2) == 0) {
                    वापस true;
                }
                वापस false;
            }
        हाल GGML_OP_REPEAT:
            वापस ggml_type_आकार(op->type) == ggml_type_आकार(op->src[0]->type) &&
                  (ggml_type_आकार(op->type) == माप(भग्न) || ggml_type_आकार(op->type) == 2);
        हाल GGML_OP_REPEAT_BACK:
            वापस op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32;
        हाल GGML_OP_ROPE:
            वापस ggml_is_contiguous_rows(op) && ggml_is_contiguous_rows(op->src[0]);
        हाल GGML_OP_ROPE_BACK:
        हाल GGML_OP_NONE:
        हाल GGML_OP_RESHAPE:
        हाल GGML_OP_VIEW:
        हाल GGML_OP_PERMUTE:
        हाल GGML_OP_TRANSPOSE:
        हाल GGML_OP_RMS_NORM:
            वापस true;
        हाल GGML_OP_GROUP_NORM:
            वापस ggml_is_contiguous(op->src[0]);
        हाल GGML_OP_NORM:
        हाल GGML_OP_L2_NORM:
            वापस op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        हाल GGML_OP_ADD:
        हाल GGML_OP_SUB:
        हाल GGML_OP_MUL:
        हाल GGML_OP_DIV:
            वापस (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                   (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) &&
                   (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16);
        हाल GGML_OP_ADD_ID:
            वापस op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 && op->src[2]->type == GGML_TYPE_I32 &&
                   op->type == GGML_TYPE_F32;
        हाल GGML_OP_SILU_BACK:
        हाल GGML_OP_RMS_NORM_BACK:
            वापस ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        हाल GGML_OP_SQR:
        हाल GGML_OP_SQRT:
        हाल GGML_OP_SIN:
        हाल GGML_OP_COS:
        हाल GGML_OP_CLAMP:
        हाल GGML_OP_LEAKY_RELU:
            वापस (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                   op->type == op->src[0]->type;
        हाल GGML_OP_OPT_STEP_ADAMW:
        हाल GGML_OP_OPT_STEP_SGD:
            वापस ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        हाल GGML_OP_OUT_PROD:
            वापस ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32
                && ggml_is_contiguous(op->src[1]) && op->src[1]->type == GGML_TYPE_F32
                && op->type == GGML_TYPE_F32;
        हाल GGML_OP_LOG:
        हाल GGML_OP_TRI:
        हाल GGML_OP_DIAG:
            वापस (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                   op->type == op->src[0]->type;
        हाल GGML_OP_ARGSORT:
            {
                अगर (!ggml_is_contiguous(op) || !ggml_is_contiguous(op->src[0])) {
                    वापस false;
                }
                // pipeline_argsort_large_f32 requires vulkan memory model.
                अगर (device->vulkan_memory_model) {
                    वापस true;
                } अन्यथा {
                    वापस op->ne[0] <= (1 << std::min(device->max_workgroup_आकार_log2, अंक_argsort_pipelines - 1));
                }
            }
        हाल GGML_OP_शीर्ष_K:
            {
                अगर (!ggml_is_contiguous(op) || !ggml_is_contiguous(op->src[0])) {
                    वापस false;
                }
                // large k falls back to radix-select
                स्थिर uint32_t min_pipeline =
                    std::max((uint32_t) log2f(भग्न(op->ne[0])) + 1, device->subgroup_आकार_log2);
                अगर (min_pipeline < अंक_topk_pipelines && device->pipeline_topk_f32[min_pipeline]) {
                    वापस true;
                }
                वापस device->pipeline_topk_radix_f32 != nullptr;
            }
        हाल GGML_OP_UPSCALE:
            अगर (op->op_params[0] & GGML_SCALE_FLAG_ANTIALIAS) {
                अगर ((op->op_params[0] & 0xFF) != GGML_SCALE_MODE_BILINEAR) {
                    वापस false;
                }
            }
            वापस op->src[0]->type == GGML_TYPE_F32;
        हाल GGML_OP_ACC:
            वापस op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        हाल GGML_OP_SET:
            वापस op->src[0]->type == op->src[1]->type && op->src[0]->type == op->type &&
                   (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_I32);
        हाल GGML_OP_CONCAT: {
            वापस ggml_vk_concat_supported(op->src[0], op->src[1], op);
        }
        हाल GGML_OP_ADD1:
            वापस (op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32)
                || (op->src[0]->type == GGML_TYPE_F16 && op->src[1]->type == GGML_TYPE_F32)
                || (op->src[0]->type == GGML_TYPE_F16 && op->src[1]->type == GGML_TYPE_F16);
        हाल GGML_OP_ARANGE:
            वापस op->type == GGML_TYPE_F32;
        हाल GGML_OP_FILL:
            वापस op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16;
        हाल GGML_OP_SCALE:
            वापस ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        हाल GGML_OP_PAD:
        हाल GGML_OP_PAD_REFLECT_1D:
        हाल GGML_OP_ROLL:
            वापस op->src[0]->type == GGML_TYPE_F32;
        हाल GGML_OP_DIAG_MASK_INF:
            वापस ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        हाल GGML_OP_SOFT_MAX:
            वापस ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32
                && (!op->src[1] || (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16));
        हाल GGML_OP_SOFT_MAX_BACK:
            वापस ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32
                && ggml_is_contiguous(op->src[1]) && op->src[1]->type == GGML_TYPE_F32;
        हाल GGML_OP_योग:
            वापस op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguously_allocated(op->src[0]);
        हाल GGML_OP_योग_ROWS:
        हाल GGML_OP_MEAN:
            वापस op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous_rows(op->src[0]);
        हाल GGML_OP_CUMSUM:
            {
                अगर (device->subgroup_arithmetic && device->subgroup_require_full_support) {
                    वापस op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous_rows(op->src[0]);
                }
                वापस false;
            }
        हाल GGML_OP_DSV4_HC_COMB:
        हाल GGML_OP_DSV4_HC_PRE:
        हाल GGML_OP_DSV4_HC_POST:
            {
                अगर (op->type != GGML_TYPE_F32) {
                    वापस false;
                }
                क्रम (uint32_t i = 0; i < GGML_MAX_SRC; ++i) {
                    अगर (op->src[i] && op->src[i]->type != GGML_TYPE_F32) {
                        वापस false;
                    }
                }
                // hc is hardcoded to 4 in the shaders. ggml only constrains it
                // to 4 for COMB, so PRE/POST have to be checked here.
                अगर (op->op == GGML_OP_DSV4_HC_PRE && op->src[0]->ne[1] != 4) {
                    वापस false;
                }
                अगर (op->op == GGML_OP_DSV4_HC_POST && op->src[1]->ne[1] != 4) {
                    वापस false;
                }
                अगर (op->op == GGML_OP_DSV4_HC_COMB) {
                    वापस device->pipeline_dsv4_hc_comb_f32 != nullptr;
                }
                वापस true;
            }
        हाल GGML_OP_SOLVE_TRI:
            {
                अगर (op->type != GGML_TYPE_F32 || op->src[0]->type != GGML_TYPE_F32) {
                    वापस false;
                }
                स्थिर uint32_t N = op->src[0]->ne[0];
                स्थिर uint32_t K = op->src[1]->ne[0];
                // K dimension limited to workgroup size
                अगर (K > 1u << device->max_workgroup_आकार_log2) {
                    वापस false;
                }
                स्थिर uint32_t batch_N = device->properties.limits.max_Compute_Shared_Memory_आकार / ((N + K) * माप(भग्न));

                अगर (batch_N == 0) {
                    वापस false;
                }
                वापस true;
            }
        हाल GGML_OP_ARGMAX:
            वापस ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        हाल GGML_OP_CROSS_ENTROPY_LOSS:
            वापस ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32
                && ggml_is_contiguous(op->src[1]) && op->src[1]->type == GGML_TYPE_F32
                && ggml_are_same_shape(op->src[0], op->src[1])
                && ggml_is_contiguous(op) && ggml_is_scalar(op) && op->type == GGML_TYPE_F32;
        हाल GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            वापस ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32 && ggml_is_scalar(op->src[0])
                && ggml_is_contiguous(op->src[1]) && op->src[1]->type == GGML_TYPE_F32
                && ggml_is_contiguous(op->src[2]) && op->src[2]->type == GGML_TYPE_F32
                && ggml_are_same_shape(op->src[1], op->src[2])
                && ggml_are_same_shape(op->src[1], op)
                && ggml_is_contiguous(op) && op->type == GGML_TYPE_F32;
        हाल GGML_OP_गिनती_EQUAL:
            वापस ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_I32
                && ggml_is_contiguous(op->src[1]) && op->src[1]->type == GGML_TYPE_I32;
        हाल GGML_OP_IM2COL:
            वापस ggml_is_contiguous(op->src[1])
                && op->src[1]->type == GGML_TYPE_F32
                && (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16);
        हाल GGML_OP_IM2COL_3D:
            वापस op->src[1]->type == GGML_TYPE_F32
                && (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16);
        हाल GGML_OP_TIMESTEP_EMBEDDING:
            वापस op->src[0]->type == GGML_TYPE_F32;
        हाल GGML_OP_CONV_2D_DW:
            वापस (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16)
                && op->src[1]->type == GGML_TYPE_F32;
        हाल GGML_OP_POOL_1D:
            वापस ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        हाल GGML_OP_POOL_2D:
            वापस ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        हाल GGML_OP_RWKV_WKV6:
        हाल GGML_OP_RWKV_WKV7:
            वापस true; // all inputs are contiguous, see ggml.c
        हाल GGML_OP_GATED_LINEAR_ATTN:
            // the shader block size is hardcoded to head_size 64
            वापस op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && op->src[0]->ne[0] == 64;
        हाल GGML_OP_LIGHTNING_INDEXER:
            {
                स्थिर ggml_tensor * q = op->src[0];
                स्थिर ggml_tensor * k = op->src[1];
                स्थिर ggml_tensor * w = op->src[2];
                स्थिर ggml_tensor * m = op->src[3];

                // the q/w/m types and the shape relationships between q, k, w, m and dst
                // are already asserted in ggml_lightning_indexer()
                अगर (!ggml_vk_lightning_indexer_k_type_supported(k->type) || !device->fp16) {
                    वापस false;
                }

                // the shader block size is hardcoded to head size 128
                अगर (q->ne[0] != 128) {
                    वापस false;
                }

                // the shader indexes the buffers by element stride, and is dispatched
                // without allow_misalign
                क्रम (स्थिर ggml_tensor * t : {q, k, w, m, op}) {
                    अगर (t->nb[0] != ggml_type_आकार(t->type) ||
                        (vk_tensor_offset(t) + t->view_offs) % device->properties.limits.min_Storage_बफर_Offset_Alignment != 0) {
                        वापस false;
                    }
                    // the strides get scaled down from bytes, so the division must be exact
                    क्रम (पूर्णांक i = 1; i < GGML_MAX_DIMS; ++i) {
                        अगर (t->nb[i] % ggml_type_आकार(t->type) != 0) {
                            वापस false;
                        }
                    }
                }
                वापस true;
            }
        हाल GGML_OP_GATED_DELTA_NET:
            {
                स्थिर uint32_t S_v = op->src[2]->ne[0];
                अगर (S_v != 16 && S_v != 32 && S_v != 64 && S_v != 128) {
                    वापस false;
                }
                क्रम (पूर्णांक i = 0; i < 6; i++) {
                    अगर (op->src[i] == nullptr || op->src[i]->type != GGML_TYPE_F32) {
                        वापस false;
                    }
                }
                वापस op->type == GGML_TYPE_F32;
            }
        हाल GGML_OP_SSM_SCAN:
            {
                क्रम (पूर्णांक i = 0; i < 6; i++) {
                    अगर (op->src[i] && ggml_is_quantized(op->src[i]->type)) {
                        वापस false;
                    }
                }
                अगर (op->src[6] && op->src[6]->type != GGML_TYPE_I32) {
                    वापस false;
                }
                अगर (op->src[0]->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
                    वापस false;
                }

                स्थिर uint32_t d_state = op->src[0]->ne[0];
                स्थिर uint32_t शीर्ष_dim = op->src[0]->ne[1];

                bool is_mamba2 = (op->src[3] && op->src[3]->nb[1] == माप(भग्न));
                अगर (!is_mamba2) {
                    वापस false;
                }

                अगर ((d_state != 128 && d_state != 256) || शीर्ष_dim % 16 != 0) {
                    वापस false;
                }

                माप_प्रकार shmem_आकार = d_state * माप(भग्न);

                अगर (shmem_आकार > device->properties.limits.max_Compute_Shared_Memory_आकार) {
                    वापस false;
                }

                अगर (!device->subgroup_basic) {
                    वापस false;
                }

                वापस true;
            }
        हाल GGML_OP_SSM_CONV:
            वापस op->src[0]->type == GGML_TYPE_F32;
        हाल GGML_OP_CONV_TRANSPOSE_1D:
            वापस op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32;
        हाल GGML_OP_COL2IM_1D:
            वापस (op->src[0]->type == GGML_TYPE_F32 ||
                    op->src[0]->type == GGML_TYPE_F16 ||
                    op->src[0]->type == GGML_TYPE_BF16) &&
                   op->type == op->src[0]->type &&
                   ggml_is_contiguous(op->src[0]) &&
                   ggml_is_contiguous(op);
        हाल GGML_OP_CONV_2D:
        हाल GGML_OP_CONV_TRANSPOSE_2D:
            {
                स्थिर bool transpose = op->op == GGML_OP_CONV_TRANSPOSE_2D;
                स्थिर int64_t अ_बाहर = !transpose ? op->src[0]->ne[3] : op->src[0]->ne[2];
                स्थिर int64_t अ_अंदर  = !transpose ? op->src[0]->ne[2] : op->src[0]->ne[3];

                // Channel-contiguous format is not supported yet.
                वापस ((op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                    (op->src[0]->nb[0] == माप(भग्न) || op->src[0]->nb[0] == माप(ggml_fp16_t) ) &&
                    op->src[1]->type == GGML_TYPE_F32 &&
                    op->type == GGML_TYPE_F32 &&
                    अ_बाहर == op->ne[2] &&
                    अ_अंदर == op->src[1]->ne[2] &&
                    ggml_is_contiguous(op->src[0]) &&
                    ggml_is_contiguous(op->src[1]) &&
                    ggml_is_contiguous(op));
            }
        हाल GGML_OP_CONV_3D:
            वापस (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                op->src[1]->type == GGML_TYPE_F32 &&
                op->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(op->src[0]) &&
                ggml_is_contiguous(op->src[1]) &&
                ggml_is_contiguous(op);
        शेष:
            वापस false;
    }

    UNUSED(dev);
}

अटल bool ggml_backend_vk_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_बफर_type_t buft) {
    अगर (buft->iface.get_name != ggml_backend_vk_बफर_type_name) {
        वापस false;
    }

    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    ggml_backend_vk_बफर_type_context * buft_ctx = (ggml_backend_vk_बफर_type_context *)buft->context;

    वापस buft_ctx->device->idx == ctx->device;
}

अटल bool ggml_backend_vk_device_offload_op(ggml_backend_dev_t dev, स्थिर ggml_tensor * op) {
    ggml_backend_vk_device_context * dev_ctx = (ggml_backend_vk_device_context *)dev->context;

    वापस ggml_vk_get_op_batch_आकार(op) >= dev_ctx->op_offload_min_batch_आकार;
}

अटल ggml_backend_event_t ggml_backend_vk_device_event_new(ggml_backend_dev_t dev) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    स्वतः device = ggml_vk_get_device(ctx->device);

    vk_event *vkev = नया vk_event;
    अगर (!vkev) {
        वापस nullptr;
    }

    // No events initially, they get created on demand
    vkev->has_event = false;

    vk::SemaphoreTypeCreateInfo tci{ vk::SemaphoreType::eTimeline, 0 };
    vk::SemaphoreCreateInfo ci{};
    ci.setPNext(&tci);
    vkev->tl_semaphore = { device->device.createSemaphore(ci), 0 };

    वापस नया ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ vkev,
    };
}

अटल व्योम ggml_backend_vk_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    स्वतः device = ggml_vk_get_device(ctx->device);

    vk_event *vkev = (vk_event *)event->context;

    device->device.destroySemaphore(vkev->tl_semaphore.s);
    क्रम (स्वतः& event : vkev->events_free) {
        device->device.destroyEvent(event);
    }
    क्रम (स्वतः& event : vkev->events_submitted) {
        device->device.destroyEvent(event);
    }
    अगर (vkev->has_event) {
        device->device.destroyEvent(vkev->event);
    }
    मिटाओ vkev;
    मिटाओ event;
}

अटल व्योम ggml_backend_vk_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    VK_LOG_DEBUG("ggml_backend_vk_device_event_synchronize(backend=" << dev << ", event=" << event << ")");
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    स्वतः device = ggml_vk_get_device(ctx->device);
    vk_event *vkev = (vk_event *)event->context;

    // Only do something if the event has actually been used
    अगर (vkev->has_event) {
        vk::Semaphore sem = vkev->tl_semaphore.s;
        uint64_t val = vkev->tl_semaphore.value;
        vk::SemaphoreWaitInfo swi{vk::SemaphoreWaitFlags{}, sem, val};
        VK_CHECK(device->device.waitSemaphores(swi, UINT64_MAX), "event_synchronize", device);

        // Reset and move submitted events
        क्रम (स्वतः& event : vkev->events_submitted) {
            device->device.resetEvent(event);
        }
        vkev->events_free.insert(vkev->events_free.end(), vkev->events_submitted.begin(), vkev->events_submitted.end());
        vkev->events_submitted.सफा();

        // Finished using current command buffer so we flag for reuse
        अगर (vkev->cmd_बफर) {
            // Only flag for reuse if it hasn't been reused already
            अगर (vkev->cmd_बफर_use_गणक == vkev->cmd_बफर->use_गणक) {
                vkev->cmd_बफर->in_use = false;
                vkev->cmd_बफर->buf.reset();
            }
            vkev->cmd_बफर = nullptr;
        }
    }
}

अटल ggml_backend_बफर_t ggml_backend_vk_device_बफर_from_host_ptr(ggml_backend_dev_t dev, व्योम * ptr, माप_प्रकार size, माप_प्रकार max_tensor_आकार) {
    VK_LOG_DEBUG("ggml_backend_vk_device_buffer_from_host_ptr(backend=" << dev << ", ptr=" << ptr << ", size=" << size << ")");
    GGML_UNUSED(max_tensor_आकार);

    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    स्वतः device = ggml_vk_get_device(ctx->device);

    vk_बफर buf = ggml_vk_बफर_from_host_ptr(device, ptr, size);

    अगर (!buf) {
        वापस {};
    }

    ggml_backend_vk_बफर_context * bufctx = नया ggml_backend_vk_बफर_context(device, std::move(buf), device->name);

    ggml_backend_बफर_t ret = ggml_backend_बफर_init(ggml_backend_vk_device_get_बफर_type(dev), ggml_backend_vk_बफर_interface, bufctx, size);

    वापस ret;
}

अटल स्थिर काष्ठा ggml_backend_device_i ggml_backend_vk_device_i = {
    /* .get_name             = */ ggml_backend_vk_device_get_name,
    /* .get_description      = */ ggml_backend_vk_device_get_description,
    /* .get_memory           = */ ggml_backend_vk_device_get_memory,
    /* .get_type             = */ ggml_backend_vk_device_get_type,
    /* .get_props            = */ ggml_backend_vk_device_get_props,
    /* .init_backend         = */ ggml_backend_vk_device_init,
    /* .get_buffer_type      = */ ggml_backend_vk_device_get_बफर_type,
    /* .get_host_buffer_type = */ ggml_backend_vk_device_get_host_बफर_type,
    /* .buffer_from_host_ptr = */ ggml_backend_vk_device_बफर_from_host_ptr,
    /* .supports_op          = */ ggml_backend_vk_device_supports_op,
    /* .supports_buft        = */ ggml_backend_vk_device_supports_buft,
    /* .offload_op           = */ ggml_backend_vk_device_offload_op,
    /* .event_new            = */ ggml_backend_vk_device_event_new,
    /* .event_free           = */ ggml_backend_vk_device_event_free,
    /* .event_synchronize    = */ ggml_backend_vk_device_event_synchronize,
};

अटल स्थिर अक्षर * ggml_backend_vk_reg_get_name(ggml_backend_reg_t reg) {
    UNUSED(reg);
    वापस GGML_VK_NAME;
}

अटल माप_प्रकार ggml_backend_vk_reg_get_device_गिनती(ggml_backend_reg_t reg) {
    UNUSED(reg);
    वापस ggml_backend_vk_get_device_गिनती();
}

अटल ggml_backend_dev_t ggml_backend_vk_reg_get_device(ggml_backend_reg_t reg, माप_प्रकार device) {
    अटल std::vector<ggml_backend_dev_t> devices;

    अटल bool initialized = false;

    {
        अटल std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        अगर (!initialized) {
            स्थिर पूर्णांक min_batch_आकार = दो_पर्या("GGML_OP_OFFLOAD_MIN_BATCH") ? म_से_प(दो_पर्या("GGML_OP_OFFLOAD_MIN_BATCH")) : 32;
            क्रम (पूर्णांक i = 0; i < ggml_backend_vk_get_device_गिनती(); i++) {
                ggml_backend_vk_device_context * ctx = नया ggml_backend_vk_device_context;
                अक्षर desc[256];
                ggml_backend_vk_get_device_description(i, desc, माप(desc));
                ctx->device = i;
                ctx->name = GGML_VK_NAME + std::to_string(i);
                ctx->description = desc;
                ctx->is_integrated_gpu = ggml_backend_vk_get_device_type(i) == vk::PhysicalDeviceType::eIntegratedGpu;
                ctx->pci_bus_id = ggml_backend_vk_get_device_pci_id(i);
                ctx->op_offload_min_batch_आकार = min_batch_आकार;
                devices.push_back(नया ggml_backend_device {
                    /* .iface   = */ ggml_backend_vk_device_i,
                    /* .reg     = */ reg,
                    /* .context = */ ctx,
                });
            }
            initialized = true;
        }
    }

    GGML_ASSERT(device < devices.size());
    वापस devices[device];
}

अटल स्थिर काष्ठा ggml_backend_reg_i ggml_backend_vk_reg_i = {
    /* .get_name         = */ ggml_backend_vk_reg_get_name,
    /* .get_device_count = */ ggml_backend_vk_reg_get_device_गिनती,
    /* .get_device       = */ ggml_backend_vk_reg_get_device,
    /* .get_proc_address = */ शून्य,
};

ggml_backend_reg_t ggml_backend_vk_reg() {
    अटल ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_vk_reg_i,
        /* .context     = */ nullptr,
    };
    प्रयास {
        ggml_vk_instance_init();
        वापस &reg;
    } पकड़ो (स्थिर vk::SystemError& e) {
        VK_LOG_DEBUG("ggml_backend_vk_reg() -> Error: System error: " << e.what());
        वापस nullptr;
    } पकड़ो (स्थिर std::exception &e) {
        VK_LOG_DEBUG("ggml_backend_vk_reg() -> Error: " << e.what());
        वापस nullptr;
    } पकड़ो (...) {
        VK_LOG_DEBUG("ggml_backend_vk_reg() -> Error: unknown exception during Vulkan init");
        वापस nullptr;
    }
}

bool ggml_vk_instance_layer_settings_available() {
#अगर_घोषित GGML_VULKAN_VALIDATE
    // Check if validation layer provides the extension
    स्थिर std::string layer_name = "VK_LAYER_KHRONOS_validation";
    क्रम (स्थिर स्वतः& layer : vk::enumerateInstanceLayerProperties()) {
        अगर (layer_name == layer.layerName.तथ्य()) {
            क्रम (स्थिर स्वतः& ext : vk::enumerateInstanceExtensionProperties(layer_name)) {
                अगर (म_भेद("VK_EXT_layer_settings", ext.extensionName.तथ्य()) == 0) {
                    वापस true;
                }
            }
        }
    }

    std::अ_त्रुटि << "ggml_vulkan: WARNING: Validation layer or layer extension VK_EXT_layer_settings not found." << std::प_पूर्ण;
#पूर्ण_अगर
    वापस false;
}

bool ggml_vk_instance_portability_enumeration_ext_available(स्थिर std::vector<vk::ExtensionProperties>& instance_extensions) {
#अगर_घोषित __APPLE__
    // Check for portability enumeration extension for MoltenVK support
    क्रम (स्थिर स्वतः& properties : instance_extensions) {
        अगर (म_भेद("VK_KHR_portability_enumeration", properties.extensionName) == 0) {
            वापस true;
        }
    }
    std::अ_त्रुटि << "ggml_vulkan: WARNING: Instance extension VK_KHR_portability_enumeration not found." << std::प_पूर्ण;
#पूर्ण_अगर
    वापस false;

    UNUSED(instance_extensions);
}

bool ggml_vk_instance_debug_utils_ext_available(
    स्थिर std::vector<vk::ExtensionProperties> & instance_extensions) {
    // Check for portability enumeration extension for MoltenVK support
    क्रम (स्थिर स्वतः & properties : instance_extensions) {
        अगर (म_भेद("VK_EXT_debug_utils", properties.extensionName) == 0) {
            वापस true;
        }
    }

    std::अ_त्रुटि << "ggml_vulkan: WARNING: Instance extension VK_EXT_debug_utils not found." << std::प_पूर्ण;
    वापस false;

    UNUSED(instance_extensions);
}

bool ggml_vk_device_is_supported(स्थिर vk::PhysicalDevice & vkdev) {
    VkPhysicalDeviceFeatures2 device_features2;
    device_features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;

    VkPhysicalDeviceVulkan11Features vk11_features;
    vk11_features.p_अग्र = nullptr;
    vk11_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    device_features2.p_अग्र = &vk11_features;

    vkGetPhysicalDeviceFeatures2(vkdev, &device_features2);

    वापस vk11_features.storageBuffer16BitAccess;
}

bool ggml_vk_khr_cooperative_matrix_support(स्थिर vk::PhysicalDeviceProperties& props, स्थिर vk::PhysicalDeviceDriverProperties& driver_props, vk_device_architecture arch) {
    चयन (props.vendorID) {
    हाल VK_VENDOR_ID_INTEL:
        // Only allowing Xe2/Xe3 GPU and integrated Xe GPUs at the moment since older hardware (ex. Arc A770) has performance regressions.
        वापस (arch == vk_device_architecture::INTEL_XE2) ||
            (arch == vk_device_architecture::INTEL_XE1 && props.deviceType == vk::PhysicalDeviceType::eIntegratedGpu && driver_props.driverID == vk::DriverId::eIntelProprietaryWindows);
    हाल VK_VENDOR_ID_AMD:
        अगर (driver_props.driverID == vk::DriverId::eAmdProprietary || driver_props.driverID == vk::DriverId::eAmdOpenSource) {
            // Workaround for AMD proprietary driver reporting support on all GPUs
            वापस arch == vk_device_architecture::AMD_RDNA3 || arch == vk_device_architecture::AMD_RDNA4;
        }
        वापस true;
    हाल VK_VENDOR_ID_QUALCOMM:
        // Only allow Adreno GPUs with hardware matrix cores (Gen 6+).
        वापस arch == vk_device_architecture::QUALCOMM_ADRENO;
    शेष:
        वापस true;
    }
}

uint32_t ggml_vk_intel_shader_core_गिनती(स्थिर vk::PhysicalDevice& vkdev) {
    VkPhysicalDeviceProperties2 props = vkdev.getProperties2();

    अगर (props.properties.vendorID != VK_VENDOR_ID_INTEL) {
        वापस 0;
    }

    स्थिर uint32_t device_id = props.properties.deviceID;

    चयन (device_id) {
    हाल 0x56A6:  // A310
        वापस 6;
    हाल 0x5693:  // A370M
    हाल 0x56A5:  // A380
    हाल 0x56B1:  // Pro A40/A50
        वापस 8;
    हाल 0x5697:  // A530M
        वापस 12;
    हाल 0x5692:  // A550M
    हाल 0x56B3:  // Pro A60
        वापस 16;
    हाल 0x56A2:  // A580
        वापस 24;
    हाल 0x5691:  // A730M
    हाल 0x56A1:  // A750
        वापस 28;
    हाल 0x56A0:  // A770
    हाल 0x5690:  // A770M
        वापस 32;
    हाल 0xE212:  // Pro B50
        वापस 16;
    हाल 0xE20C:  // B570
        वापस 18;
    हाल 0xE20B:  // B580
    हाल 0xE211:  // Pro B60
    हाल 0xE222:  // Pro B65
        वापस 20;
    हाल 0xE223:  // Pro B70
        वापस 32;
    हाल 0xB080:  // PTL Xe3 LPG 2x6 (12 subslices)
        वापस 12;
    शेष:
        वापस 0;
    }
}

bool ggml_vk_intel_windows_driver_in_range(uint32_t driver_version, uint32_t lower_major, uint32_t lower_minor, uint32_t upper_major, uint32_t upper_minor) {
#अगर defined(_WIN32)
    // Intel Windows encodes xxx.yyyy as [31:14].[13:0].
    स्थिर uint32_t major = driver_version >> 14;
    स्थिर uint32_t minor = driver_version & 0x3fff;

    स्थिर bool ge_lower = major > lower_major || (major == lower_major && minor >= lower_minor);
    स्थिर bool lt_upper = major < upper_major || (major == upper_major && minor < upper_minor);

    वापस ge_lower && lt_upper;
#अन्यथा
    GGML_UNUSED(driver_version);
    GGML_UNUSED(lower_major);
    GGML_UNUSED(lower_minor);
    GGML_UNUSED(upper_major);
    GGML_UNUSED(upper_minor);
    वापस true;
#पूर्ण_अगर
}

GGML_BACKEND_DL_IMPL(ggml_backend_vk_reg)


// out-of-lined header method definitions

व्योम vk_queue_handle_synchronized::submit(vk::ArrayProxy<स्थिर vk::SubmitInfo> submits, vk::Fence fence) {
    // Workaround for NVIDIA driver bug
    std::unique_lock<std::mutex> device_guard;
    अगर (device_submit_mutex) {
        device_guard = std::unique_lock<std::mutex>(*device_submit_mutex);
    }
    std::lock_guard<std::mutex> guard(mutex);
    प्रयास {
        queue.submit(submits, fence);
    } पकड़ो (vk::DeviceLostError &) {
        अगर (स्वतः dev = device.lock()) {
            ggml_vk_print_device_lost_info(dev);
        }
        फेंको;
    }
}

व्योम vk_queue_handle_unsynchronized::submit(vk::ArrayProxy<स्थिर vk::SubmitInfo> submits, vk::Fence fence) {
    // Workaround for NVIDIA driver bug
    std::unique_lock<std::mutex> device_guard;
    अगर (device_submit_mutex) {
        device_guard = std::unique_lock<std::mutex>(*device_submit_mutex);
    }
    प्रयास {
        queue.submit(submits, fence);
    } पकड़ो (vk::DeviceLostError &) {
        अगर (स्वतः dev = device.lock()) {
            ggml_vk_print_device_lost_info(dev);
        }
        फेंको;
    }
}

vk_device_struct::~vk_device_struct() {
    // the build failed before the device was created, nothing to destroy
    अगर (device == VK_NULL_HANDLE) {
        वापस;
    }

    VK_LOG_DEBUG("destroy device " << name);

    device.destroyFence(fence);

    ggml_vk_destroy_बफर(sync_staging);

    अगर (compute_queue) compute_queue->cmd_pool.destroy(device);
    अगर (transfer_queue) transfer_queue->cmd_pool.destroy(device);

    // Explicitly clear to ensure queues drop their shared_ptrs to handles
    // before the Vulkan logical device instance is destroyed
    compute_queue.reset();
    transfer_queue.reset();

    क्रम (स्वतः& pipeline : all_pipelines) {
        अगर (pipeline.expired()) {
            जारी;
        }

        vk_pipeline pl = pipeline.lock();
        ggml_vk_destroy_pipeline(device, pl);
    }
    all_pipelines.सफा();

    device.destroyDescriptorSetLayout(dsl);

    device.destroy();
}

व्योम vk_perf_logger::print_timings(bool force) {
    अगर (timings.empty()) {
        वापस;
    }
    print_गिनती++;
    अगर ((print_गिनती % vk_perf_logger_frequency) != 0 && !force) {
        वापस;
    }
    print_गिनती = 0;
    uint64_t योग_all_op_times = 0;
    std::अ_त्रुटि << "----------------\nVulkan Timings:" << std::प_पूर्ण;
    क्रम (स्थिर स्वतः & t : timings) {
        uint64_t योग_op_times = 0;
        क्रम (स्थिर स्वतः & समय : t.second) {
            योग_op_times += समय;
        }
        std::अ_त्रुटि << t.first << ": " << t.second.size() << " x " << (योग_op_times / t.second.size() / 1000.0)
                  << " us = " << (योग_op_times / 1000.0) << " us";

        // If we have as many flops entries as timing entries for the op, then compute and log the flops/S.
        स्वतः it = flops.find(t.first);
        अगर (it != flops.end() && (it->second).size() == t.second.size()) {
            uint64_t योग_op_flops = 0;
            क्रम (स्थिर स्वतः & elem : it->second) {
                योग_op_flops += elem;
            }
            std::अ_त्रुटि << " ("
                      << (द्विगुन(योग_op_flops) / (1000.0 * 1000.0 * 1000.0)) /
                             (द्विगुन(योग_op_times) / (1000.0 * 1000.0 * 1000.0))
                      << " GFLOPS/s)";
        }

        योग_all_op_times += योग_op_times;

        std::अ_त्रुटि << std::प_पूर्ण;
    }

    अगर (timings.size() > 0) {
        std::अ_त्रुटि << "Total time: " << योग_all_op_times / 1000.0 << " us." << std::प_पूर्ण;
    }

    timings.सफा();
    flops.सफा();
}

std::string vk_perf_logger::get_गाँठ_fusion_name(स्थिर ggml_tensor * गाँठ, स्थिर अक्षर *fusion_name, uint64_t *n_flops) {
    *n_flops = ggml_vk_get_गाँठ_flops(गाँठ);
    std::string fusion_str;
    अगर (fusion_name) {
        fusion_str = fusion_name + std::string(" ");
    }
    अगर (गाँठ->op == GGML_OP_UNARY) {
        वापस fusion_str + ggml_unary_op_name(ggml_get_unary_op(गाँठ));
    }
    अगर (गाँठ->op == GGML_OP_MUL_MAT || गाँठ->op == GGML_OP_MUL_MAT_ID) {
        स्थिर uint64_t m     = गाँठ->ne[0];
        स्थिर uint64_t n     = गाँठ->ne[1];
        स्थिर uint64_t k     = गाँठ->src[1]->ne[0];
        स्थिर uint64_t batch = गाँठ->ne[2] * गाँठ->ne[3];
        std::string    name  = ggml_op_name(गाँठ->op);
        अगर ((गाँठ->op == GGML_OP_MUL_MAT && n <= mul_mat_vec_max_cols) ||
            (गाँठ->op == GGML_OP_MUL_MAT_ID && गाँठ->src[2]->ne[1] == 1)) {
            name += "_VEC";
        }
        name += " ";
        name += ggml_type_name(गाँठ->src[0]->type);
        name += " m=" + std::to_string(m) + " n=" + std::to_string(n) + " k=" + std::to_string(k);
        अगर (गाँठ->op == GGML_OP_MUL_MAT_ID) {
            name += " n_expert=" + std::to_string(गाँठ->src[0]->ne[2]);
        }
        अगर (batch > 1) {
            name += " batch=" + std::to_string(batch);
        }
        वापस fusion_str + name;
    }
    अगर (गाँठ->op == GGML_OP_CONV_2D || गाँठ->op == GGML_OP_CONV_TRANSPOSE_2D) {
        std::string   name    = ggml_op_name(गाँठ->op);
        स्थिर ggml_tensor * knl = गाँठ->src[0];
        uint64_t      Cout    = गाँठ->ne[2];
        uint64_t      आकार_K  = गाँठ->src[1]->ne[2] * knl->ne[0] * knl->ne[1];
        uint64_t      आकार_N_2  = गाँठ->ne[3] * गाँठ->ne[0] * गाँठ->ne[1];
        name += " M=Cout=" + std::to_string(Cout) + ", K=Cin*KW*KH=" + std::to_string(आकार_K) +
                ", N=N*OW*OH=" + std::to_string(आकार_N_2);
        वापस fusion_str + name;
    }
    अगर (गाँठ->op == GGML_OP_RMS_NORM) {
        std::string   name    = ggml_op_name(गाँठ->op);
        name += "(" + std::to_string(गाँठ->ne[0]) + "," + std::to_string(गाँठ->ne[1]) + "," + std::to_string(गाँठ->ne[2]) + "," + std::to_string(गाँठ->ne[3]) + ")";
        वापस fusion_str + name;
    }
    अगर (गाँठ->op == GGML_OP_FLASH_ATTN_EXT) {
        स्थिर ggml_tensor * dst = गाँठ;
        स्थिर ggml_tensor * q = गाँठ->src[0];
        स्थिर ggml_tensor * k = गाँठ->src[1];
        स्थिर ggml_tensor * v = गाँठ->src[2];
        स्थिर ggml_tensor * m = गाँठ->src[3];
        std::stringstream name;
        name << fusion_str;
        name << ggml_op_name(गाँठ->op) <<
            " dst(" << dst->ne[0] << "," << dst->ne[1] << "," << dst->ne[2] << "," << dst->ne[3] << "), " <<
            " q(" << q->ne[0] << "," << q->ne[1] << "," << q->ne[2] << "," << q->ne[3] << "), " <<
            " k(" << k->ne[0] << "," << k->ne[1] << "," << k->ne[2] << "," << k->ne[3] << "), " <<
            " v(" << v->ne[0] << "," << v->ne[1] << "," << v->ne[2] << "," << v->ne[3] << "), " <<
            " m(" << (m?m->ne[0]:0) << "," << (m?m->ne[1]:0) << "," << (m?m->ne[2]:0) << "," << (m?m->ne[3]:0) << ")";
        वापस name.str();
    }
    अगर (गाँठ->op == GGML_OP_शीर्ष_K) {
        std::stringstream name;
        name << fusion_str;
        name << ggml_op_name(गाँठ->op) <<
            " K=" << गाँठ->ne[0] <<
            " (" << गाँठ->src[0]->ne[0] << "," << गाँठ->src[0]->ne[1] << "," << गाँठ->src[0]->ne[2] << "," << गाँठ->src[0]->ne[3] << ")";
        वापस name.str();
    }
    वापस fusion_str + ggml_op_name(गाँठ->op);
}

ggml_backend_vk_बफर_context::~ggml_backend_vk_बफर_context() {
    ggml_vk_destroy_बफर(dev_बफर);
}

ggml_vk_debug_label::ggml_vk_debug_label(vk_context & ctx, स्थिर std::string & pipeline_name, uint32_t wg0, uint32_t wg1, uint32_t wg2) {
    अगर (!vk_instance.debug_utils_support || ctx->s == nullptr) {
        वापस;
    }
    begin(ctx, pipeline_name + " (" + std::to_string(wg0) + "," + std::to_string(wg1) + "," + std::to_string(wg2) + ")");
}

ggml_vk_debug_label::ggml_vk_debug_label(vk_context & ctx, स्थिर ggml_cgraph * cgraph, पूर्णांक गाँठ_idx, पूर्णांक n_fused) {
    अगर (!vk_instance.debug_utils_support || ctx->s == nullptr) {
        वापस;
    }
    std::string name = ggml_op_name(cgraph->nodes[गाँठ_idx]->op);
    क्रम (पूर्णांक i = 1; i <= n_fused; i++) {
        name += "+";
        name += ggml_op_name(cgraph->nodes[गाँठ_idx + i]->op);
    }
    name += " ";
    name += cgraph->nodes[गाँठ_idx]->name;
    begin(ctx, name);
}

ggml_vk_debug_label::ggml_vk_debug_label(vk_queue_handle * handle, स्थिर अक्षर * name) {
    अगर (!vk_instance.debug_utils_support || handle == nullptr) {
        वापस;
    }
    vk::DebugUtilsLabelEXT label = {};
    label.pLabelName = name;
    label.color = std::array<भग्न, 4>{1.0f, 1.0f, 1.0f, 1.0f};

    qhandle = handle;
    std::lock_guard<vk_queue_handle> guard(*qhandle);
    vk_instance.pfn_vkQueueBeginDebugUtilsLabelEXT(qhandle->queue, reinterpret_cast<VkDebugUtilsLabelEXT *>(&label));
}

व्योम ggml_vk_debug_label::बंद() {
    अगर (subctx != nullptr) {
        // close on the current command buffer, which may differ from the one begin used
        अगर (subctx->s != nullptr) {
            vk_instance.pfn_vkCmdEndDebugUtilsLabelEXT(subctx->s->बफर->buf);
        }
        subctx->debug_labels.pop_back();
        subctx = nullptr;
    }
    अगर (qhandle != nullptr) {
        std::lock_guard<vk_queue_handle> guard(*qhandle);
        vk_instance.pfn_vkQueueEndDebugUtilsLabelEXT(qhandle->queue);
        qhandle = nullptr;
    }
}

व्योम ggml_vk_debug_label::begin(vk_context & ctx, स्थिर std::string & name) {
    अगर (!vk_instance.debug_utils_support || ctx->s == nullptr) {
        वापस;
    }
    subctx = ctx.लो();
    subctx->debug_labels.push_back(name);
    ggml_vk_cmd_label_begin(subctx->s->बफर->buf, subctx->debug_labels.back().c_str());
}

