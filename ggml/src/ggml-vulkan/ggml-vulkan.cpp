#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-vulkan-common.h"

ਸਥਿਰ vk_device_architecture get_device_architecture(ਅਚਲ vk::PhysicalDevice& device) {
    vk::PhysicalDeviceProperties props = device.getProperties();

    ਜੇ (props.vendorID == VK_VENDOR_ID_AMD) {
        ਅਚਲ std::vector<vk::ExtensionProperties> ext_props = device.enumerateDeviceExtensionProperties();

        ਤਾਰਕਿਕ amd_shader_core_properties = ਝੂਠ;
        ਤਾਰਕਿਕ integer_dot_ਗੁਣਨਫਲ = ਝੂਠ;
        ਤਾਰਕਿਕ subgroup_ਮਾਪ_control = ਝੂਠ;
        ਤਾਰਕਿਕ shader_float8 = ਝੂਠ;

        ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ& properties : ext_props) {
            ਜੇ (strcmp("VK_AMD_shader_core_properties", properties.extensionName) == 0) {
                amd_shader_core_properties = ਸੱਚ;
            } ਨਹੀਂਤਾਂ ਜੇ (strcmp("VK_KHR_shader_integer_dot_product", properties.extensionName) == 0) {
                integer_dot_ਗੁਣਨਫਲ = ਸੱਚ;
            } ਨਹੀਂਤਾਂ ਜੇ (strcmp("VK_EXT_subgroup_size_control", properties.extensionName) == 0) {
                subgroup_ਮਾਪ_control = ਸੱਚ;
            } ਨਹੀਂਤਾਂ ਜੇ (strcmp("VK_EXT_shader_float8", properties.extensionName) == 0) {
                shader_float8 = ਸੱਚ;
            }
        }

        ਜੇ (!amd_shader_core_properties || !integer_dot_ਗੁਣਨਫਲ || !subgroup_ਮਾਪ_control) {
            ਵਾਪਸ vk_device_architecture::OTHER;
        }

        vk::PhysicalDeviceProperties2 props2;
        vk::PhysicalDeviceShaderCorePropertiesAMD shader_core_props_amd;
        vk::Physical_Device_Shader_Integer_Dot_ਗੁਣਨਫਲ_Properties_KHR integer_dot_props;
        vk::Physical_Device_Subgroup_ਮਾਪ_Control_Properties_EXT subgroup_ਮਾਪ_control_props;

        props2.p_ਅਗਲਾ = &shader_core_props_amd;
        shader_core_props_amd.p_ਅਗਲਾ = &integer_dot_props;
        integer_dot_props.p_ਅਗਲਾ = &subgroup_ਮਾਪ_control_props;

        device.getProperties2(&props2);

        ਜੇ (subgroup_ਮਾਪ_control_props.max_Subgroup_ਮਾਪ == 64 && subgroup_ਮਾਪ_control_props.min_Subgroup_ਮਾਪ == 64) {
            ਵਾਪਸ vk_device_architecture::AMD_GCN;
        }
        ਜੇ (subgroup_ਮਾਪ_control_props.max_Subgroup_ਮਾਪ == 64 && subgroup_ਮਾਪ_control_props.min_Subgroup_ਮਾਪ == 32) {
            // RDNA
            ਜੇ (shader_core_props_amd.wavefronts_ਪ੍ਰਤੀ_Simd == 20) {
                ਵਾਪਸ vk_device_architecture::AMD_RDNA1;
            }
            ਜੇ (shader_float8) {
                ਵਾਪਸ vk_device_architecture::AMD_RDNA4;
            }
            ਜੇ (integer_dot_props.integer_Dot_Product4x8_ਦੋਆਧਾਰੀ_ਅੰਕ_Packed_Mixed_Signedness_Accelerated) {
                ਵਾਪਸ vk_device_architecture::AMD_RDNA3;
            }
            ਵਾਪਸ vk_device_architecture::AMD_RDNA2;
        }
    } ਨਹੀਂਤਾਂ ਜੇ (props.vendorID == VK_VENDOR_ID_INTEL) {
        ਅਚਲ std::vector<vk::ExtensionProperties> ext_props = device.enumerateDeviceExtensionProperties();

        ਤਾਰਕਿਕ subgroup_ਮਾਪ_control = ਝੂਠ;
        ਤਾਰਕਿਕ integer_dot_ਗੁਣਨਫਲ = ਝੂਠ;

        ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ& properties : ext_props) {
            ਜੇ (strcmp("VK_EXT_subgroup_size_control", properties.extensionName) == 0) {
                subgroup_ਮਾਪ_control = ਸੱਚ;
            } ਨਹੀਂਤਾਂ ਜੇ (strcmp("VK_KHR_shader_integer_dot_product", properties.extensionName) == 0) {
                integer_dot_ਗੁਣਨਫਲ = ਸੱਚ;
            }
        }

        ਜੇ (!subgroup_ਮਾਪ_control || !integer_dot_ਗੁਣਨਫਲ) {
            ਵਾਪਸ vk_device_architecture::OTHER;
        }

        vk::PhysicalDeviceProperties2 props2;
        vk::Physical_Device_Subgroup_ਮਾਪ_Control_Properties_EXT subgroup_ਮਾਪ_control_props;
        vk::Physical_Device_Shader_Integer_Dot_ਗੁਣਨਫਲ_Properties_KHR integer_dot_props;

        props2.p_ਅਗਲਾ = &subgroup_ਮਾਪ_control_props;
        subgroup_ਮਾਪ_control_props.p_ਅਗਲਾ = &integer_dot_props;
        device.getProperties2(&props2);

        ਜੇ (subgroup_ਮਾਪ_control_props.min_Subgroup_ਮਾਪ == 16) {
            // Xe2 architecture uses SIMD16 while previous Xe and Gen architecture uses SIMD8.
            // Minimum subgroup size matches the SIMD width so we distinguish architecture by checking this value.
            // https://www.intel.com/content/www/us/en/content-details/824434/2024-intel-tech-tour-xe2-and-lunar-lake-s-gpu.html
            // https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-0/intel-xe-gpu-architecture.html
            ਵਾਪਸ vk_device_architecture::INTEL_XE2;
        } ਨਹੀਂਤਾਂ ਜੇ (subgroup_ਮਾਪ_control_props.min_Subgroup_ਮਾਪ == 8 &&
                 integer_dot_ਗੁਣਨਫਲ && integer_dot_props.integer_Dot_Product4x8_ਦੋਆਧਾਰੀ_ਅੰਕ_Packed_Signed_Accelerated) {
            ਵਾਪਸ vk_device_architecture::INTEL_XE1;
        }
    } ਨਹੀਂਤਾਂ ਜੇ (props.vendorID == VK_VENDOR_ID_NVIDIA) {
        ਅਚਲ std::vector<vk::ExtensionProperties> ext_props = device.enumerateDeviceExtensionProperties();

        ਤਾਰਕਿਕ cooperative_ਮੈਟ੍ਰਿਕਸ = ਝੂਠ;
        ਤਾਰਕਿਕ sm_builtins = ਝੂਠ;

        // Detect "pre-turing" based on lack of coopmat support.
        ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ& properties : ext_props) {
            ਜੇ (strcmp("VK_KHR_cooperative_matrix", properties.extensionName) == 0) {
                cooperative_ਮੈਟ੍ਰਿਕਸ = ਸੱਚ;
            } ਨਹੀਂਤਾਂ ਜੇ (strcmp("VK_NV_shader_sm_builtins", properties.extensionName) == 0) {
                sm_builtins = ਸੱਚ;
            }
        }

        ਜੇ (!cooperative_ਮੈਟ੍ਰਿਕਸ) {
            ਵਾਪਸ vk_device_architecture::NVIDIA_PRE_TURING;
        }

        ਜੇ (sm_builtins) {
            vk::PhysicalDeviceProperties2 props2;
            vk::PhysicalDeviceShaderSMBuiltinsPropertiesNV sm_props;

            props2.p_ਅਗਲਾ = &sm_props;

            device.getProperties2(&props2);

            // Turing has 32, following architectures have 48
            ਜੇ (sm_props.shader_Warps_ਪ੍ਰਤੀ_SM == 32) {
                ਵਾਪਸ vk_device_architecture::NVIDIA_TURING;
            }
        }
    } ਨਹੀਂਤਾਂ ਜੇ(props.vendorID == VK_VENDOR_ID_QUALCOMM){
        ਅਚਲ std::vector<vk::ExtensionProperties> ext_props = device.enumerateDeviceExtensionProperties();

        ਤਾਰਕਿਕ cooperative_ਮੈਟ੍ਰਿਕਸ = ਝੂਠ;
        ਤਾਰਕਿਕ cooperative_ਮੈਟ੍ਰਿਕਸ_conversion = ਝੂਠ;

        ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ& properties : ext_props) {
            ਜੇ (strcmp("VK_KHR_cooperative_matrix", properties.extensionName) == 0) {
                cooperative_ਮੈਟ੍ਰਿਕਸ = ਸੱਚ;
            } ਨਹੀਂਤਾਂ ਜੇ (strcmp("VK_QCOM_cooperative_matrix_conversion", properties.extensionName) == 0) {
                cooperative_ਮੈਟ੍ਰਿਕਸ_conversion = ਸੱਚ;
            }
        }

        ਜੇ (cooperative_ਮੈਟ੍ਰਿਕਸ && cooperative_ਮੈਟ੍ਰਿਕਸ_conversion) {
            ਵਾਪਸ vk_device_architecture::QUALCOMM_ADRENO;
        }
    }
    ਵਾਪਸ vk_device_architecture::OTHER;
}

ਤਾਰਕਿਕ ggml_vk_lightning_indexer_k_type_supported(ggml_type type) {
    ਵਾਪਸ std::find(lightning_indexer_k_types.begin(), lightning_indexer_k_types.end(), type) != lightning_indexer_k_types.end();
}
ਖਾਲੀ ggml_vk_print_device_fault_info(ਅਚਲ vk_device& device) {
    ਜੇ (!device->device_fault || !device->pfn_vkGetDeviceFaultInfoEXT) {
        ਵਾਪਸ;
    }

    VkDeviceFaultCountsEXT fault_counts {};
    fault_counts.sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT;
    Vk_ਨਤੀਜਾ res = device->pfn_vkGetDeviceFaultInfoEXT(device->device, &fault_counts, ਖਾਲੀ_ਸੂਚਕ);
    ਜੇ (res != VK_SUCCESS) {
        GGML_LOG_ERROR("ggml_vulkan: vkGetDeviceFaultInfoEXT (counts) failed: %d\n", res);
        ਵਾਪਸ;
    }

    std::vector<Vk_Device_Fault_ਪਤਾ_Info_EXT> ਪਤਾ_infos(fault_counts.ਪਤਾ_Info_ਗਿਣਤੀ);
    std::vector<VkDeviceFaultVendorInfoEXT> vendor_infos(fault_counts.vendor_Info_ਗਿਣਤੀ);

    VkDeviceFaultInfoEXT fault_info {};
    fault_info.sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT;
    fault_info.p_ਪਤਾ_Infos = ਪਤਾ_infos.ਅੰਕੜੇ();
    fault_info.pVendorInfos = vendor_infos.ਅੰਕੜੇ();

    res = device->pfn_vkGetDeviceFaultInfoEXT(device->device, &fault_counts, &fault_info);
    ਜੇ (res != VK_SUCCESS) {
        GGML_LOG_ERROR("ggml_vulkan: vkGetDeviceFaultInfoEXT (info) failed: %d\n", res);
        ਵਾਪਸ;
    }

    ਜੇ (fault_counts.ਪਤਾ_Info_ਗਿਣਤੀ == 0 && fault_counts.vendor_Info_ਗਿਣਤੀ == 0 && fault_info.description[0] == '\0') {
        ਵਾਪਸ;
    }

    ਜੇ (fault_info.description[0] != '\0') {
        GGML_LOG_ERROR("ggml_vulkan: device fault on %s: %s\n", device->name.c_str(), fault_info.description);
    }

    ਲਈ (uint32_t i = 0; i < fault_counts.ਪਤਾ_Info_ਗਿਣਤੀ; i++) {
        ਅਚਲ ਸਵੈਚਾਲਿਤ& info = ਪਤਾ_infos[i];
        GGML_LOG_CONT("  address fault %u: type=%d address=0x%llx precision=0x%llx\n",
                i, (ਪੂਰਨ)info.ਪਤਾ_Type,
                (ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਲੰਮਾ)info.reported_ਪਤਾ,
                (ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਲੰਮਾ)info.ਪਤਾ_Precision);
    }
    ਲਈ (uint32_t i = 0; i < fault_counts.vendor_Info_ਗਿਣਤੀ; i++) {
        ਅਚਲ ਸਵੈਚਾਲਿਤ& info = vendor_infos[i];
        GGML_LOG_CONT("  vendor fault %u: %s (code=0x%llx data=0x%llx)\n",
                i, info.description,
                (ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਲੰਮਾ)info.vendor_Fault_ਕੋਡ,
                (ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਲੰਮਾ)info.vendor_Fault_ਅੰਕੜੇ);
    }
}
uint64_t ggml_vk_get_ਗੰਢ_flops(ਅਚਲ ggml_tensor * ਗੰਢ) {
    ਜੇ (ਗੰਢ->op == GGML_OP_MUL_MAT || ਗੰਢ->op == GGML_OP_MUL_MAT_ID) {
        ਅਚਲ uint64_t m     = ਗੰਢ->ne[0];
        ਅਚਲ uint64_t n     = ਗੰਢ->ne[1];
        ਅਚਲ uint64_t k     = ਗੰਢ->src[1]->ne[0];
        ਅਚਲ uint64_t batch = ਗੰਢ->ne[2] * ਗੰਢ->ne[3];
        ਵਾਪਸ m * n * (k + (k - 1)) * batch;
    }
    ਜੇ (ਗੰਢ->op == GGML_OP_CONV_2D || ਗੰਢ->op == GGML_OP_CONV_TRANSPOSE_2D) {
        ਅਚਲ ggml_tensor * knl = ਗੰਢ->src[0];
        ਅਚਲ uint64_t Cout  = ਗੰਢ->ne[2];
        ਅਚਲ uint64_t ਮਾਪ_K = ਗੰਢ->src[1]->ne[2] * knl->ne[0] * knl->ne[1];
        ਅਚਲ uint64_t ਮਾਪ_N_2 = ਗੰਢ->ne[3] * ਗੰਢ->ne[0] * ਗੰਢ->ne[1];
        ਵਾਪਸ Cout * ਮਾਪ_N_2 * (ਮਾਪ_K + (ਮਾਪ_K - 1));
    }
    ਜੇ (ਗੰਢ->op == GGML_OP_CONV_3D) {
        ਅਚਲ ggml_tensor * knl = ਗੰਢ->src[0];
        ਅਚਲ uint64_t OC     = ggml_get_op_params_i32(ਗੰਢ, 11);
        ਅਚਲ uint64_t IC     = ggml_get_op_params_i32(ਗੰਢ, 9);
        ਅਚਲ uint64_t ਮਾਪ_K = IC * knl->ne[0] * knl->ne[1] * knl->ne[2];
        ਅਚਲ uint64_t ਮਾਪ_N_2 = ਗੰਢ->ne[3] / OC * ਗੰਢ->ne[0] * ਗੰਢ->ne[1] * ਗੰਢ->ne[2];
        ਵਾਪਸ OC * ਮਾਪ_N_2 * (ਮਾਪ_K + (ਮਾਪ_K - 1));
    }
    ਜੇ (ਗੰਢ->op == GGML_OP_FLASH_ATTN_EXT) {
        ਅਚਲ ggml_tensor * q = ਗੰਢ->src[0];
        ਅਚਲ ggml_tensor * k = ਗੰਢ->src[1];
        ਅਚਲ ggml_tensor * v = ਗੰਢ->src[2];
        ਵਾਪਸ 2ull * q->ne[1] * q->ne[2] * (k->ne[0] + v->ne[0]) * k->ne[1] * q->ne[3];
    }
    ਵਾਪਸ 0;
}
ਖਾਲੀ ggml_vk_print_ਗੰਢ_list(ਅਚਲ ggml_cgraph * cgraph, ਪੂਰਨ start, ਪੂਰਨ end) {
    uint64_t ਕੁੱਲ_flops = 0;
    ਪੂਰਨ n_ops = 0;
    ਲਈ (ਪੂਰਨ j = start; j <= end && j < cgraph->n_nodes; j++) {
        uint64_t flops = ggml_vk_get_ਗੰਢ_flops(cgraph->nodes[j]);
        ਕੁੱਲ_flops += flops;
        n_ops++;
        ਜੇ (flops > 0) {
            GGML_LOG_CONT("  node %d: %s (%s) [%.2f GFLOP]\n",
                    j, cgraph->nodes[j]->name, ggml_op_name(cgraph->nodes[j]->op),
                    flops / 1e9);
        } ਨਹੀਂਤਾਂ {
            GGML_LOG_CONT("  node %d: %s (%s)\n",
                    j, cgraph->nodes[j]->name, ggml_op_name(cgraph->nodes[j]->op));
        }
    }
    GGML_LOG_CONT("  total: %d ops, %.2f GFLOP\n", n_ops, ਕੁੱਲ_flops / 1e9);
}
ਖਾਲੀ ggml_vk_print_device_ਗੁਆਚਾ_info(ਅਚਲ vk_device& device) {
    ggml_vk_print_device_fault_info(device);
    ਜੇ (device->serialize_submissions && device->diag_cgraph != ਖਾਲੀ_ਸੂਚਕ && device->diag_prev_ਸ਼ੁਰੂ >= 0) {
        GGML_LOG_ERROR("ggml_vulkan: device lost on %s, likely caused by previous submission (nodes %d to %d):\n",
                device->name.c_str(), device->diag_prev_ਸ਼ੁਰੂ, device->diag_prev_ਸਿਰਾ);
        ggml_vk_print_ਗੰਢ_list(device->diag_cgraph, device->diag_prev_ਸ਼ੁਰੂ, device->diag_prev_ਸਿਰਾ);
    } ਨਹੀਂਤਾਂ {
        GGML_LOG_ERROR("ggml_vulkan: device lost on %s\n", device->name.c_str());
    }
}
ਖਾਲੀ * ਅਚਲ vk_ptr_ਅਧਾਰ = (ਖਾਲੀ *)(uintptr_t) 0x1000;  // NOLINT

uint64_t vk_tensor_ਵਿੱਥ(ਅਚਲ ggml_tensor * tensor) {
    ਜੇ (tensor->view_src) {
        ਵਾਪਸ (uint8_t *) tensor->view_src->ਅੰਕੜੇ - (uint8_t *) vk_ptr_ਅਧਾਰ;
    }
    ਵਾਪਸ (uint8_t *) tensor->ਅੰਕੜੇ - (uint8_t *) vk_ptr_ਅਧਾਰ;
}

size_t ggml_vk_tensor_ਬਫਰ_ਵਿੱਥ(ਅਚਲ ggml_backend_vk_context * ctx, ਅਚਲ ggml_tensor * t) {
    // vk_tensor_offset() is relative to vk_ptr_base, but mapped host tensors need an offset relative to their Vulkan buffer.
    ਜੇ (ctx->device->uma) {
        vk_ਬਫਰ buf = ਖਾਲੀ_ਸੂਚਕ;
        size_t off = 0;
        ggml_vk_host_get(ctx->device, t->ਅੰਕੜੇ, buf, off);
        ਜੇ (buf) {
            ਵਾਪਸ off;
        }
    }
    ਵਾਪਸ (size_t)(vk_tensor_ਵਿੱਥ(t) + t->view_offs);
}
size_t ggml_vk_descriptor_ਵਿੱਥ(size_t tensor_ਵਿੱਥ, size_t alignment, size_t type_ਮਾਪ) {
    // Move the descriptor back until its distance to the tensor is divisible by the tensor type size.
    size_t descriptor_ਵਿੱਥ = tensor_ਵਿੱਥ & ~(alignment - 1);
    ਜਦੋਂ ((tensor_ਵਿੱਥ - descriptor_ਵਿੱਥ) % type_ਮਾਪ != 0) {
        GGML_ASSERT(descriptor_ਵਿੱਥ >= alignment);
        descriptor_ਵਿੱਥ -= alignment;
    }

    ਵਾਪਸ descriptor_ਵਿੱਥ;
}
uint32_t get_misalign_ਬਾਈਟਾਂ(ਅਚਲ ggml_backend_vk_context * ctx, ਅਚਲ ggml_tensor * t) {
    ਅਚਲ size_t tensor_ਵਿੱਥ = ggml_vk_tensor_ਬਫਰ_ਵਿੱਥ(ctx, t);
    ਅਚਲ size_t descriptor_ਵਿੱਥ = ggml_vk_descriptor_ਵਿੱਥ(
        tensor_ਵਿੱਥ, ctx->device->properties.limits.min_Storage_ਬਫਰ_ਵਿੱਥ_Alignment, ggml_type_ਮਾਪ(t->type));
    GGML_ASSERT(tensor_ਵਿੱਥ - descriptor_ਵਿੱਥ <= UINT32_MAX);
    ਵਾਪਸ tensor_ਵਿੱਥ - descriptor_ਵਿੱਥ;
}

uint32_t ggml_vk_concat_unit_ਮਾਪ(ggml_type type) {
    ਅਚਲ uint32_t type_ਮਾਪ = ggml_type_ਮਾਪ(type);

    ਜੇ (!ggml_is_quantized(type)) {
        ਵਾਪਸ type_ਮਾਪ;
    }

    // Use the widest existing concat shader that evenly divides a quant block.
    ਜੇ (type_ਮਾਪ % 8 == 0) {
        ਵਾਪਸ 8;
    }
    ਜੇ (type_ਮਾਪ % 4 == 0) {
        ਵਾਪਸ 4;
    }
    ਜੇ (type_ਮਾਪ % 2 == 0) {
        ਵਾਪਸ 2;
    }
    ਵਾਪਸ 1;
}
ਤਾਰਕਿਕ ggml_vk_concat_supported(ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ਅਚਲ ggml_tensor * dst) {
    ਜੇ (src0->type != src1->type || src0->type != dst->type) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (!ggml_is_quantized(src0->type)) {
        ਅਚਲ size_t type_ਮਾਪ = ggml_type_ਮਾਪ(src0->type);
        ਵਾਪਸ type_ਮਾਪ == 1 || type_ਮਾਪ == 2 || type_ਮਾਪ == 4 || type_ਮਾਪ == 8;
    }

    // Quantized tensor rows are block-aligned when created.
    ਵਾਪਸ ggml_is_contiguous_ਪੰਕਤੀਆਂ(src0) && ggml_is_contiguous_ਪੰਕਤੀਆਂ(src1) && ggml_is_contiguous_ਪੰਕਤੀਆਂ(dst);
}
ਸਥਿਰ ਤਾਰਕਿਕ vk_instance_initialized = ਝੂਠ;

vk_instance_t vk_instance;

ਸਥਿਰ Vk_Device_ਮਾਪ ggml_vk_get_max_ਬਫਰ_range(ਅਚਲ ggml_backend_vk_context * ctx, ਅਚਲ vk_ਬਫਰ &buf, ਅਚਲ Vk_Device_ਮਾਪ ਵਿੱਥ) {
    ਅਚਲ Vk_Device_ਮਾਪ range = std::min(Vk_Device_ਮਾਪ{buf->size - ਵਿੱਥ},
                                        Vk_Device_ਮਾਪ{ctx->device->properties.limits.max_Storage_ਬਫਰ_Range});
    ਵਾਪਸ range;
}

ਖਾਲੀ ggml_vk_wait_for_fence(ggml_backend_vk_context * ctx) {
    // Use waitForFences while most of the graph executes. Hopefully the CPU can sleep
    // during this wait.
    ਜੇ (ctx->almost_ready_fence_pending) {
        VK_CHECK(ctx->device->device.waitForFences({ ctx->almost_ready_fence }, ਸੱਚ, UINT64_MAX), "almost_ready_fence", ctx->device);
        ctx->device->device.resetFences({ ctx->almost_ready_fence });
        ctx->almost_ready_fence_pending = ਝੂਠ;
    }

    // Spin (w/pause) waiting for the graph to finish executing.
    vk::ਨਤੀਜਾ_2 ਨਤੀਜਾ;
    ਲਈ (;;) {
        ਕੋਸ਼ਿਸ਼ {
            ਨਤੀਜਾ = ctx->device->device.get_Fence_ਸਥਿਤੀ(ctx->fence);
        } ਫੜੋ (vk::Device_ਗੁਆਚਾ_Error &) {
            ggml_vk_print_device_ਗੁਆਚਾ_info(ctx->device);
            GGML_LOG_ERROR("ggml_vulkan: getFenceStatus at %s:%d\n", __FILE__, __LINE__);
            ਸੁੱਟੋ;
        }
        ਜੇ (ਨਤੀਜਾ == vk::ਨਤੀਜਾ_2::eSuccess) {
            ਰੋਕੋ;
        }
        ਜੇ (ਨਤੀਜਾ != vk::ਨਤੀਜਾ_2::eNotReady) {
            GGML_LOG_ERROR("ggml_vulkan: error %s at %s:%d\n", to_string(ਨਤੀਜਾ).c_str(), __FILE__, __LINE__);
            ਸੁੱਟੋ vk::SystemError(vk::make_error_code(ਨਤੀਜਾ), "ggml_vulkan: getFenceStatus");
        }
        ਲਈ (uint32_t i = 0; i < 100; ++i) {
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

ਸਥਿਰ ਤਾਰਕਿਕ ggml_vk_strip_decode_ਸਦਿਸ਼(ਅਚਲ uint32_t * code, size_t ਸ਼ਬਦ_ਗਿਣਤੀ, std::vector<uint32_t> & ਬਾਹਰਲਾ) {
    ਸਥਿਰ ਅਚਲ ਅੱਖਰ k_Decode_ਸਦਿਸ਼_Ext[] = "SPV_NV_cooperative_matrix_decode_vector";

    ਜੇ (ਸ਼ਬਦ_ਗਿਣਤੀ < 5) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਤਾਰਕਿਕ ਵਰਤਦਾ_decode_ਸਦਿਸ਼ = ਝੂਠ;
    ਲਈ (size_t pos = 5; pos < ਸ਼ਬਦ_ਗਿਣਤੀ; ) {
        uint32_t ਸ਼ਬਦ_2 = code[pos];
        uint32_t wc   = ਸ਼ਬਦ_2 >> spv::ਸ਼ਬਦ_ਗਿਣਤੀ_ਖਿਸਕਾਓ;
        uint32_t op   = ਸ਼ਬਦ_2 & spv::Op_ਕੋਡ_ਨਕਾਬ;
        GGML_ASSERT(wc > 0 && pos + wc <= ਸ਼ਬਦ_ਗਿਣਤੀ);
        ਜੇ (op == spv::OpExtension && wc >= 2) {
            ਅਚਲ ਅੱਖਰ * s = reinterpret_cast<ਅਚਲ ਅੱਖਰ *>(&code[pos + 1]);
            ਜੇ (strcmp(s, k_Decode_ਸਦਿਸ਼_Ext) == 0) {
                ਵਰਤਦਾ_decode_ਸਦਿਸ਼ = ਸੱਚ;
                ਰੋਕੋ;
            }
        }
        pos += wc;
    }

    ਜੇ (!ਵਰਤਦਾ_decode_ਸਦਿਸ਼) {
        ਵਾਪਸ ਝੂਠ;
    }

    VK_LOG_DEBUG("ggml_vk_strip_decode_vector: stripping SPV_NV_cooperative_matrix_decode_vector");

    // Bulk-copy unchanged runs and only break the run when an instruction needs to
    // be dropped or patched. Use reserve + insert/push_back so the destination buffer
    // is touched exactly once (no zero-initialization pass from resize()).
    ਬਾਹਰਲਾ.clear();
    ਬਾਹਰਲਾ.reserve(ਸ਼ਬਦ_ਗਿਣਤੀ);

    size_t ਲੜੀਵਾਰ_ਸ਼ੁਰੂ = 0;
    ਸਵੈਚਾਲਿਤ flush_ਲੜੀਵਾਰ = [&](size_t ਉੱਪਰ_ਮੰਜ਼ਿਲ) {
        ਜੇ (ਉੱਪਰ_ਮੰਜ਼ਿਲ > ਲੜੀਵਾਰ_ਸ਼ੁਰੂ) {
            ਬਾਹਰਲਾ.insert(ਬਾਹਰਲਾ.end(), code + ਲੜੀਵਾਰ_ਸ਼ੁਰੂ, code + ਉੱਪਰ_ਮੰਜ਼ਿਲ);
        }
    };

    ਲਈ (size_t pos = 5; pos < ਸ਼ਬਦ_ਗਿਣਤੀ; ) {
        uint32_t ਸ਼ਬਦ_2 = code[pos];
        uint32_t wc   = ਸ਼ਬਦ_2 >> spv::ਸ਼ਬਦ_ਗਿਣਤੀ_ਖਿਸਕਾਓ;
        uint32_t op   = ਸ਼ਬਦ_2 & spv::Op_ਕੋਡ_ਨਕਾਬ;
        GGML_ASSERT(wc > 0 && pos + wc <= ਸ਼ਬਦ_ਗਿਣਤੀ);

        ਜੇ (op == spv::OpExtension && wc >= 2) {
            ਅਚਲ ਅੱਖਰ * s = reinterpret_cast<ਅਚਲ ਅੱਖਰ *>(&code[pos + 1]);
            ਜੇ (strcmp(s, k_Decode_ਸਦਿਸ਼_Ext) == 0) {
                flush_ਲੜੀਵਾਰ(pos);
                pos += wc;
                ਲੜੀਵਾਰ_ਸ਼ੁਰੂ = pos;
                ਜਾਰੀ;
            }
        }

        ਜੇ (op == spv::OpCapability && wc == 2 && code[pos + 1] == k_Spv_Capability_Cooperative_ਮੈਟ੍ਰਿਕਸ_Decode_ਸਦਿਸ਼_NV) {
            flush_ਲੜੀਵਾਰ(pos);
            pos += wc;
            ਲੜੀਵਾਰ_ਸ਼ੁਰੂ = pos;
            ਜਾਰੀ;
        }

        ਜੇ (op == k_Spv_Op_Cooperative_ਮੈਟ੍ਰਿਕਸ_ਲੱਦੋ_Tensor_NV) {
            // [opcode/wc][ResultType][Result][Pointer][Object][TensorLayout][MemOperand mask][mem extras...][TA mask][ta extras...]
            GGML_ASSERT(wc >= 8);

            uint32_t mem_ਨਕਾਬ = code[pos + 6];
            size_t   cur      = pos + 7;
            // Each of these MemoryAccess bits (when set) carries one trailing operand.
            cur += (mem_ਨਕਾਬ & 0x2)     ? 1 : 0; // Aligned
            cur += (mem_ਨਕਾਬ & 0x8)     ? 1 : 0; // MakePointerAvailable
            cur += (mem_ਨਕਾਬ & 0x10)    ? 1 : 0; // MakePointerVisible
            cur += (mem_ਨਕਾਬ & 0x10000) ? 1 : 0; // AliasScopeINTELMask
            cur += (mem_ਨਕਾਬ & 0x20000) ? 1 : 0; // NoAliasINTELMask
            GGML_ASSERT(cur < pos + wc);

            uint32_t ta_ਨਕਾਬ = code[cur];
            ਜੇ ((ta_ਨਕਾਬ & k_Spv_Tensor_Addressing_Decode_ਸਦਿਸ਼_Func_ਦੋਆਧਾਰੀ_ਅੰਕ) == 0) {
                pos += wc;
                ਜਾਰੀ; // leave instruction inside the current unchanged run
            }

            flush_ਲੜੀਵਾਰ(pos);

            // Append unchanged prefix of the instruction (header through the mem-extras).
            size_t inst_ਸ਼ੁਰੂ = ਬਾਹਰਲਾ.size();
            size_t pre_n      = cur - pos;
            ਬਾਹਰਲਾ.insert(ਬਾਹਰਲਾ.end(), code + pos, code + pos + pre_n);

            // Emit TA mask with the DecodeVectorFunc bit cleared.
            ਬਾਹਰਲਾ.push_back(ta_ਨਕਾਬ & ~k_Spv_Tensor_Addressing_Decode_ਸਦਿਸ਼_Func_ਦੋਆਧਾਰੀ_ਅੰਕ);

            // TA extras: TensorView (0x1) and DecodeFunc (0x2) are kept verbatim;
            // DecodeVectorFunc (0x4) is dropped along with its trailing id operand.
            size_t keep_ta_extras = ((ta_ਨਕਾਬ & 0x1) ? 1 : 0) + ((ta_ਨਕਾਬ & 0x2) ? 1 : 0);
            ਜੇ (keep_ta_extras) {
                ਬਾਹਰਲਾ.insert(ਬਾਹਰਲਾ.end(), code + cur + 1, code + cur + 1 + keep_ta_extras);
            }

            GGML_ASSERT(wc == pre_n + 1 + keep_ta_extras + 1);

            // Patch the instruction header with the new (one-shorter) word count.
            uint32_t new_wc = wc - 1;
            ਬਾਹਰਲਾ[inst_ਸ਼ੁਰੂ] = (new_wc << spv::ਸ਼ਬਦ_ਗਿਣਤੀ_ਖਿਸਕਾਓ) | op;

            pos += wc;
            ਲੜੀਵਾਰ_ਸ਼ੁਰੂ = pos;
            ਜਾਰੀ;
        }

        pos += wc;
    }

    flush_ਲੜੀਵਾਰ(ਸ਼ਬਦ_ਗਿਣਤੀ);
    ਵਾਪਸ ਸੱਚ;
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_vk_roll_bk_loop(ਅਚਲ uint32_t * code, size_t ਸ਼ਬਦ_ਗਿਣਤੀ, std::vector<uint32_t> & ਬਾਹਰਲਾ) {
    ਜੇ (ਸ਼ਬਦ_ਗਿਣਤੀ < 5) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਬਣਤਰ vk_spv_loop {
        size_t   ਸਿਰਲੇਖ;
        size_t   end;
        uint32_t control;
    };

    std::vector<vk_spv_loop> loops;

    // Collect a list of all loops in the module.
    ਲਈ (size_t pos = 5; pos < ਸ਼ਬਦ_ਗਿਣਤੀ; ) {
        ਅਚਲ uint32_t wc = code[pos] >> spv::ਸ਼ਬਦ_ਗਿਣਤੀ_ਖਿਸਕਾਓ;
        ਅਚਲ uint32_t op = code[pos] & spv::Op_ਕੋਡ_ਨਕਾਬ;
        ਜੇ (wc == 0 || pos + wc > ਸ਼ਬਦ_ਗਿਣਤੀ) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਜੇ (op == spv::Op_Loop_ਮਿਲਾਓ && wc >= 4) { loops.push_back({ pos, 0, code[pos + 3] }); }

        ਜੇ (op == spv::OpLabel && wc >= 2) {
            ਲਈ (ਸਵੈਚਾਲਿਤ & l : loops) {
                ਜੇ (l.end == 0 && code[l.ਸਿਰਲੇਖ + 1] == code[pos + 1]) { l.end = pos; }
            }
        }

        pos += wc;
    }

    ਸਵੈਚਾਲਿਤ encloses = [](ਅਚਲ vk_spv_loop & a, ਅਚਲ vk_spv_loop & b) {
        ਵਾਪਸ a.ਸਿਰਲੇਖ < b.ਸਿਰਲੇਖ && b.ਸਿਰਲੇਖ < a.end;
    };

    // Find the BK loop.
    ਅਚਲ vk_spv_loop * bk = ਖਾਲੀ_ਸੂਚਕ;
    ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & h : loops) {
        ਜੇ (h.control != spv::Loop_Control_Unroll_ਨਕਾਬ) {
            ਜਾਰੀ;
        }
        ਅਚਲ vk_spv_loop * ਮਾਪਾ = ਖਾਲੀ_ਸੂਚਕ;
        ਤਾਰਕਿਕ has_ਬੱਚਾ = ਝੂਠ;
        ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & g : loops) {
            ਜੇ (encloses(g, h) && (!ਮਾਪਾ || g.ਸਿਰਲੇਖ > ਮਾਪਾ->ਸਿਰਲੇਖ)) {
                ਮਾਪਾ = &g;
            }
            ਜੇ (encloses(h, g)) {
                has_ਬੱਚਾ = ਸੱਚ;
            }
        }
        // BK loop should be the last loop nested inside the loop with no hint
        // and have at least one child loop.
        ਜੇ (ਮਾਪਾ &&
            ਮਾਪਾ->control == spv::Loop_Control_ਨਕਾਬ_None &&
            has_ਬੱਚਾ &&
            (!bk || h.ਸਿਰਲੇਖ > bk->ਸਿਰਲੇਖ)) {
            bk = &h;
        }
    }
    ਜੇ (!bk) {
        ਵਾਪਸ ਝੂਠ;
    }

    // set DontUnroll instead of Unroll
    ਬਾਹਰਲਾ.ਸੌਂਪੋ(code, code + ਸ਼ਬਦ_ਗਿਣਤੀ);
    ਬਾਹਰਲਾ[bk->ਸਿਰਲੇਖ + 3] = spv::Loop_Control_Dont_Unroll_ਨਕਾਬ;
    ਵਾਪਸ ਸੱਚ;
}

ਸਥਿਰ ਖਾਲੀ ggml_vk_create_pipeline_func(vk_device& device, vk_pipeline& pipeline, size_t spv_ਮਾਪ, ਅਚਲ ਖਾਲੀ* spv_ਅੰਕੜੇ, ਅਚਲ std::string entrypoint,
                                         uint32_t parameter_ਗਿਣਤੀ, std::array<uint32_t, 3> wg_denoms, std::vector<uint32_t> specialization_constants,
                                         ਤਾਰਕਿਕ disable_robustness, ਤਾਰਕਿਕ require_ਭਰਿਆ_subgroups, uint32_t required_subgroup_ਮਾਪ) {
    VK_LOG_DEBUG("ggml_vk_create_pipeline(" << device->name << ", " << pipeline->name << ", " << entrypoint << ", " << parameter_ਗਿਣਤੀ <<
                 ", (" << wg_denoms[0] << "," << wg_denoms[1] << "," << wg_denoms[2] << "), specialization_constants, " <<
                 disable_robustness << ", " << require_ਭਰਿਆ_subgroups << ", " << required_subgroup_ਮਾਪ << ")");
    GGML_ASSERT(parameter_ਗਿਣਤੀ > 0);
    GGML_ASSERT(parameter_ਗਿਣਤੀ <= MAX_PARAMETER_ਗਿਣਤੀ);
    GGML_ASSERT(wg_denoms[0] > 0 && wg_denoms[1] > 0 && wg_denoms[2] > 0); // NOLINT

    vk::ShaderModuleCreateInfo shader_module_create_info({}, spv_ਮਾਪ, reinterpret_cast<ਅਚਲ uint32_t *>(spv_ਅੰਕੜੇ));

    // Patch SPIR-V to enable supported FP16 float controls, avoiding the need
    // for separate shader variants.
    std::vector<uint32_t> spirv;
    ਜੇ (device->float_controls_rte_fp16 || device->float_controls_denorm_preserve_fp16) {
        ਅਚਲ uint32_t* spv_ਸ਼ਬਦ = reinterpret_cast<ਅਚਲ uint32_t *>(spv_ਅੰਕੜੇ);
        size_t ਸ਼ਬਦ_ਗਿਣਤੀ = spv_ਮਾਪ / ਆਕਾਰ(uint32_t);
        spirv.ਸੌਂਪੋ(spv_ਸ਼ਬਦ, spv_ਸ਼ਬਦ + ਸ਼ਬਦ_ਗਿਣਤੀ);

        // Find insertion points respecting SPIR-V layout order:
        //   Header(5) -> OpCapability -> OpExtension -> ... -> OpEntryPoint -> OpExecutionMode -> ...
        size_t pos = 5; // skip header
        size_t cap_ਪਾਓ_pos = pos;
        size_t ext_ਪਾਓ_pos = pos;
        size_t exec_ਪਾਓ_pos = pos;
        uint32_t ਇੰਦਰਾਜ_ਬਿੰਦੂ_id = 0;

        ਜਦੋਂ (pos < spirv.size()) {
            uint32_t opcode = spirv[pos] & spv::Op_ਕੋਡ_ਨਕਾਬ;
            uint32_t len    = spirv[pos] >> spv::ਸ਼ਬਦ_ਗਿਣਤੀ_ਖਿਸਕਾਓ;
            ਜੇ (len == 0) ਰੋਕੋ;

            ਜੇ (opcode == spv::OpCapability) {
                cap_ਪਾਓ_pos = pos + len;
                ext_ਪਾਓ_pos = pos + len;
            } ਨਹੀਂਤਾਂ ਜੇ (opcode == spv::OpExtension) {
                ext_ਪਾਓ_pos = pos + len;
            } ਨਹੀਂਤਾਂ ਜੇ (opcode == spv::Op_ਇੰਦਰਾਜ_ਬਿੰਦੂ) {
                ਇੰਦਰਾਜ_ਬਿੰਦੂ_id = spirv[pos + 2];
                exec_ਪਾਓ_pos = pos + len;
            } ਨਹੀਂਤਾਂ ਜੇ (opcode == spv::Op_Execution_ਬਹੁਲਕ || opcode == spv::Op_Execution_ਬਹੁਲਕ_Id) {
                exec_ਪਾਓ_pos = pos + len;
            } ਨਹੀਂਤਾਂ ਜੇ (ਇੰਦਰਾਜ_ਬਿੰਦੂ_id != 0) {
                ਰੋਕੋ;
            }

            pos += len;
        }

        // Insert from latest position first so earlier indices stay valid.

        ਜੇ (device->float_controls_rte_fp16) {
            // OpExecutionMode %entrypoint RoundingModeRTE 16
            uint32_t exec_ਬਹੁਲਕ[] = { (4u << spv::ਸ਼ਬਦ_ਗਿਣਤੀ_ਖਿਸਕਾਓ) | spv::Op_Execution_ਬਹੁਲਕ, ਇੰਦਰਾਜ_ਬਿੰਦੂ_id, spv::Execution_ਬਹੁਲਕ_Rounding_ਬਹੁਲਕ_RTE, 16 };
            spirv.insert(spirv.begin() + exec_ਪਾਓ_pos, std::begin(exec_ਬਹੁਲਕ), std::end(exec_ਬਹੁਲਕ));
        }

        ਜੇ (device->float_controls_denorm_preserve_fp16) {
            // OpExecutionMode %entrypoint DenormPreserve 16
            uint32_t exec_ਬਹੁਲਕ[] = { (4u << spv::ਸ਼ਬਦ_ਗਿਣਤੀ_ਖਿਸਕਾਓ) | spv::Op_Execution_ਬਹੁਲਕ, ਇੰਦਰਾਜ_ਬਿੰਦੂ_id, spv::Execution_ਬਹੁਲਕ_Denorm_Preserve, 16 };
            spirv.insert(spirv.begin() + exec_ਪਾਓ_pos, std::begin(exec_ਬਹੁਲਕ), std::end(exec_ਬਹੁਲਕ));
        }

        // OpExtension "SPV_KHR_float_controls"
        ਅਚਲ ਅੱਖਰ ext_str[] = "SPV_KHR_float_controls";
        size_t ext_str_ਸ਼ਬਦ = CEIL_DIV(ਆਕਾਰ(ext_str), ਆਕਾਰ(uint32_t));
        std::vector<uint32_t> extension(1 + ext_str_ਸ਼ਬਦ, 0);
        extension[0] = (uint32_t)((1 + ext_str_ਸ਼ਬਦ) << spv::ਸ਼ਬਦ_ਗਿਣਤੀ_ਖਿਸਕਾਓ) | spv::OpExtension;
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(&extension[1], ext_str, ਆਕਾਰ(ext_str));
        spirv.insert(spirv.begin() + ext_ਪਾਓ_pos, extension.begin(), extension.end());

        ਜੇ (device->float_controls_rte_fp16) {
            // OpCapability RoundingModeRTE
            uint32_t capability[] = { (2u << spv::ਸ਼ਬਦ_ਗਿਣਤੀ_ਖਿਸਕਾਓ) | spv::OpCapability, spv::Capability_Rounding_ਬਹੁਲਕ_RTE };
            spirv.insert(spirv.begin() + cap_ਪਾਓ_pos, std::begin(capability), std::end(capability));
        }

        ਜੇ (device->float_controls_denorm_preserve_fp16) {
            // OpCapability DenormPreserve
            uint32_t capability[] = { (2u << spv::ਸ਼ਬਦ_ਗਿਣਤੀ_ਖਿਸਕਾਓ) | spv::OpCapability, spv::CapabilityDenormPreserve };
            spirv.insert(spirv.begin() + cap_ਪਾਓ_pos, std::begin(capability), std::end(capability));
        }

        shader_module_create_info = vk::ShaderModuleCreateInfo({}, spirv.size() * ਆਕਾਰ(uint32_t), spirv.ਅੰਕੜੇ());
    }

#ਜੇ defined(GGML_VULKAN_COOPMAT2_DECODE_ਸਦਿਸ਼_GLSLC_SUPPORT)
    ਜੇ (device->coopmat2 && !device->coopmat2_decode_ਸਦਿਸ਼) {
        ਅਚਲ uint32_t * src   = spirv.ਖਾਲੀ_ਥਾਵਾਂ() ? reinterpret_cast<ਅਚਲ uint32_t *>(spv_ਅੰਕੜੇ) : spirv.ਅੰਕੜੇ();
        size_t           src_n = spirv.ਖਾਲੀ_ਥਾਵਾਂ() ? spv_ਮਾਪ / ਆਕਾਰ(uint32_t) : spirv.size();
        std::vector<uint32_t> stripped;
        ਜੇ (ggml_vk_strip_decode_ਸਦਿਸ਼(src, src_n, stripped)) {
            spirv = std::move(stripped);
            shader_module_create_info = vk::ShaderModuleCreateInfo({}, spirv.size() * ਆਕਾਰ(uint32_t), spirv.ਅੰਕੜੇ());
        }
    }
#ਜੇ_ਅੰਤ

#ਜੇ VK_ਸਿਰਲੇਖ_ਸੰਸਕਰਣ >= 287
    // Roll the mul_mm BK loop on Asahi Linux. Skip bf16 and the mul_mmq pipelines.
    ਜੇ (device->driver_id == vk::DriverId::eMesaHoneykrisp &&
        pipeline->name.rfind("matmul", 0) == 0 &&
        pipeline->name.find("bf16") == std::string::npos &&
        pipeline->name.find("q8_1") == std::string::npos) {
        ਅਚਲ uint32_t * src   = spirv.ਖਾਲੀ_ਥਾਵਾਂ() ? reinterpret_cast<ਅਚਲ uint32_t *>(spv_ਅੰਕੜੇ) : spirv.ਅੰਕੜੇ();
        size_t           src_n = spirv.ਖਾਲੀ_ਥਾਵਾਂ() ? spv_ਮਾਪ / ਆਕਾਰ(uint32_t) : spirv.size();
        std::vector<uint32_t> rolled;
        ਜੇ (ggml_vk_roll_bk_loop(src, src_n, rolled)) {
            spirv = std::move(rolled);
            shader_module_create_info = vk::ShaderModuleCreateInfo({}, spirv.size() * ਆਕਾਰ(uint32_t), spirv.ਅੰਕੜੇ());
        }
    }
#ਜੇ_ਅੰਤ

    ਕੋਸ਼ਿਸ਼ {
        pipeline->shader_module = device->device.createShaderModule(shader_module_create_info);
    } ਫੜੋ (ਅਚਲ vk::SystemError& e) {
        std::cerr << "ggml_vulkan: shader module creation failed for " << pipeline->name << ": " << e.what() << std::ਸਤਰ_ਅੰਤ;
        ਸੁੱਟੋ;
    }

    vk::ਧੱਕੋ_Constant_Range pcr(
        vk::Shader_Stage_Flag_ਬਿੱਟ::eCompute,
        0,
        pipeline->ਧੱਕੋ_constant_ਮਾਪ
    );

    vk::PipelineLayoutCreateInfo pipeline_layout_create_info(vk::PipelineLayoutCreateFlags(), device->dsl, pcr);
    ਕੋਸ਼ਿਸ਼ {
        pipeline->layout = device->device.createPipelineLayout(pipeline_layout_create_info);
    } ਫੜੋ (ਅਚਲ vk::SystemError& e) {
        std::cerr << "ggml_vulkan: pipeline layout creation failed for " << pipeline->name << ": " << e.what() << std::ਸਤਰ_ਅੰਤ;
        ਸੁੱਟੋ;
    }

    std::vector<vk::Specialization_Map_ਇੰਦਰਾਜ> specialization_entries(specialization_constants.size());

    ਲਈ (size_t i = 0; i < specialization_constants.size(); i++) {
        specialization_entries[i].constantID = i;
        specialization_entries[i].ਵਿੱਥ = i * ਆਕਾਰ(uint32_t);
        specialization_entries[i].size = ਆਕਾਰ(uint32_t);
    }

    vk::SpecializationInfo specialization_info(
        specialization_entries.size(),
        specialization_entries.ਅੰਕੜੇ(),
        specialization_constants.size() * ਆਕਾਰ(uint32_t),
        specialization_constants.ਅੰਕੜੇ()
    );

    vk::PipelineShaderStageCreateFlags pipeline_shader_stage_create_flags{};

    ਜੇ (device->subgroup_require_ਭਰਿਆ_support && require_ਭਰਿਆ_subgroups) {
        pipeline_shader_stage_create_flags |= vk::Pipeline_Shader_Stage_Create_Flag_ਬਿੱਟ::e_Require_ਭਰਿਆ_Subgroups_EXT;
    }

    vk::PipelineShaderStageCreateInfo pipeline_shader_create_info(
            pipeline_shader_stage_create_flags,
            vk::Shader_Stage_Flag_ਬਿੱਟ::eCompute,
            pipeline->shader_module,
            entrypoint.c_str(),
            &specialization_info);

    vk::Pipeline_Shader_Stage_Required_Subgroup_ਮਾਪ_Create_Info_EXT pipeline_shader_stage_required_subgroup_ਮਾਪ_create_info;
    pipeline_shader_stage_required_subgroup_ਮਾਪ_create_info.required_Subgroup_ਮਾਪ = required_subgroup_ਮਾਪ;
    ਜੇ (device->subgroup_ਮਾਪ_control && required_subgroup_ਮਾਪ > 0) {
        GGML_ASSERT(device->subgroup_min_ਮਾਪ <= required_subgroup_ਮਾਪ && required_subgroup_ਮਾਪ <= device->subgroup_max_ਮਾਪ);
        pipeline_shader_create_info.ਸੈੱਟ_PNext(&pipeline_shader_stage_required_subgroup_ਮਾਪ_create_info);
    }

    vk::ComputePipelineCreateInfo compute_pipeline_create_info(
        device->pipeline_executable_properties_support ?
            vk::Pipeline_Create_Flag_ਬਿੱਟ::eCaptureStatisticsKHR :
            vk::PipelineCreateFlags{},
        pipeline_shader_create_info,
        pipeline->layout);

    vk::PipelineRobustnessCreateInfoEXT rci;

    ਜੇ (device->pipeline_robustness && disable_robustness) {
        rci.storageBuffers = vk::Pipeline_Robustness_ਬਫਰ_Behavior_EXT::eDisabled;
        rci.ਇਕਸਾਰ_Buffers = vk::Pipeline_Robustness_ਬਫਰ_Behavior_EXT::eDisabled;
        compute_pipeline_create_info.ਸੈੱਟ_PNext(&rci);
    }

#ਜੇ defined(VK_EXT_shader_64bit_indexing)
    vk::PipelineCreateFlags2CreateInfo pipelineFlags2CreateInfo;
    ਜੇ (pipeline->is_64b_indexing)
    {
        pipelineFlags2CreateInfo.flags = vk::PipelineCreateFlagBits2::e64_ਦੋਆਧਾਰੀ_ਅੰਕ_Indexing_EXT;
        ਜੇ (device->pipeline_executable_properties_support) {
            pipelineFlags2CreateInfo.flags |= vk::PipelineCreateFlagBits2::eCaptureStatisticsKHR;
        }
        pipelineFlags2CreateInfo.ਸੈੱਟ_PNext(compute_pipeline_create_info.p_ਅਗਲਾ);
        compute_pipeline_create_info.ਸੈੱਟ_PNext(&pipelineFlags2CreateInfo);
    }
#ਜੇ_ਅੰਤ

    ਕੋਸ਼ਿਸ਼ {
        pipeline->pipeline = device->device.createComputePipeline(VK_NULL_HANDLE, compute_pipeline_create_info).value;
    } ਫੜੋ (ਅਚਲ vk::SystemError& e) {
        std::cerr << "ggml_vulkan: compute pipeline creation failed for " << pipeline->name << ": " << e.what() << std::ਸਤਰ_ਅੰਤ;
        ਸੁੱਟੋ;
    }

    ਜੇ (vk_instance.debug_utils_support) {
        vk::DebugUtilsObjectNameInfoEXT duoni;
        duoni.objectType = vk::ObjectType::ePipeline;
        duoni.pObjectName = pipeline->name.c_str();
        duoni.objectHandle = /*reinterpret_cast*/(uint64_t)(static_cast<VkPipeline>(pipeline->pipeline));
        vk_instance.pfn_vk_ਸੈੱਟ_Debug_Utils_Object_Name_EXT(device->device, &static_cast<VkDebugUtilsObjectNameInfoEXT &>(duoni));
    }

    ਜੇ (device->pipeline_executable_properties_support) {
        vk::PipelineExecutableInfoKHR executableInfo;
        executableInfo.pipeline = pipeline->pipeline;

        ਸਵੈਚਾਲਿਤ statistics = device->device.getPipelineExecutableStatisticsKHR(executableInfo);

        ਤਾਰਕਿਕ print_stats = !vk_pipeline_stats_filter.ਖਾਲੀ_ਥਾਵਾਂ() &&
                           pipeline->name.find(vk_pipeline_stats_filter) != std::string::npos;
        ਜੇ (print_stats) {
            std::cerr << "ggml_vulkan: pipeline stats for " << pipeline->name << ":" << std::ਸਤਰ_ਅੰਤ;
        }

        ਲਈ (ਸਵੈਚਾਲਿਤ & s : statistics) {
            ਜੇ (print_stats) {
                std::cerr << "ggml_vulkan:   " << s.name.ਅੰਕੜੇ() << ": ";
                ਚੋਣ (s.format) {
                    ਹਾਲਤ vk::PipelineExecutableStatisticFormatKHR::eBool32:
                        std::cerr << (s.value.b32 ? "true" : "false");
                        ਰੋਕੋ;
                    ਹਾਲਤ vk::PipelineExecutableStatisticFormatKHR::eInt64:
                        std::cerr << s.value.i64;
                        ਰੋਕੋ;
                    ਹਾਲਤ vk::PipelineExecutableStatisticFormatKHR::eUint64:
                        std::cerr << s.value.u64;
                        ਰੋਕੋ;
                    ਹਾਲਤ vk::PipelineExecutableStatisticFormatKHR::eFloat64:
                        std::cerr << s.value.f64;
                        ਰੋਕੋ;
                }
                std::cerr << std::ਸਤਰ_ਅੰਤ;
            }
            // "Register Count" is reported by NVIDIA drivers.
            ਜੇ (strcmp(s.name, "Register Count") == 0) {
                VK_LOG_DEBUG(pipeline->name << " " << s.name << ": " << s.value.u64 << " registers");
                pipeline->ਰਜਿਸਟਰ_ਗਿਣਤੀ = (uint32_t)s.value.u64;
            }
        }
    }

    {
        std::lock_guard<std::mutex> guard(device->compile_mutex);
        device->all_pipelines.push_back(pipeline);
        pipeline->compiled = ਸੱਚ;
        pipeline->compile_pending = ਝੂਠ;
    }
    device->compile_cv.notify_all();
}

ਖਾਲੀ ggml_vk_destroy_pipeline(vk::Device& device, vk_pipeline& pipeline) {
    VK_LOG_DEBUG("ggml_pipeline_destroy_pipeline(" << pipeline->name << ")");
    device.destroyPipelineLayout(pipeline->layout);

    device.destroyShaderModule(pipeline->shader_module);

    device.destroyPipeline(pipeline->pipeline);
}

ਖਾਲੀ ggml_pipeline_request_descriptor_sets(ggml_backend_vk_context *ctx, vk_pipeline& pipeline, uint32_t n) {
    VK_LOG_DEBUG("ggml_pipeline_request_descriptor_sets(" << pipeline->name << ", " << n << ")");
    ctx->pipeline_descriptor_ਸੈੱਟ_requirements += n;
    ਜੇ (!pipeline->compiled) {
        ggml_vk_ਲੱਦੋ_shaders(ctx->device, pipeline);
    }
    ggml_pipeline_allocate_descriptor_sets(ctx);
}

ਖਾਲੀ ggml_pipeline_allocate_descriptor_sets(ggml_backend_vk_context * ctx) {

    ਜੇ (ctx->descriptor_sets.size() >= ctx->pipeline_descriptor_ਸੈੱਟ_requirements) {
        // Enough descriptors are available
        ਵਾਪਸ;
    }

    vk_device& device = ctx->device;

    // Grow by 50% to avoid frequent allocations
    uint32_t needed = std::max(3 * ctx->descriptor_sets.size() / 2, size_t{ctx->pipeline_descriptor_ਸੈੱਟ_requirements});
    uint32_t ਮੰਜ਼ਿਲ_alloc = needed - ctx->descriptor_sets.size();
    uint32_t pool_ਬਚਿਆ = VK_DEVICE_DESCRIPTOR_POOL_ਮਾਪ - ctx->descriptor_sets.size() % VK_DEVICE_DESCRIPTOR_POOL_ਮਾਪ;
    uint32_t pool_idx = ctx->descriptor_sets.size() / VK_DEVICE_DESCRIPTOR_POOL_ਮਾਪ;

    ਜਦੋਂ (ਮੰਜ਼ਿਲ_alloc > 0) {
        ਅਚਲ uint32_t alloc_ਗਿਣਤੀ = std::min(pool_ਬਚਿਆ, ਮੰਜ਼ਿਲ_alloc);
        ਮੰਜ਼ਿਲ_alloc -= alloc_ਗਿਣਤੀ;
        pool_ਬਚਿਆ = VK_DEVICE_DESCRIPTOR_POOL_ਮਾਪ;

        ਜੇ (pool_idx >= ctx->descriptor_pools.size()) {
            vk::Descriptor_Pool_ਮਾਪ descriptor_pool_ਮਾਪ(vk::DescriptorType::e_Storage_ਬਫਰ, MAX_PARAMETER_ਗਿਣਤੀ * VK_DEVICE_DESCRIPTOR_POOL_ਮਾਪ);
            vk::DescriptorPoolCreateInfo descriptor_pool_create_info({}, VK_DEVICE_DESCRIPTOR_POOL_ਮਾਪ, descriptor_pool_ਮਾਪ);
            ctx->descriptor_pools.push_back(device->device.createDescriptorPool(descriptor_pool_create_info));
        }

        std::vector<vk::Descriptor_ਸੈੱਟ_Layout> layouts(alloc_ਗਿਣਤੀ);
        ਲਈ (uint32_t i = 0; i < alloc_ਗਿਣਤੀ; i++) {
            layouts[i] = device->dsl;
        }
        vk::Descriptor_ਸੈੱਟ_Allocate_Info descriptor_ਸੈੱਟ_alloc_info(ctx->descriptor_pools[pool_idx], alloc_ਗਿਣਤੀ, layouts.ਅੰਕੜੇ());
        std::vector<vk::Descriptor_ਸੈੱਟ> sets = device->device.allocateDescriptorSets(descriptor_ਸੈੱਟ_alloc_info);
        ctx->descriptor_sets.insert(ctx->descriptor_sets.end(), sets.begin(), sets.end());

        pool_idx++;
    }
    ctx->descriptor_ਸੈੱਟ_bindings.resize(ctx->descriptor_sets.size());
}

ਸਥਿਰ vk_command_ਬਫਰ* ggml_vk_create_cmd_ਬਫਰ(vk_device& device, vk_command_pool& p) {
    VK_LOG_DEBUG("ggml_vk_create_cmd_buffer()");
    vk::Command_ਬਫਰ_Allocate_Info command_ਬਫਰ_alloc_info(
        p.pool,
        vk::Command_ਬਫਰ_Level::ePrimary,
        1);
    ਅਚਲ std::vector<vk::Command_ਬਫਰ> cmd_buffers = device->device.allocateCommandBuffers(command_ਬਫਰ_alloc_info);
    p.cmd_buffers.push_back({ cmd_buffers.front(), 0, ਸੱਚ });
    ਵਾਪਸ &p.cmd_buffers[p.cmd_buffers.size()-1];
}

ਖਾਲੀ ggml_vk_submit(vk_context& ctx, vk::Fence fence) {
    ਜੇ (ctx->seqs.ਖਾਲੀ_ਥਾਵਾਂ()) {
        ਜੇ (fence) {
            ctx->p->q->handle->submit({}, fence);
        }
        ਵਾਪਸ;
    }
    VK_LOG_DEBUG("ggml_vk_submit(" << ctx << ", " << fence << ")");

    std::vector<std::vector<uint64_t>> tl_wait_vals;
    std::vector<std::vector<uint64_t>> tl_signal_vals;
    std::vector<std::vector<vk::Semaphore>> tl_wait_semaphores;
    std::vector<std::vector<vk::Semaphore>> tl_signal_semaphores;
    std::vector<vk::TimelineSemaphoreSubmitInfo> tl_submit_infos;
    std::vector<vk::SubmitInfo> submit_infos;
    ਪੂਰਨ idx = -1;
    std::vector<std::vector<vk::PipelineStageFlags>> stage_flags;

    size_t reserve = 0;

    ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ& sequence : ctx->seqs) {
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

    ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ& sequence : ctx->seqs) {
        ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ& submission : sequence) {
            stage_flags.push_back({});
            idx++;
            tl_wait_vals.push_back({});
            tl_wait_semaphores.push_back({});
            tl_signal_vals.push_back({});
            tl_signal_semaphores.push_back({});
            ਲਈ (size_t i = 0; i < submission.wait_semaphores.size(); i++) {
                stage_flags[idx].push_back(ctx->p->q->stage_flags);
                tl_wait_vals[idx].push_back(submission.wait_semaphores[i].value);
                tl_wait_semaphores[idx].push_back(submission.wait_semaphores[i].s);
            }
            ਲਈ (size_t i = 0; i < submission.signal_semaphores.size(); i++) {
                tl_signal_vals[idx].push_back(submission.signal_semaphores[i].value);
                tl_signal_semaphores[idx].push_back(submission.signal_semaphores[i].s);
            }
            tl_submit_infos.push_back({
                (uint32_t) submission.wait_semaphores.size(),
                tl_wait_vals[idx].ਅੰਕੜੇ(),
                (uint32_t) submission.signal_semaphores.size(),
                tl_signal_vals[idx].ਅੰਕੜੇ(),
            });
            tl_submit_infos[idx].sType = vk::StructureType::eTimelineSemaphoreSubmitInfo;
            tl_submit_infos[idx].p_ਅਗਲਾ = ਖਾਲੀ_ਸੂਚਕ;
            vk::SubmitInfo si{
                (uint32_t) submission.wait_semaphores.size(),
                tl_wait_semaphores[idx].ਅੰਕੜੇ(),
                stage_flags[idx].ਅੰਕੜੇ(),
                1,
                &submission.ਬਫਰ->buf,
                (uint32_t) submission.signal_semaphores.size(),
                tl_signal_semaphores[idx].ਅੰਕੜੇ(),
            };
            si.ਸੈੱਟ_PNext(&tl_submit_infos[idx]);
            submit_infos.push_back(si);
        }
    }

    ctx->p->q->handle->submit(submit_infos, fence);

    ctx->seqs.clear();
}

uint32_t ggml_vk_ਲੱਭੋ_ਕਤਾਰ_family_ਸੂਚਕਾਂਕ(std::vector<vk::ਕਤਾਰ_Family_Properties>& ਕਤਾਰ_family_props, ਅਚਲ vk::ਕਤਾਰ_Flags& required, ਅਚਲ vk::ਕਤਾਰ_Flags& avoid, int32_t compute_ਸੂਚਕਾਂਕ, uint32_t min_ਅੰਕ_queues) {
    VK_LOG_DEBUG("ggml_vk_find_queue_family_index()");
    ਅਚਲ uint32_t qfsize = ਕਤਾਰ_family_props.size();

    // Try with avoid preferences first
    ਲਈ (uint32_t i = 0; i < qfsize; i++) {
        ਜੇ (ਕਤਾਰ_family_props[i].ਕਤਾਰ_ਗਿਣਤੀ >= min_ਅੰਕ_queues && (compute_ਸੂਚਕਾਂਕ < 0 || i != (uint32_t) compute_ਸੂਚਕਾਂਕ) && ਕਤਾਰ_family_props[i].ਕਤਾਰ_Flags_2 & required && !(ਕਤਾਰ_family_props[i].ਕਤਾਰ_Flags_2 & avoid)) {
            ਵਾਪਸ i;
        }
    }

    // Fall back to only required
    ਲਈ (size_t i = 0; i < qfsize; i++) {
        ਜੇ (ਕਤਾਰ_family_props[i].ਕਤਾਰ_ਗਿਣਤੀ >= min_ਅੰਕ_queues && (compute_ਸੂਚਕਾਂਕ < 0 || i != (uint32_t) compute_ਸੂਚਕਾਂਕ) && ਕਤਾਰ_family_props[i].ਕਤਾਰ_Flags_2 & required) {
            ਵਾਪਸ i;
        }
    }

    // Fall back to reusing compute queue
    ਲਈ (size_t i = 0; i < qfsize; i++) {
        ਜੇ (ਕਤਾਰ_family_props[i].ਕਤਾਰ_ਗਿਣਤੀ >= min_ਅੰਕ_queues && ਕਤਾਰ_family_props[i].ਕਤਾਰ_Flags_2 & required) {
            ਵਾਪਸ i;
        }
    }

    // Fall back to ignoring min_num_queries
    ਲਈ (size_t i = 0; i < qfsize; i++) {
        ਜੇ (ਕਤਾਰ_family_props[i].ਕਤਾਰ_Flags_2 & required) {
            ਵਾਪਸ i;
        }
    }

    // All commands that are allowed on a queue that supports transfer operations are also allowed on a queue that supports either graphics or compute operations.
    // Thus, if the capabilities of a queue family include VK_QUEUE_GRAPHICS_BIT or VK_QUEUE_COMPUTE_BIT, then reporting the VK_QUEUE_TRANSFER_BIT capability separately for that queue family is optional.
    ਜੇ (compute_ਸੂਚਕਾਂਕ >= 0) {
        ਵਾਪਸ compute_ਸੂਚਕਾਂਕ;
    }

    std::cerr << "ggml_vulkan: No suitable queue family index found." << std::ਸਤਰ_ਅੰਤ;

    ਲਈ(ਸਵੈਚਾਲਿਤ &q_family : ਕਤਾਰ_family_props) {
        std::cerr << "Queue number: "  + std::to_string(q_family.ਕਤਾਰ_ਗਿਣਤੀ) << " flags: " + to_string(q_family.ਕਤਾਰ_Flags_2) << std::ਸਤਰ_ਅੰਤ;
    }
    abort();
}

std::unique_ptr<vk_ਕਤਾਰ> ggml_vk_create_ਕਤਾਰ(vk_device& device, uint32_t ਕਤਾਰ_family_ਸੂਚਕਾਂਕ, uint32_t ਕਤਾਰ_ਸੂਚਕਾਂਕ, vk::PipelineStageFlags&& stage_flags, ਤਾਰਕਿਕ transfer_only) {
    VK_LOG_DEBUG("ggml_vk_create_queue()");
    std::lock_guard<std::recursive_mutex> guard(device->mutex);

    ਸਵੈਚਾਲਿਤ q = std::make_unique<vk_ਕਤਾਰ>();
    q->ਕਤਾਰ_family_ਸੂਚਕਾਂਕ = ਕਤਾਰ_family_ਸੂਚਕਾਂਕ;
    q->transfer_only = transfer_only;

    std::shared_ptr<vk_ਕਤਾਰ_handle> h;
    vk::Device_ਕਤਾਰ_Info2 ਕਤਾਰ_info2{};
    ਕਤਾਰ_info2.ਕਤਾਰ_Family_ਸੂਚਕਾਂਕ = ਕਤਾਰ_family_ਸੂਚਕਾਂਕ;
    ਕਤਾਰ_info2.ਕਤਾਰ_ਸੂਚਕਾਂਕ_2 = ਕਤਾਰ_ਸੂਚਕਾਂਕ;

    ਜੇ (device->has_internally_synchronized_queues) {
        h = std::make_shared<vk_ਕਤਾਰ_handle_unsynchronized>();
        ਕਤਾਰ_info2.flags = eInternallySynchronizedKHR;
    } ਨਹੀਂਤਾਂ {
        h = std::make_shared<vk_ਕਤਾਰ_handle_synchronized>();
    }

    h->queue = device->device.getQueue2(ਕਤਾਰ_info2);
    h->device = device;
    // Avoid concurrent submissions on NVIDIA due to driver bug.
    ਜੇ (device->vendor_id == VK_VENDOR_ID_NVIDIA) {
        h->device_submit_mutex = &device->ਕਤਾਰ_submit_mutex;
    }
    q->handle = h;

    q->cmd_pool.init(device, q.get());

    q->stage_flags = stage_flags;
    ਵਾਪਸ q;
}

std::unique_ptr<vk_ਕਤਾਰ> ggml_vk_create_aliased_ਕਤਾਰ(vk_device& device, ਅਚਲ std::unique_ptr<vk_ਕਤਾਰ>& ਸਰੋਤ) {
    std::lock_guard<std::recursive_mutex> guard(device->mutex);
    ਸਵੈਚਾਲਿਤ q = std::make_unique<vk_ਕਤਾਰ>();
    q->handle = ਸਰੋਤ->handle;
    q->ਕਤਾਰ_family_ਸੂਚਕਾਂਕ = ਸਰੋਤ->ਕਤਾਰ_family_ਸੂਚਕਾਂਕ;
    q->stage_flags = ਸਰੋਤ->stage_flags;
    q->transfer_only = ਸਰੋਤ->transfer_only;
    q->cmd_pool.init(device, q.get());
    ਵਾਪਸ q;
}

vk_context ggml_vk_create_context(ggml_backend_vk_context * ctx, vk_command_pool& p) {
    vk_context ਨਤੀਜਾ = std::make_shared<vk_context_struct>();
    VK_LOG_DEBUG("ggml_vk_create_context(" << ਨਤੀਜਾ << ")");
    ctx->gc.contexts.emplace_back(ਨਤੀਜਾ);
    ਨਤੀਜਾ->p = &p;
    ਵਾਪਸ ਨਤੀਜਾ;
}

vk_context ggml_vk_create_temporary_context(vk_command_pool& p) {
    vk_context ਨਤੀਜਾ = std::make_shared<vk_context_struct>();
    VK_LOG_DEBUG("ggml_vk_create_temporary_context(" << ਨਤੀਜਾ << ")");
    ਨਤੀਜਾ->p = &p;
    ਵਾਪਸ ਨਤੀਜਾ;
}

ਸਥਿਰ vk_semaphore * ggml_vk_create_ਦੁਆਧਾਰੀ_semaphore(ggml_backend_vk_context * ctx) {
    VK_LOG_DEBUG("ggml_vk_create_timeline_semaphore()");
    vk::SemaphoreTypeCreateInfo tci{ vk::SemaphoreType::e_ਦੁਆਧਾਰੀ, 0 };
    vk::SemaphoreCreateInfo ci{};
    ci.ਸੈੱਟ_PNext(&tci);
    vk::Semaphore semaphore = ctx->device->device.createSemaphore(ci);
    ctx->gc.semaphores.push_back({ semaphore, 0 });
    ਵਾਪਸ &ctx->gc.semaphores[ctx->gc.semaphores.size() - 1];
}

ਸਥਿਰ vk_semaphore * ggml_vk_create_timeline_semaphore(ggml_backend_vk_context * ctx) {
    VK_LOG_DEBUG("ggml_vk_create_timeline_semaphore()");
    ਜੇ (ctx->semaphore_idx >= ctx->gc.tl_semaphores.size()) {
        vk::SemaphoreTypeCreateInfo tci{ vk::SemaphoreType::eTimeline, 0 };
        vk::SemaphoreCreateInfo ci{};
        ci.ਸੈੱਟ_PNext(&tci);
        vk::Semaphore semaphore = ctx->device->device.createSemaphore(ci);
        ctx->gc.tl_semaphores.push_back({ semaphore, 0 });
    }
    ਵਾਪਸ &ctx->gc.tl_semaphores[ctx->semaphore_idx++];
}

ਸਥਿਰ vk::Event ggml_vk_create_event(ggml_backend_vk_context * ctx) {
    ਜੇ (ctx->event_idx >= ctx->gc.events.size()) {
        ctx->gc.events.push_back(ctx->device->device.createEvent({}));
    }
    ਵਾਪਸ ctx->gc.events[ctx->event_idx++];
}

ਖਾਲੀ ggml_vk_command_pool_cleanup(vk_device& device, vk_command_pool& p) {
    VK_LOG_DEBUG("ggml_vk_command_pool_cleanup()");

    // Requires command buffers to be done
    device->device.resetCommandPool(p.pool);
    // Don't clear the command buffers and mark them as not in use.
    // This allows us to reuse them
    ਲਈ (ਸਵੈਚਾਲਿਤ& cmd_ਬਫਰ : p.cmd_buffers) {
        cmd_ਬਫਰ.ਵਿੱਚ_use = ਝੂਠ;
    }
}

ਖਾਲੀ ggml_vk_ਕਤਾਰ_command_pools_cleanup(vk_device& device) {
    VK_LOG_DEBUG("ggml_vk_queue_command_pools_cleanup()");

    // Arbitrary frequency to cleanup/reuse command buffers
    ਸਥਿਰ constexpr uint32_t cleanup_frequency = 10;

    ਜੇ (device->compute_ਕਤਾਰ && device->compute_ਕਤਾਰ->cmd_pool.buffers_ਵਿੱਚ_use() >= cleanup_frequency) {
        ggml_vk_command_pool_cleanup(device, device->compute_ਕਤਾਰ->cmd_pool);
    }
    ਜੇ (device->transfer_ਕਤਾਰ && device->transfer_ਕਤਾਰ->cmd_pool.buffers_ਵਿੱਚ_use() >= cleanup_frequency) {
        ggml_vk_command_pool_cleanup(device, device->transfer_ਕਤਾਰ->cmd_pool);
    }
}

vk_subbuffer ggml_vk_subbuffer(ਅਚਲ ggml_backend_vk_context* ctx, ਅਚਲ vk_ਬਫਰ& buf, size_t ਵਿੱਥ) {
    ਵਾਪਸ { buf, ਵਿੱਥ, ggml_vk_get_max_ਬਫਰ_range(ctx, buf, ਵਿੱਥ) };
}

ਖਾਲੀ ggml_vk_sync_buffers(ggml_backend_vk_context* ctx, vk_context& subctx) {
    VK_LOG_DEBUG("ggml_vk_sync_buffers()");

    ਅਚਲ ਤਾਰਕਿਕ transfer_ਕਤਾਰ = subctx->p->q->transfer_only;

    ਜੇ (ctx) {
        ctx->prealloc_x_need_sync = ctx->prealloc_y_need_sync = ctx->prealloc_ਵੰਡੋ_k_need_sync = ਝੂਠ;
    }

    subctx->s->ਬਫਰ->buf.pipelineBarrier(
        subctx->p->q->stage_flags,
        subctx->p->q->stage_flags,
        {},
        { {
          { !transfer_ਕਤਾਰ ? (vk::Access_Flag_ਬਿੱਟ::eShaderRead | vk::Access_Flag_ਬਿੱਟ::eShaderWrite | vk::Access_Flag_ਬਿੱਟ::eTransferRead | vk::Access_Flag_ਬਿੱਟ::eTransferWrite) : (vk::Access_Flag_ਬਿੱਟ::eTransferRead | vk::Access_Flag_ਬਿੱਟ::eTransferWrite) },
          { !transfer_ਕਤਾਰ ? (vk::Access_Flag_ਬਿੱਟ::eShaderRead | vk::Access_Flag_ਬਿੱਟ::eShaderWrite | vk::Access_Flag_ਬਿੱਟ::eTransferRead | vk::Access_Flag_ਬਿੱਟ::eTransferWrite) : (vk::Access_Flag_ਬਿੱਟ::eTransferRead | vk::Access_Flag_ਬਿੱਟ::eTransferWrite) }
        } },
        {},
        {}
    );
}

ਸਥਿਰ ਖਾਲੀ ggml_vk_reset_event(vk_context& ctx, vk::Event& event) {
    VK_LOG_DEBUG("ggml_vk_set_event()");

    ctx->s->ਬਫਰ->buf.resetEvent(
        event,
        ctx->p->q->stage_flags
    );
}

ਖਾਲੀ ggml_vk_ਸੈੱਟ_event(vk_context& ctx, vk::Event& event) {
    VK_LOG_DEBUG("ggml_vk_set_event()");

    ctx->s->ਬਫਰ->buf.ਸੈੱਟ_Event(
        event,
        ctx->p->q->stage_flags
    );
}

ਖਾਲੀ ggml_vk_wait_events(vk_context& ctx, std::vector<vk::Event>&& events) {
    VK_LOG_DEBUG("ggml_vk_wait_events()");
    ਜੇ (events.ਖਾਲੀ_ਥਾਵਾਂ()) {
        ਵਾਪਸ;
    }

    ctx->s->ਬਫਰ->buf.waitEvents(
        events,
        ctx->p->q->stage_flags,
        ctx->p->q->stage_flags,
        {},
        {},
        {}
    );
}

ਸਥਿਰ vk_fa_tuning_params get_fa_tuning_params_scalar(ਅਚਲ vk_device& device, uint32_t hsk, uint32_t hsv, uint32_t n_ਪੰਕਤੀਆਂ, uint32_t n_kv, ggml_type k_type, ggml_type v_type, ਤਾਰਕਿਕ f32acc) {

    vk_fa_tuning_params ਨਤੀਜਾ{};
    ਨਤੀਜਾ.path = FA_SCALAR;

    ਜੇ (device->vendor_id == VK_VENDOR_ID_INTEL) {
        // Disable subgroup use due to performance issues when enforcing subgroup sizes
        ਨਤੀਜਾ.subgroup_ਮਾਪ = 32;
        ਨਤੀਜਾ.disable_subgroups = ਸੱਚ;
    } ਨਹੀਂਤਾਂ ਜੇ (device->vendor_id == VK_VENDOR_ID_AMD && device->architecture != AMD_GCN) {
        ਨਤੀਜਾ.subgroup_ਮਾਪ = n_ਪੰਕਤੀਆਂ < 4 ? 32 : device->subgroup_ਮਾਪ;
    } ਨਹੀਂਤਾਂ {
        ਨਤੀਜਾ.subgroup_ਮਾਪ = device->subgroup_ਮਾਪ;
    }

    // Row split splits the workgroup so that synchronization only has to happen within subgroups, which avoids barriers
    uint32_t ਪੰਕਤੀ_ਵੰਡੋ_max_hsk = 64;
    ਜੇ (device->vendor_id == VK_VENDOR_ID_AMD && device->architecture != AMD_GCN && !device->uma) {
        ਪੰਕਤੀ_ਵੰਡੋ_max_hsk = n_ਪੰਕਤੀਆਂ <= 8 ? 64 : 128;
    }
    ਨਤੀਜਾ.ਪੰਕਤੀ_ਵੰਡੋ = (n_ਪੰਕਤੀਆਂ < 4 || hsk <= ਪੰਕਤੀ_ਵੰਡੋ_max_hsk) ? 1 : 4;

    ਜੇ (ਨਤੀਜਾ.subgroup_ਮਾਪ > 32 && (n_ਪੰਕਤੀਆਂ < 4 || hsk < (ਨਤੀਜਾ.ਪੰਕਤੀ_ਵੰਡੋ == 1 ? 128 : 64))) {
        ਨਤੀਜਾ.workgroup_ਮਾਪ = ਨਤੀਜਾ.subgroup_ਮਾਪ * 2;
    } ਨਹੀਂਤਾਂ {
        ਨਤੀਜਾ.workgroup_ਮਾਪ = ਨਤੀਜਾ.subgroup_ਮਾਪ * 4;
    }

    ਅਚਲ uint32_t D = hsk | hsv;

    ਅਚਲ ਤਾਰਕਿਕ reduce_ਖੰਡ_ਪੰਕਤੀਆਂ = D & 8 || n_kv < 1024 || device->vendor_id == VK_VENDOR_ID_INTEL;

    ਜੇ (n_ਪੰਕਤੀਆਂ == 1) {
        ਨਤੀਜਾ.ਖੰਡ_ਪੰਕਤੀਆਂ = 1;
        ਨਤੀਜਾ.ਖੰਡ_ਖਾਨੇ_ਗਿਣਤੀ = 64;
    } ਨਹੀਂਤਾਂ {
        // row_split 1 means higher register use per row, so block size has to be adjusted
        ਜੇ (ਨਤੀਜਾ.ਪੰਕਤੀ_ਵੰਡੋ == 1) {
            ਨਤੀਜਾ.ਖੰਡ_ਪੰਕਤੀਆਂ = n_ਪੰਕਤੀਆਂ == 2 ? 2 : ((n_ਪੰਕਤੀਆਂ <= 4 || reduce_ਖੰਡ_ਪੰਕਤੀਆਂ) ? 4 : 8);
        } ਨਹੀਂਤਾਂ {
            ਨਤੀਜਾ.ਖੰਡ_ਪੰਕਤੀਆਂ = n_ਪੰਕਤੀਆਂ <= 4 ? 4 : ((n_ਪੰਕਤੀਆਂ <= 8 || reduce_ਖੰਡ_ਪੰਕਤੀਆਂ) ? 8 : 16);
        }

        ਨਤੀਜਾ.ਖੰਡ_ਖਾਨੇ_ਗਿਣਤੀ = (D & 8) ? 64 : 32;
    }

    ਅਚਲ uint32_t D_lsb = D ^ (D & (D-1));  // extract lowest set bit

    ਨਤੀਜਾ.d_ਵੰਡੋ = std::min(std::min(ਨਤੀਜਾ.subgroup_ਮਾਪ, 8u), D_lsb / 4);

    ਨਤੀਜਾ.shmem_staging = (device->vendor_id == VK_VENDOR_ID_NVIDIA && hsk < 256 && hsv < 256) ? 1 : 0;

    ਜੇ (!reduce_ਖੰਡ_ਪੰਕਤੀਆਂ && !ggml_vk_flash_attn_scalar_shmem_support(device, ਨਤੀਜਾ, hsk, hsv, f32acc, k_type, v_type)) {
        ਨਤੀਜਾ.ਖੰਡ_ਪੰਕਤੀਆਂ /= 2;
    }

    // On AMD RDNA, for small head sizes and big batch size the shader uses few registers, so too many subgroups get scheduled
    // at once and end up thrashing the cache. Fix this by setting a large (unused) shmem buffer that reduces occupancy.
    // This targets an occupancy of 4 subgroups per SIMD.
    ਜੇ (device->vendor_id == VK_VENDOR_ID_AMD && device->properties.limits.max_Compute_Shared_ਸਮ੍ਰਿਤੀ_ਮਾਪ == 65536) {
        ਜੇ (device->architecture != AMD_GCN && n_ਪੰਕਤੀਆਂ >= 64 && hsk <= 128) {
            // 30kb target for hsk > 64, 26kb for <= 64 due to smaller workgroup size
            // Values are guessed, tested on RDNA2
            ਨਤੀਜਾ.ਹੱਦਬੰਦੀ_occupancy_shmem = (hsk <= 64 ? 26 : 30) * 1024 / 4 / 4;
        } ਨਹੀਂਤਾਂ ਜੇ (device->architecture == AMD_GCN && n_ਪੰਕਤੀਆਂ <= 8 && hsk >= 256) {
            // Same thing for GCN, with an occupancy target of 2 subgroups per SIMD.
            // Here low-batch FA with large head size is affected.
            // n_rows < 4 switch because workgroup size switches from 128 to 256 there.
            ਨਤੀਜਾ.ਹੱਦਬੰਦੀ_occupancy_shmem = (n_ਪੰਕਤੀਆਂ < 4 ? 14 : 26) * 1024 / 4 / 4;
        }
    }

    ਵਾਪਸ ਨਤੀਜਾ;
}

ਸਥਿਰ vk_fa_tuning_params get_fa_tuning_params_coopmat1(ਅਚਲ vk_device& device, uint32_t hsk, uint32_t hsv, uint32_t n_ਪੰਕਤੀਆਂ, uint32_t n_kv, ggml_type k_type, ggml_type v_type, ਤਾਰਕਿਕ f32acc) {
    GGML_ਅਣਵਰਤਿਆ(n_ਪੰਕਤੀਆਂ);
    GGML_ਅਣਵਰਤਿਆ(n_kv);
    GGML_ਅਣਵਰਤਿਆ(k_type);
    GGML_ਅਣਵਰਤਿਆ(v_type);
    GGML_ਅਣਵਰਤਿਆ(f32acc);

    vk_fa_tuning_params ਨਤੀਜਾ{};
    ਨਤੀਜਾ.path = FA_COOPMAT1;

    ਅਚਲ uint32_t D = hsk | hsv;

    ਅਚਲ uint32_t coopmat_ਖੰਡ_ਪੰਕਤੀਆਂ = 16;
    ਅਚਲ uint32_t coopmat_ਖੰਡ_ਖਾਨੇ_ਗਿਣਤੀ = 16;

    ਅਚਲ uint32_t ਅੰਕ_subgroups = 4;

    ਨਤੀਜਾ.ਖੰਡ_ਪੰਕਤੀਆਂ = coopmat_ਖੰਡ_ਪੰਕਤੀਆਂ;
    ਨਤੀਜਾ.ਖੰਡ_ਖਾਨੇ_ਗਿਣਤੀ = coopmat_ਖੰਡ_ਖਾਨੇ_ਗਿਣਤੀ * ਅੰਕ_subgroups;
    ਨਤੀਜਾ.ਪੰਕਤੀ_ਵੰਡੋ = ਅੰਕ_subgroups;
    ਨਤੀਜਾ.subgroup_ਮਾਪ = device->subgroup_ਮਾਪ;
    ਨਤੀਜਾ.workgroup_ਮਾਪ = ਅੰਕ_subgroups * ਨਤੀਜਾ.subgroup_ਮਾਪ;

    ਅਚਲ uint32_t D_lsb = D ^ (D & (D-1));  // extract lowest set bit
    ਨਤੀਜਾ.d_ਵੰਡੋ = std::min(std::min(ਨਤੀਜਾ.subgroup_ਮਾਪ, 8u), D_lsb / 4);

    ਨਤੀਜਾ.shmem_staging = (device->vendor_id == VK_VENDOR_ID_NVIDIA && hsk < 256 && hsv < 256) ? 1 : 0;

    ਵਾਪਸ ਨਤੀਜਾ;
}

ਸਥਿਰ vk_fa_tuning_params get_fa_tuning_params_coopmat2(ਅਚਲ vk_device& device, uint32_t hsk, uint32_t hsv, uint32_t n_ਪੰਕਤੀਆਂ, uint32_t n_kv, ggml_type k_type, ggml_type v_type, ਤਾਰਕਿਕ f32acc) {
    GGML_ਅਣਵਰਤਿਆ(n_kv);
    GGML_ਅਣਵਰਤਿਆ(f32acc);

    vk_fa_tuning_params ਨਤੀਜਾ{};
    ਨਤੀਜਾ.path = FA_COOPMAT2;

    ਅਚਲ uint32_t D = hsk | hsv;

    ਅਚਲ ਤਾਰਕਿਕ small_ਪੰਕਤੀਆਂ = n_ਪੰਕਤੀਆਂ < 32;

    ਜੇ (small_ਪੰਕਤੀਆਂ) {
        ਨਤੀਜਾ.ਖੰਡ_ਪੰਕਤੀਆਂ = 32;
        ਨਤੀਜਾ.ਖੰਡ_ਖਾਨੇ_ਗਿਣਤੀ = 32;
    } ਨਹੀਂਤਾਂ ਜੇ (ggml_is_quantized(k_type) || ggml_is_quantized(v_type) || hsk >= 256 || hsv >= 256) {
        ਨਤੀਜਾ.ਖੰਡ_ਪੰਕਤੀਆਂ = (hsk >= 512 || hsv >= 512) ? 32 : 64;
        ਨਤੀਜਾ.ਖੰਡ_ਖਾਨੇ_ਗਿਣਤੀ = 32;
    } ਨਹੀਂਤਾਂ {
        ਨਤੀਜਾ.ਖੰਡ_ਪੰਕਤੀਆਂ = 64;
        ਨਤੀਜਾ.ਖੰਡ_ਖਾਨੇ_ਗਿਣਤੀ = 64;
    }

    ਨਤੀਜਾ.subgroup_ਮਾਪ = device->subgroup_ਮਾਪ;
    ਨਤੀਜਾ.workgroup_ਮਾਪ = (small_ਪੰਕਤੀਆਂ && (D % 32) == 0) ? 256 : 128;

    ਵਾਪਸ ਨਤੀਜਾ;
}

vk_fa_tuning_params get_fa_tuning_params(ਅਚਲ vk_device& device, uint32_t hsk, uint32_t hsv, uint32_t n_ਪੰਕਤੀਆਂ, uint32_t n_kv, ggml_type k_type, ggml_type v_type, ਤਾਰਕਿਕ f32acc) {
    Fa_ਕੋਡ_ਰਸਤਾ path = device->coopmat2 ? FA_COOPMAT2 :
                      device->coopmat1_fa_support ? FA_COOPMAT1 : FA_SCALAR;

    ਜੇ (path == FA_COOPMAT2 && k_type == GGML_TYPE_BF16 && !device->coopmat2_bf16_support) {
        path = FA_COOPMAT1;
    }
    ਜੇ (path == FA_COOPMAT1 && k_type == GGML_TYPE_BF16 && !device->coopmat_bf16_support) {
        path = FA_SCALAR;
    }

    ਜੇ (path == FA_COOPMAT1 && device->architecture == vk_device_architecture::NVIDIA_TURING) {
        // Nvidia compiler bug, see https://github.com/ggml-org/llama.cpp/pull/19075#issuecomment-3820716090
        path = FA_SCALAR;
    }

    ਜੇ (path == FA_COOPMAT1) {
        ਤਾਰਕਿਕ shape_ok = (f32acc && device->coopmat_support_16x16x16_f32acc) ||
                        (!f32acc && device->coopmat_support_16x16x16_f16acc);
        ਅਚਲ vk_fa_tuning_params params = get_fa_tuning_params_coopmat1(device, hsk, hsv, n_ਪੰਕਤੀਆਂ, n_kv, k_type, v_type, f32acc);
        ਤਾਰਕਿਕ shmem_ok = ggml_vk_flash_attn_coopmat_shmem_support(device, params, hsk, hsv, f32acc, k_type, v_type);

        ਜੇ (!shape_ok || !shmem_ok) {
            path = FA_SCALAR;
        }
    }

    // scalar is faster than coopmat when N==1
    ਜੇ (n_ਪੰਕਤੀਆਂ == 1 && (path == FA_COOPMAT1 || path == FA_COOPMAT2)) {
        path = FA_SCALAR;
    }

    ਚੋਣ (path) {
    ਹਾਲਤ FA_SCALAR:
        ਵਾਪਸ get_fa_tuning_params_scalar(device, hsk, hsv, n_ਪੰਕਤੀਆਂ, n_kv, k_type, v_type, f32acc);
    ਹਾਲਤ FA_COOPMAT1:
        ਵਾਪਸ get_fa_tuning_params_coopmat1(device, hsk, hsv, n_ਪੰਕਤੀਆਂ, n_kv, k_type, v_type, f32acc);
    ਹਾਲਤ FA_COOPMAT2:
        ਵਾਪਸ get_fa_tuning_params_coopmat2(device, hsk, hsv, n_ਪੰਕਤੀਆਂ, n_kv, k_type, v_type, f32acc);
    ਮੂਲ:
        ਸੁੱਟੋ std::runtime_error("unsupported FaCodePath");
    }
}

vk_fa_pipeline_state get_fa_pipeline_state(ਅਚਲ vk_device& device, ਅਚਲ vk_fa_tuning_params& params, uint32_t hsk, uint32_t hsv, ਤਾਰਕਿਕ aligned, ਤਾਰਕਿਕ f32acc,
                                                  ਤਾਰਕਿਕ use_ਨਕਾਬ, ਤਾਰਕਿਕ use_ਨਕਾਬ_opt, ਤਾਰਕਿਕ use_logit_softcap, ਤਾਰਕਿਕ use_sparse, ggml_type k_type, ggml_type v_type) {
    ਅਚਲ ਤਾਰਕਿਕ old_amd_windows = device->vendor_id == VK_VENDOR_ID_AMD && device->driver_id == vk::DriverId::eAmdProprietary &&
                                 (device->architecture == AMD_GCN || device->architecture == AMD_RDNA1 || device->architecture == AMD_RDNA2);

    uint32_t flags = (use_ਨਕਾਬ_opt      ? 1 : 0) |
                     (use_ਨਕਾਬ          ? 2 : 0) |
                     (use_logit_softcap ? 4 : 0) |
                     (old_amd_windows   ? 8 : 0) |
                     (use_sparse        ? 16 : 0);

    ਅਚਲ uint32_t subgroup_ਮਾਪ = params.disable_subgroups ? 0 : params.subgroup_ਮਾਪ;

    ਵਾਪਸ vk_fa_pipeline_state{hsk, hsv, params.ਖੰਡ_ਪੰਕਤੀਆਂ, params.ਖੰਡ_ਖਾਨੇ_ਗਿਣਤੀ, params.d_ਵੰਡੋ, params.ਪੰਕਤੀ_ਵੰਡੋ, params.shmem_staging, params.path, params.workgroup_ਮਾਪ, subgroup_ਮਾਪ, aligned, f32acc, flags, params.ਹੱਦਬੰਦੀ_occupancy_shmem, k_type, v_type};
}

ਸਥਿਰ uint32_t fa_ਖੰਡ_ਬਾਈਟਾਂ(ggml_type t) {
    ਜੇ (t == GGML_TYPE_F32) {
        ਵਾਪਸ 16u;
    }
    ਵਾਪਸ (uint32_t) ggml_type_ਮਾਪ(t);
}

ਸਥਿਰ std::vector<uint32_t> get_fa_spec_constants(ਅਚਲ vk_fa_pipeline_state& state) {
    ਵਾਪਸ {
        /* 0 WorkGroupSize   */ state.workgroup_ਮਾਪ,
        /* 1 Br              */ state.Br,
        /* 2 Bc              */ state.Bc,
        /* 3 HSK             */ state.HSK,
        /* 4 HSV             */ state.HSV,
        /* 5 Clamp           */ static_cast<uint32_t>(!state.aligned),
        /* 6 D_split         */ state.D_ਵੰਡੋ,
        /* 7 row_split       */ state.ਪੰਕਤੀ_ਵੰਡੋ,
        /* 8 SubGroupSize    */ state.subgroup_ਮਾਪ,
        /* 9 SHMEM_STAGING   */ state.shmem_staging ? 1u : 0u,
        /*10 Flags           */ state.flags,
        /*11 LIMIT_OCCUPANCY_SHMEM */ state.ਹੱਦਬੰਦੀ_occupancy_shmem,
        /*12 FaTypeK         */ static_cast<uint32_t>(state.k_type),
        /*13 FaTypeV         */ static_cast<uint32_t>(state.v_type),
        /*14 FaBlockBytesK   */ fa_ਖੰਡ_ਬਾਈਟਾਂ(state.k_type),
        /*15 FaBlockBytesV   */ fa_ਖੰਡ_ਬਾਈਟਾਂ(state.v_type),
    };
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_vk_matmul_shmem_support(ਅਚਲ vk_device& device, ਅਚਲ std::vector<uint32_t>& warptile, ਤਾਰਕਿਕ mul_mat_id, ggml_type src0_type) {

    uint32_t lut_ਮਾਪ = 0;
    ਚੋਣ (src0_type) {
    ਹਾਲਤ GGML_TYPE_IQ1_S:
    ਹਾਲਤ GGML_TYPE_IQ1_M:
        // Regular matmul uses the compact uint16_t IQ1 grid; the expanded
        // uint32_t grid is only enabled for the q8_1/int-dot vector path.
        lut_ਮਾਪ = 2*2048;
        ਰੋਕੋ;
    ਹਾਲਤ GGML_TYPE_IQ2_XXS:
        lut_ਮਾਪ = 8*256;
        ਰੋਕੋ;
    ਹਾਲਤ GGML_TYPE_IQ2_XS:
        lut_ਮਾਪ = 8*512;
        ਰੋਕੋ;
    ਹਾਲਤ GGML_TYPE_IQ2_S:
        lut_ਮਾਪ = 8*1024;
        ਰੋਕੋ;
    ਹਾਲਤ GGML_TYPE_IQ3_XXS:
        lut_ਮਾਪ = 4*256;
        ਰੋਕੋ;
    ਹਾਲਤ GGML_TYPE_IQ3_S:
        lut_ਮਾਪ = 4*512;
        ਰੋਕੋ;
    ਹਾਲਤ GGML_TYPE_IQ4_NL:
    ਹਾਲਤ GGML_TYPE_IQ4_XS:
    ਹਾਲਤ GGML_TYPE_MXFP4:
        lut_ਮਾਪ = 4*16;
        ਰੋਕੋ;
    ਹਾਲਤ GGML_TYPE_NVFP4:
        // Same kvalues budget as MXFP4 plus ue4m3_fp32_lut[128] (types.glsl, DATA_A_NVFP4).
        lut_ਮਾਪ = 4*16 + 128u * (uint32_t)ਆਕਾਰ(ਦਸ਼ਮਲਵ);
        ਰੋਕੋ;
    ਮੂਲ:
        ਰੋਕੋ;
    }

    // Needs to be kept up to date on shader changes
    // Needs to stay aligned with ggml_vk_mul_mm_spec.
    ਅਚਲ ਤਾਰਕਿਕ intel_shmem_stride_pad_zero = device->vendor_id == VK_VENDOR_ID_INTEL && device->coopmat_support &&
                                              device->driver_id == vk::DriverId::eIntelProprietaryWindows;
    ਅਚਲ uint32_t bank_conflict_ਵਿੱਥ = intel_shmem_stride_pad_zero ? 0 : (device->coopmat_support ? 8 : 1);
    ਅਚਲ uint32_t type_ਮਾਪ = device->fp16 ? ਆਕਾਰ(ggml_fp16_t) : ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ uint32_t warps = warptile[0] / warptile[10];

    ਅਚਲ uint32_t ਲੱਦੋ_bufs = (warptile[1] + warptile[2]) * (warptile[3] + bank_conflict_ਵਿੱਥ) * type_ਮਾਪ;
    ਅਚਲ uint32_t mmid_ਪੰਕਤੀ_ids = mul_mat_id ? (warptile[2] * 2 * ਆਕਾਰ(uint16_t)) : 0;
    ਅਚਲ uint32_t coopmat_stage = device->coopmat_support ? warptile[7] * warptile[8] / warps * ਆਕਾਰ(ਦਸ਼ਮਲਵ) : 0;

    ਅਚਲ uint32_t ਕੁੱਲ_ਮਾਪ = ਲੱਦੋ_bufs + mmid_ਪੰਕਤੀ_ids + coopmat_stage + lut_ਮਾਪ;
    ਅਚਲ ਤਾਰਕਿਕ supported = ਕੁੱਲ_ਮਾਪ <= device->properties.limits.max_Compute_Shared_ਸਮ੍ਰਿਤੀ_ਮਾਪ;

    VK_LOG_DEBUG("ggml_vk_matmul_shmem_support(warptile=(" << warptile[0] << "," << warptile[1] << "," << warptile[2] << "), "
                 "mul_mat_id=" << mul_mat_id << ", src0_type=" << ggml_type_name(src0_type) << ", supported=" << supported);

    ਵਾਪਸ supported;
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_vk_matmul_int_shmem_support(ਅਚਲ vk_device& device, ਅਚਲ std::vector<uint32_t>& warptile, ਤਾਰਕਿਕ mul_mat_id, ggml_type src0_type) {

    // FLOAT_TYPE in the shader is float16_t with fp16 support, otherwise float.
    ਅਚਲ uint32_t fp_ਮਾਪ   = device->fp16 ? 2u : 4u;
    ਅਚਲ uint32_t fp_align  = fp_ਮਾਪ;
    ਅਚਲ uint32_t fp2_ਮਾਪ  = 2u * fp_ਮਾਪ;
    ਅਚਲ uint32_t fp2_align = device->fp16 ? 4u : 8u;

    ਬਣਤਰ member { uint32_t size, align; };
    ਸਵੈਚਾਲਿਤ std430_ਮਾਪ = [](std::initializer_list<member> members) {
        uint32_t off = 0, struct_align = 1;
        ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ &m : members) {
            off = (off + m.align - 1) & ~(m.align - 1);
            off += m.size;
            struct_align = std::max(struct_align, m.align);
        }
        ਵਾਪਸ (off + struct_align - 1) & ~(struct_align - 1);
    };

    uint32_t ਖੰਡ_a_ਮਾਪ = 0;
    ਚੋਣ (src0_type) {
        ਹਾਲਤ GGML_TYPE_Q2_0:    ਖੰਡ_a_ਮਾਪ = std430_ਮਾਪ({{32, 4}, {fp_ਮਾਪ,  fp_align}});                  ਰੋਕੋ; // qs[8] + dm
        ਹਾਲਤ GGML_TYPE_Q4_0:    ਖੰਡ_a_ਮਾਪ = std430_ਮਾਪ({{16, 4}, {fp_ਮਾਪ,  fp_align}});                  ਰੋਕੋ; // qs[16/4] + dm
        ਹਾਲਤ GGML_TYPE_Q4_1:    ਖੰਡ_a_ਮਾਪ = std430_ਮਾਪ({{16, 4}, {fp2_ਮਾਪ, fp2_align}});                 ਰੋਕੋ; // qs[16/4] + dm(vec2)
        ਹਾਲਤ GGML_TYPE_Q5_0:    ਖੰਡ_a_ਮਾਪ = std430_ਮਾਪ({{16, 4}, {4, 4}, {fp_ਮਾਪ,  fp_align}});          ਰੋਕੋ; // qs[16/4] + qh + dm
        ਹਾਲਤ GGML_TYPE_Q5_1:    ਖੰਡ_a_ਮਾਪ = std430_ਮਾਪ({{16, 4}, {4, 4}, {fp2_ਮਾਪ, fp2_align}});         ਰੋਕੋ; // qs[16/4] + qh + dm(vec2)
        ਹਾਲਤ GGML_TYPE_Q8_0:    ਖੰਡ_a_ਮਾਪ = std430_ਮਾਪ({{32, 4}, {fp_ਮਾਪ,  fp_align}});                  ਰੋਕੋ; // qs[8] + dm
        ਹਾਲਤ GGML_TYPE_IQ4_XS:  ਖੰਡ_a_ਮਾਪ = std430_ਮਾਪ({{32, 4}, {fp_ਮਾਪ,  fp_align}});                  ਰੋਕੋ; // qs[8] + d
        ਹਾਲਤ GGML_TYPE_MXFP4:   ਖੰਡ_a_ਮਾਪ = std430_ਮਾਪ({{32, 4}, {fp_ਮਾਪ,  fp_align}});                  ਰੋਕੋ; // qs[8] + d
        ਹਾਲਤ GGML_TYPE_IQ4_NL:  ਖੰਡ_a_ਮਾਪ = std430_ਮਾਪ({{32, 4}, {fp_ਮਾਪ,  fp_align}});                  ਰੋਕੋ; // qs[8] + d
        ਹਾਲਤ GGML_TYPE_NVFP4:   ਖੰਡ_a_ਮਾਪ = std430_ਮਾਪ({{32, 4}, {fp2_ਮਾਪ, fp2_align}});                 ਰੋਕੋ; // qs[8] + d_scales(vec2)
        ਹਾਲਤ GGML_TYPE_Q2_K:    ਖੰਡ_a_ਮਾਪ = std430_ਮਾਪ({{ 8, 4}, {2, 2}, {fp2_ਮਾਪ, fp2_align}});         ਰੋਕੋ; // qs[2] + scales(u8vec2) + dm(vec2)
        ਹਾਲਤ GGML_TYPE_Q3_K:    ਖੰਡ_a_ਮਾਪ = std430_ਮਾਪ({{16, 4}, {fp2_ਮਾਪ, fp2_align}});                 ਰੋਕੋ; // qs[4] + d_scales(vec2)
        ਹਾਲਤ GGML_TYPE_Q4_K:    ਖੰਡ_a_ਮਾਪ = std430_ਮਾਪ({{16, 4}, {fp2_ਮਾਪ, fp2_align}});                 ਰੋਕੋ; // qs[4] + dm(vec2)
        ਹਾਲਤ GGML_TYPE_Q5_K:    ਖੰਡ_a_ਮਾਪ = std430_ਮਾਪ({{32, 4}, {fp2_ਮਾਪ, fp2_align}});                 ਰੋਕੋ; // qs[8] + dm(vec2)
        ਹਾਲਤ GGML_TYPE_Q6_K:    ਖੰਡ_a_ਮਾਪ = std430_ਮਾਪ({{32, 4}, {fp2_ਮਾਪ, fp2_align}});                 ਰੋਕੋ; // qs[8] + d_scales(vec2)
        ਹਾਲਤ GGML_TYPE_IQ3_S:   ਖੰਡ_a_ਮਾਪ = std430_ਮਾਪ({{32, 4}, {fp_ਮਾਪ,  fp_align}});                  ਰੋਕੋ; // qs[8] + d
        ਮੂਲ:
            ਵਾਪਸ ਝੂਠ;
    }

    // IQ3_S also copies its 512-entry grid into shared memory (types.glsl, init_iq_shmem)
    ਅਚਲ uint32_t lut_ਮਾਪ = (src0_type == GGML_TYPE_IQ3_S) ? 4*512 : 0;

    // block_b_cache: { int32_t qs[8]; FLOAT_TYPEV2 ds; }
    ਅਚਲ uint32_t ਖੰਡ_b_ਮਾਪ = std430_ਮਾਪ({{32, 4}, {fp2_ਮਾਪ, fp2_align}});

    ਅਚਲ uint32_t BM = warptile[1];
    ਅਚਲ uint32_t BN = warptile[2];
    // mul_mmq.comp: BK_STEP=1 for MUL_MAT_ID, 4 otherwise.
    ਅਚਲ uint32_t BK_ਕਦਮ = mul_mat_id ? 1u : 4u;

    ਅਚਲ uint32_t buf_a_ਮਾਪ = BM * BK_ਕਦਮ * ਖੰਡ_a_ਮਾਪ;
    ਅਚਲ uint32_t buf_b_ਮਾਪ = BN * BK_ਕਦਮ * ਖੰਡ_b_ਮਾਪ;
    ਅਚਲ uint32_t mmid_ਪੰਕਤੀ_ids = mul_mat_id ? (BN * 2u * (uint32_t)ਆਕਾਰ(uint16_t)) : 0u;

    ਅਚਲ uint32_t ਕੁੱਲ_ਮਾਪ = buf_a_ਮਾਪ + buf_b_ਮਾਪ + mmid_ਪੰਕਤੀ_ids + lut_ਮਾਪ;
    ਅਚਲ ਤਾਰਕਿਕ supported = ਕੁੱਲ_ਮਾਪ <= device->properties.limits.max_Compute_Shared_ਸਮ੍ਰਿਤੀ_ਮਾਪ;

    VK_LOG_DEBUG("ggml_vk_matmul_int_shmem_support(warptile=(" << warptile[0] << "," << warptile[1] << "," << warptile[2] << "), "
                 "mul_mat_id=" << mul_mat_id << ", src0_type=" << ggml_type_name(src0_type) << ", total=" << ਕੁੱਲ_ਮਾਪ << ", supported=" << supported);

    ਵਾਪਸ supported;
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_vk_matmul_cm1_int_shmem_support(ਅਚਲ vk_device& device, ਅਚਲ std::vector<uint32_t>& warptile, ਤਾਰਕਿਕ mul_mat_id, ggml_type src0_type) {

    ਤਾਰਕਿਕ kscales2 = ਝੂਠ;    // two scale sets per block
    ਤਾਰਕਿਕ has_dm   = ਝੂਠ;    // d+m as vec2 + b-side sum
    ਤਾਰਕਿਕ has_kvalues = ਝੂਠ;
    ਚੋਣ (src0_type) {
        ਹਾਲਤ GGML_TYPE_Q4_0: ਹਾਲਤ GGML_TYPE_Q5_0: ਹਾਲਤ GGML_TYPE_Q8_0:
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q4_1: ਹਾਲਤ GGML_TYPE_Q5_1:
        ਹਾਲਤ GGML_TYPE_Q4_K: ਹਾਲਤ GGML_TYPE_Q5_K:
            has_dm = ਸੱਚ;                          ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_IQ4_NL: ਹਾਲਤ GGML_TYPE_IQ4_XS: ਹਾਲਤ GGML_TYPE_MXFP4:
            has_kvalues = ਸੱਚ;                     ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q3_K: ਹਾਲਤ GGML_TYPE_Q6_K:
            kscales2 = ਸੱਚ;                        ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_NVFP4:
            kscales2 = ਸੱਚ; has_kvalues = ਸੱਚ;    ਰੋਕੋ;
        ਮੂਲ:
            ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ uint32_t BM         = warptile[1];
    ਅਚਲ uint32_t BN         = warptile[2];

    ਅਚਲ uint32_t BK      = 32;
    ਅਚਲ uint32_t BK_ਕਦਮ = mul_mat_id ? 2u : 4u;
    ਅਚਲ uint32_t QPITCH  = BK_ਕਦਮ * (BK / 4u) + 4u;
    ਅਚਲ uint32_t KSCALES = kscales2 ? 2u : 1u;

    uint32_t ਕੁੱਲ = 0;
    ਕੁੱਲ += BM * QPITCH * (uint32_t)ਆਕਾਰ(uint32_t);   // buf_a_qs
    ਕੁੱਲ += BN * QPITCH * (uint32_t)ਆਕਾਰ(uint32_t);   // buf_b_qs
    ਕੁੱਲ += has_dm ? (BM * BK_ਕਦਮ * 2u * (uint32_t)ਆਕਾਰ(ਦਸ਼ਮਲਵ))   // buf_a_dm (vec2)
                    : (BM * BK_ਕਦਮ * KSCALES * (uint32_t)ਆਕਾਰ(ਦਸ਼ਮਲਵ)); // buf_a_d
    ਕੁੱਲ += BN * BK_ਕਦਮ * (uint32_t)ਆਕਾਰ(ਦਸ਼ਮਲਵ);     // buf_b_d
    ਜੇ (has_dm) {
        ਕੁੱਲ += BN * BK_ਕਦਮ * (uint32_t)ਆਕਾਰ(ਦਸ਼ਮਲਵ); // buf_b_s
    }
    ਜੇ (has_kvalues) {
        ਕੁੱਲ += 16u * (uint32_t)ਆਕਾਰ(int8_t);         // cm1_kvalues[16]
    }
    ਜੇ (src0_type == GGML_TYPE_NVFP4 && !device->ocp_fp4) {
        ਕੁੱਲ += 128u * (uint32_t)ਆਕਾਰ(ਦਸ਼ਮਲਵ);         // ue4m3_fp32_lut[128]
    }
    ਜੇ (mul_mat_id) {
        ਕੁੱਲ += BN * 2u * (uint32_t)ਆਕਾਰ(uint16_t);   // row_ids[BN] (u16vec2)
    }

    ਅਚਲ ਤਾਰਕਿਕ supported = ਕੁੱਲ <= device->properties.limits.max_Compute_Shared_ਸਮ੍ਰਿਤੀ_ਮਾਪ;

    VK_LOG_DEBUG("ggml_vk_matmul_cm1_int_shmem_support(warptile=(" << warptile[0] << "," << warptile[1] << "," << warptile[2] << "), "
                 "mul_mat_id=" << mul_mat_id << ", src0_type=" << ggml_type_name(src0_type) << ", total=" << ਕੁੱਲ << ", supported=" << supported);

    ਵਾਪਸ supported;
}

ਸਥਿਰ ਅਚਲ std::unordered_map<std::string, uint32_t> rdna1_pipelines = {
    {"soft_max", 64}, {"im2col", 64},
    {"argmax", 64}, {"mul_mat_vec", 64},
    {"mul_mat_vec_f16", 32}, {"mul_mat_vec_f32_f16", 32}
};

ਸਥਿਰ ਅਚਲ std::unordered_map<std::string, uint32_t> rdna2_pipelines = {
    {"soft_max", 64}, {"im2col", 64},
};

ਸਥਿਰ std::vector<GpuPipelineConfig> gpu_pipeline_configs = {
    {
        vk_device_architecture::AMD_RDNA1,
        {
            rdna1_pipelines,
        },
        RDNA_DEFAULT_SUBGROUP_ਮਾਪ
    },
    {
        vk_device_architecture::AMD_RDNA2,
        {
            rdna2_pipelines,
        },
        RDNA_DEFAULT_SUBGROUP_ਮਾਪ
    },
};

uint32_t get_subgroup_ਮਾਪ(ਅਚਲ std::string &pipeline_name, ਅਚਲ vk_device_architecture &arch) {
    ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ &config : gpu_pipeline_configs) {
        ਜੇ (config.arch == arch) {
            ਸਵੈਚਾਲਿਤ pipIt = config.pipelines.find(pipeline_name);
            ਜੇ (pipIt != config.pipelines.end()) {
                ਵਾਪਸ pipIt->second;
            }
            std::vector<std::pair<std::string, uint32_t>> sorted_pipelines(config.pipelines.begin(), config.pipelines.end());
            std::sort(sorted_pipelines.begin(), sorted_pipelines.end(),
                      [](ਅਚਲ ਸਵੈਚਾਲਿਤ &a, ਅਚਲ ਸਵੈਚਾਲਿਤ &b) { ਵਾਪਸ a.ਪਹਿਲਾ.size() > b.ਪਹਿਲਾ.size(); });
            ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ &ਇੰਦਰਾਜ : sorted_pipelines) {
                ਜੇ (pipeline_name.find(ਇੰਦਰਾਜ.ਪਹਿਲਾ) != std::string::npos) {
                    ਵਾਪਸ ਇੰਦਰਾਜ.second;
                }
            }
            ਵਾਪਸ config.default_subgroup_ਮਾਪ;
        }
    }
    ਵਾਪਸ 0; // If no matching configuration is found
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_vk_fa_type_needs_shmem(ggml_type type) {
    ਚੋਣ (type) {
    ਹਾਲਤ GGML_TYPE_IQ4_NL:
        ਵਾਪਸ ਸੱਚ;
    ਮੂਲ:
        ਵਾਪਸ ਝੂਠ;
    }
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_vk_fa_scalar_ਵਰਤਦਾ_mmq(ਅਚਲ vk_device& device, ggml_type k_type, ggml_type v_type) {
#ਜੇ defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
    ਵਾਪਸ device->integer_dot_ਗੁਣਨਫਲ && device->subgroup_clustered &&
           !ggml_vk_fa_type_needs_shmem(v_type) &&
           (k_type == GGML_TYPE_Q4_0 || k_type == GGML_TYPE_Q4_1 ||
            k_type == GGML_TYPE_Q5_0 || k_type == GGML_TYPE_Q5_1 ||
            k_type == GGML_TYPE_Q8_0);
#ਨਹੀਂਤਾਂ
    GGML_ਅਣਵਰਤਿਆ(device);
    GGML_ਅਣਵਰਤਿਆ(k_type);
    GGML_ਅਣਵਰਤਿਆ(v_type);
    ਵਾਪਸ ਝੂਠ;
#ਜੇ_ਅੰਤ
}

ਖਾਲੀ ggml_vk_ਲੱਦੋ_shaders(vk_device& device, vk_pipeline requested) {
    VK_LOG_DEBUG("ggml_vk_load_shaders(" << device->name << ")");

    // some shaders have a minimum subgroup size
    ਅਚਲ uint32_t subgroup_ਮਾਪ_8 = std::max(device->subgroup_ਮਾਪ, 8u);
    ਅਚਲ uint32_t subgroup_ਮਾਪ_16 = std::max(device->subgroup_ਮਾਪ, 16u);
    ਅਚਲ uint32_t subgroup_ਮਾਪ_32 = std::max(device->subgroup_ਮਾਪ, 32u);

    // clamp WARP for l_/m_ warptiles so WM <= BM (breaks on subgroupSize > 64)
    ਅਚਲ uint32_t mm_warp_8  = std::min(subgroup_ਮਾਪ_8,  64u);
    ਅਚਲ uint32_t mm_warp_16 = std::min(subgroup_ਮਾਪ_16, 64u);

    ਅਚਲ uint32_t mul_mat_subgroup_ਮਾਪ = (device->vendor_id == VK_VENDOR_ID_INTEL && device->subgroup_ਮਾਪ_control) ? device->subgroup_min_ਮਾਪ : device->subgroup_ਮਾਪ;
    ਅਚਲ uint32_t mul_mat_subgroup_ਮਾਪ_8 = std::max(mul_mat_subgroup_ਮਾਪ, 8u);
    ਅਚਲ uint32_t mul_mat_subgroup_ਮਾਪ_16 = std::max(mul_mat_subgroup_ਮਾਪ, 16u);
    ਅਚਲ uint32_t mul_mat_subgroup_ਮਾਪ_32 = std::max(mul_mat_subgroup_ਮਾਪ, 32u);
    ਅਚਲ uint32_t mul_mat_mm_warp_8  = std::min(mul_mat_subgroup_ਮਾਪ_8,  64u);
    ਅਚਲ uint32_t mul_mat_mm_warp_16 = std::min(mul_mat_subgroup_ਮਾਪ_16, 64u);

    ਅਚਲ ਤਾਰਕਿਕ subgroup_min_ਮਾਪ_16 = (!device->subgroup_ਮਾਪ_control && device->subgroup_ਮਾਪ >= 16) ||
                                      (device->subgroup_ਮਾਪ_control && device->subgroup_max_ਮਾਪ >= 16);

    // mulmat
    // Warptile layout (indices match mul_mm.comp constantIDs):
    //   [0..9]  : BLOCK_SIZE, BM, BN, BK, WM, WN, WMITER, TM, TN, TK
    //   [10]    : WARP / required_subgroup_size (read via WARP_SIZE_IDX)
    ਸਥਿਰ constexpr size_t WARP_ਮਾਪ_IDX = 10;
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
    ਅਚਲ ਤਾਰਕਿਕ cm1_use_wave32 = device->vendor_id == VK_VENDOR_ID_AMD &&
                                device->subgroup_ਮਾਪ_control &&
                                device->subgroup_min_ਮਾਪ <= 32 && device->subgroup_max_ਮਾਪ >= 32;
    ਅਚਲ uint32_t cm1_sg = cm1_use_wave32 ? 32 : device->subgroup_ਮਾਪ;

    vk_pipeline wait_pipeline;
    CompileTask claimed_task {};
    ਤਾਰਕਿਕ has_claimed_task = ਝੂਠ;

    // The rest of the walk reads and writes shared device state, so hold the
    // lock until we're done deciding what to compile.
    std::unique_lock<std::mutex> compile_ਤਾਲਾ(device->compile_mutex);

    ਜੇ (device->coopmat2) {
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
        ਅਚਲ uint32_t mmqid_bk = device->coopmat2_decode_ਸਦਿਸ਼ ? 64u : 32u;
        l_warptile_mmqid = { 256, 128, 128, mmqid_bk, 1 };
        m_warptile_mmqid = { 256, 128, 64,  mmqid_bk, 0 };
        s_warptile_mmqid = { 256, 128, 64,  mmqid_bk, 0 };
        l_mmqid_wg_denoms = { 128, 128, 1 };
        m_mmqid_wg_denoms = { 128, 64, 1 };
        s_mmqid_wg_denoms = { 128, 64, 1 };

        l_align = 128;
        m_align =  64;
        s_align =  32;
    } ਨਹੀਂਤਾਂ {
        // Matrix cores require different warp group sizes
        ਅਚਲ uint32_t tm_l = device->coopmat_support ? device->coopmat_m : 4;
        ਅਚਲ uint32_t tm_m = device->coopmat_support ? device->coopmat_m : 4;
        ਅਚਲ uint32_t tm_s = device->coopmat_support ? device->coopmat_m : 2;
        ਅਚਲ uint32_t tn_l = device->coopmat_support ? device->coopmat_n : 4;
        ਅਚਲ uint32_t tn_m = device->coopmat_support ? device->coopmat_n : 2;
        ਅਚਲ uint32_t tn_s = device->coopmat_support ? device->coopmat_n : 2;
        ਅਚਲ uint32_t tk_l = device->coopmat_support ? device->coopmat_k : 1;
        ਅਚਲ uint32_t tk_m = device->coopmat_support ? device->coopmat_k : 1;
        ਅਚਲ uint32_t tk_s = device->coopmat_support ? device->coopmat_k : 1;

        ਅਚਲ uint32_t itm = device->coopmat_int_m;
        ਅਚਲ uint32_t itn = device->coopmat_int_n;
        ਅਚਲ uint32_t itk = device->coopmat_int_k;

        ਅਚਲ uint32_t s_warptile_wm = device->subgroup_ਮਾਪ == 8 ? 8 : 32;

        l_warptile = { 128,             128, 128, 16, mm_warp_8 * 2, 64, 2, tm_l, tn_l, tk_l, mm_warp_8 };
        m_warptile = { 128,              64,  64, 16, mm_warp_8,     32, 2, tm_m, tn_m, tk_m, mm_warp_8 };
        s_warptile = { subgroup_ਮਾਪ_32, 32,  32, 16, s_warptile_wm, 32, 2, tm_s, tn_s, tk_s, subgroup_ਮਾਪ_8 };

        l_warptile_mmq = { 128,             128, 128, 32, mm_warp_8 * 2, 64, 2, tm_l, tn_l, tk_l, mm_warp_8 };
        m_warptile_mmq = { 128,              64,  64, 32, mm_warp_8,     32, 2, tm_m, tn_m, tk_m, mm_warp_8 };
        s_warptile_mmq = { subgroup_ਮਾਪ_32, 32,  32, 32, s_warptile_wm, 32, 2, tm_s, tn_s, tk_s, subgroup_ਮਾਪ_8 };

        // Integer MMQ has a smaller shared memory profile, but heavier register use
        l_warptile_mmq_int = { 128,             128, 128, 32, mm_warp_8 * 2, 64, 2, 4, 4, 1, mm_warp_8 };
        m_warptile_mmq_int = { 128,              64,  64, 32, mm_warp_8,     32, 2, 2, 2, 1, mm_warp_8 };
        s_warptile_mmq_int = { subgroup_ਮਾਪ_32, 32,  32, 32, s_warptile_wm, 32, 2, 2, 1, 1, subgroup_ਮਾਪ_8 };

        ਅਚਲ ਸਵੈਚਾਲਿਤ cm1_bs = [cm1_sg](uint32_t bm, uint32_t bn) {
            ਵਾਪਸ cm1_sg * (bm / std::min(cm1_sg, bm)) * (bn / 32);
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
        s_warptile_mmq_int_k = { subgroup_ਮਾਪ_32,   32,  32, 32, s_warptile_wm, 32, 1, 2, 1, 1, subgroup_ਮਾਪ_8 };

        l_warptile_id = { 128,                      128, 128, 16, mul_mat_mm_warp_16 * 2, 64, 2, tm_l, tn_l, tk_l, mul_mat_mm_warp_16 };
        m_warptile_id = { 128,                       64,  64, 16, mul_mat_mm_warp_16,     32, 2, tm_m, tn_m, tk_m, mul_mat_mm_warp_16 };
        s_warptile_id = { mul_mat_subgroup_ਮਾਪ_16,  32,  32, 16, s_warptile_wm,          32, 2, tm_s, tn_s, tk_s, mul_mat_subgroup_ਮਾਪ_16 };

        l_warptile_mmqid = { 128,                       128, 128, 32, mul_mat_mm_warp_8 * 2, 64, 2, tm_l, tn_l, tk_l, mul_mat_mm_warp_8 };
        m_warptile_mmqid = { 128,                        64,  64, 32, mul_mat_mm_warp_8,     32, 2, tm_m, tn_m, tk_m, mul_mat_mm_warp_8 };
        s_warptile_mmqid = { mul_mat_subgroup_ਮਾਪ_32,   32,  32, 32, s_warptile_wm,         32, 2, tm_s, tn_s, tk_s, mul_mat_subgroup_ਮਾਪ_8 };

        l_warptile_mmqid_int = { 128,                       128, 128, 32, mul_mat_mm_warp_8 * 2, 64, 2, 4, 4, 1, mul_mat_mm_warp_8 };
        m_warptile_mmqid_int = { 128,                        64,  64, 32, mul_mat_mm_warp_8,     32, 2, 2, 2, 1, mul_mat_mm_warp_8 };
        s_warptile_mmqid_int = { mul_mat_subgroup_ਮਾਪ_32,   32,  32, 32, s_warptile_wm,         32, 2, 2, 1, 1, mul_mat_subgroup_ਮਾਪ_8 };

        l_warptile_mmqid_int_k = { 128,                     128, 128, 32, mul_mat_mm_warp_16 * 2, 64, 1, 4, 4, 1, mul_mat_mm_warp_16 };
        m_warptile_mmqid_int_k = { 128,                      64,  64, 32, mul_mat_mm_warp_16,     32, 1, 2, 2, 1, mul_mat_mm_warp_16 };
        s_warptile_mmqid_int_k = { mul_mat_subgroup_ਮਾਪ_32, 32,  32, 32, s_warptile_wm,          32, 1, 2, 1, 1, mul_mat_subgroup_ਮਾਪ_16 };

        // chip specific tuning
        ਜੇ ((device->architecture == AMD_GCN) && (device->driver_id != vk::DriverId::eAmdProprietary)) {
            m_warptile_mmq = m_warptile_mmq_int = { 256, 64, 64, 32, 16, 16, 2, 2, 2, 1, 16 };
            m_warptile_mmqid = m_warptile_mmqid_int = { 256, 64, 64, 32, 16, 16, 2, 2, 2, 1, 16 };
        } ਨਹੀਂਤਾਂ ਜੇ (device->vendor_id == VK_VENDOR_ID_AMD && device->coopmat_support && device->driver_id != vk::DriverId::eAmdProprietary) {
            // This is intentionally using tx_m values, slight performance increase
            l_warptile = { 256, 128, 128, 16, mm_warp_8, 64, 2, tm_m, tn_m, tk_m, mm_warp_8 };
            l_warptile_mmq = l_warptile_mmq_int = { 256, 128, 128, 32, mm_warp_8, 64, 2, tm_m, tn_m, tk_m, mm_warp_8 };
            l_warptile_mmq_int_k = { 256, 128, 128, 32, mm_warp_16, 64, 1, 4, 2, 1, mm_warp_16 };
        } ਨਹੀਂਤਾਂ ਜੇ (device->vendor_id == VK_VENDOR_ID_QUALCOMM && device->coopmat_support) {
            m_warptile     = { 64, 64, 64, 16, 64, 64, 1, tm_l, tn_l, tk_l, 64 };
            m_warptile_mmq = { 64, 64, 64, 32, 64, 64, 1, tm_m, tn_m, tk_m, 64 };
        }

        l_mmq_wg_denoms = l_wg_denoms = {128, 128, 1 };
        m_mmq_wg_denoms = m_wg_denoms = { 64,  64, 1 };
        s_mmq_wg_denoms = s_wg_denoms = { 32,  32, 1 };
        l_align = 128;
        m_align =  64;
        s_align =  32;

        ਜੇ (device->vendor_id == VK_VENDOR_ID_INTEL && device->coopmat_support) {
            // Xe1/Xe2/Xe3 with coopmat enabled - warptile performance tuning
            l_warptile = { 512, 128, 128, 16, mm_warp_8, 32, 2, tm_l, tn_l, tk_l, mm_warp_8 };
            ਜੇ (device->architecture == INTEL_XE1) {
                l_warptile_mmq  = { 512, 256, 128, 32, 32, 32, 2, tm_l, tn_l, tk_l, 16 };
                l_mmq_wg_denoms = { 256, 128, 1 };
                l_align         = 32;  //set as BK
            } ਨਹੀਂਤਾਂ {
                l_warptile_mmq  = { 512, 128, 256, 32, 32, 32, 2, tm_l, tn_l, tk_l, 16 };
                l_mmq_wg_denoms = { 128, 256, 1 };
                l_align         = 32;  //set as BK
            }
        }

        ਅਚਲ ਤਾਰਕਿਕ use_cm1_int = device->coopmat_int_support &&
                                 (device->architecture == AMD_RDNA3 || device->architecture == AMD_RDNA4);

        ਲਈ (uint32_t i = 0; i < GGML_TYPE_ਗਿਣਤੀ; ++i) {
            ggml_type t = (ggml_type)i;
            // Disable medium and large matrix multiplication if not enough shared memory is available
            // Check mmq warptiles as the largest configuration
            // Throw an error if not enough for any matrix multiplication is available
            ਜੇ (!ggml_vk_matmul_shmem_support(device, s_warptile_mmq, ਝੂਠ, t)) {
                std::cerr << "ggml_vulkan: Error: Shared memory size too small for matrix multiplication." << std::ਸਤਰ_ਅੰਤ;
                ਸੁੱਟੋ std::runtime_error("Shared memory size too small for matrix multiplication.");
            } ਨਹੀਂਤਾਂ ਜੇ (!ggml_vk_matmul_shmem_support(device, m_warptile_mmq, ਝੂਠ, t)) {
                device->mul_mat_m[i] = ਝੂਠ;
                device->mul_mat_l[i] = ਝੂਠ;
            } ਨਹੀਂਤਾਂ ਜੇ (!ggml_vk_matmul_shmem_support(device, l_warptile_mmq, ਝੂਠ, t)) {
                device->mul_mat_l[i] = ਝੂਠ;
            }

            // Disable mul_mat_id if not enough shared memory is available
            ਜੇ (!ggml_vk_matmul_shmem_support(device, s_warptile_mmqid, ਸੱਚ, t)) {
                device->mul_mat_id_s[i] = ਝੂਠ;
                device->mul_mat_id_m[i] = ਝੂਠ;
                device->mul_mat_id_l[i] = ਝੂਠ;
            } ਨਹੀਂਤਾਂ ਜੇ (!ggml_vk_matmul_shmem_support(device, m_warptile_mmqid, ਸੱਚ, t)) {
                device->mul_mat_id_m[i] = ਝੂਠ;
                device->mul_mat_id_l[i] = ਝੂਠ;
            } ਨਹੀਂਤਾਂ ਜੇ (!ggml_vk_matmul_shmem_support(device, l_warptile_mmqid, ਸੱਚ, t)) {
                device->mul_mat_id_l[i] = ਝੂਠ;
            }

            // The q8_1 mmq path has its own (larger) shmem layout, check it separately.
            // K-quants and IQ3_S use the _int_k warptiles, others use _int.
            // cm1 splits k-tiles on the KSCALES==2 types and shares tiles between dense/id.
            ਅਚਲ ਤਾਰਕਿਕ is_k_quant = (t == GGML_TYPE_Q2_K || t == GGML_TYPE_Q3_K ||
                                     t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q5_K ||
                                     t == GGML_TYPE_Q6_K || t == GGML_TYPE_IQ3_S);
            ਅਚਲ ਤਾਰਕਿਕ cm1_k_tile = (t == GGML_TYPE_Q3_K || t == GGML_TYPE_Q6_K ||
                                     t == GGML_TYPE_NVFP4);

            ਅਚਲ ਸਵੈਚਾਲਿਤ & s_int   = use_cm1_int ? (cm1_k_tile ? s_warptile_mmq_cm1_int_k : s_warptile_mmq_cm1_int)
                                               : (is_k_quant  ? s_warptile_mmq_int_k     : s_warptile_mmq_int);
            ਅਚਲ ਸਵੈਚਾਲਿਤ & m_int   = use_cm1_int ? (cm1_k_tile ? m_warptile_mmq_cm1_int_k : m_warptile_mmq_cm1_int)
                                               : (is_k_quant  ? m_warptile_mmq_int_k     : m_warptile_mmq_int);
            ਅਚਲ ਸਵੈਚਾਲਿਤ & l_int   = use_cm1_int ? (cm1_k_tile ? l_warptile_mmq_cm1_int_k : l_warptile_mmq_cm1_int)
                                               : (is_k_quant  ? l_warptile_mmq_int_k     : l_warptile_mmq_int);
            ਅਚਲ ਸਵੈਚਾਲਿਤ & s_intid = use_cm1_int ? (cm1_k_tile ? s_warptile_mmq_cm1_int_k : s_warptile_mmq_cm1_int)
                                               : (is_k_quant  ? s_warptile_mmqid_int_k   : s_warptile_mmqid_int);
            ਅਚਲ ਸਵੈਚਾਲਿਤ & m_intid = use_cm1_int ? (cm1_k_tile ? m_warptile_mmq_cm1_int_k : m_warptile_mmq_cm1_int)
                                               : (is_k_quant  ? m_warptile_mmqid_int_k   : m_warptile_mmqid_int);
            ਅਚਲ ਸਵੈਚਾਲਿਤ & l_intid = use_cm1_int ? (cm1_k_tile ? l_warptile_mmq_cm1_int_k : l_warptile_mmq_cm1_int)
                                               : (is_k_quant  ? l_warptile_mmqid_int_k   : l_warptile_mmqid_int);

            ਅਚਲ ਸਵੈਚਾਲਿਤ int_shmem_support = [&](ਅਚਲ std::vector<uint32_t>& wt, ਤਾਰਕਿਕ id) {
                ਵਾਪਸ use_cm1_int ? ggml_vk_matmul_cm1_int_shmem_support(device, wt, id, t)
                                   : ggml_vk_matmul_int_shmem_support(device, wt, id, t);
            };

            ਜੇ (!int_shmem_support(s_int, ਝੂਠ)) {
                device->mul_mat_s_int[i] = ਝੂਠ;
                device->mul_mat_m_int[i] = ਝੂਠ;
                device->mul_mat_l_int[i] = ਝੂਠ;
            } ਨਹੀਂਤਾਂ ਜੇ (!int_shmem_support(m_int, ਝੂਠ)) {
                device->mul_mat_m_int[i] = ਝੂਠ;
                device->mul_mat_l_int[i] = ਝੂਠ;
            } ਨਹੀਂਤਾਂ ਜੇ (!int_shmem_support(l_int, ਝੂਠ)) {
                device->mul_mat_l_int[i] = ਝੂਠ;
            }

            ਜੇ (!int_shmem_support(s_intid, ਸੱਚ)) {
                device->mul_mat_id_s_int[i] = ਝੂਠ;
                device->mul_mat_id_m_int[i] = ਝੂਠ;
                device->mul_mat_id_l_int[i] = ਝੂਠ;
            } ਨਹੀਂਤਾਂ ਜੇ (!int_shmem_support(m_intid, ਸੱਚ)) {
                device->mul_mat_id_m_int[i] = ਝੂਠ;
                device->mul_mat_id_l_int[i] = ਝੂਠ;
            } ਨਹੀਂਤਾਂ ਜੇ (!int_shmem_support(l_intid, ਸੱਚ)) {
                device->mul_mat_id_l_int[i] = ਝੂਠ;
            }
        }
    }

    ਸਵੈਚਾਲਿਤ ਅਚਲ &ggml_vk_create_pipeline = [&](vk_device& device, vk_pipeline& ਅਧਾਰ_pipeline, ਅਚਲ ਅੱਖਰ *name, size_t spv_ਮਾਪ, ਅਚਲ ਖਾਲੀ* spv_ਅੰਕੜੇ, ਅਚਲ ਅੱਖਰ *entrypoint,
                                              uint32_t parameter_ਗਿਣਤੀ, uint32_t ਧੱਕੋ_constant_ਮਾਪ, std::array<uint32_t, 3> wg_denoms, ਅਚਲ std::vector<uint32_t>& specialization_constants,
                                              uint32_t align, ਤਾਰਕਿਕ disable_robustness = ਝੂਠ, ਤਾਰਕਿਕ require_ਭਰਿਆ_subgroups = ਝੂਠ, uint32_t required_subgroup_ਮਾਪ = 0) {

        ਜੇ (!require_ਭਰਿਆ_subgroups && required_subgroup_ਮਾਪ == 0) {
            required_subgroup_ਮਾਪ = get_subgroup_ਮਾਪ(name, device->architecture);
        }

        vk_pipeline *ptr = &ਅਧਾਰ_pipeline;

        ਪੂਰਨ ਅੰਕ_pipelines = 1;
#ਜੇ defined(VK_EXT_shader_64bit_indexing)
        ਜੇ (device->shader_64b_indexing) {
            ਅੰਕ_pipelines = 2;
        }
#ਜੇ_ਅੰਤ
        ਲਈ (ਪੂਰਨ i = 0; i < ਅੰਕ_pipelines; ++i, ptr = &(*ptr)->ਅਗਲਾ) {
            vk_pipeline &pipeline = *ptr;
            ਜੇ (!pipeline) {
                pipeline = std::make_shared<vk_pipeline_struct>();
            }
            ਜੇ (!pipeline->initialized) {
                pipeline->name = name;
                pipeline->parameter_ਗਿਣਤੀ = parameter_ਗਿਣਤੀ;
                pipeline->ਧੱਕੋ_constant_ਮਾਪ = ਧੱਕੋ_constant_ਮਾਪ;
                pipeline->wg_denoms = wg_denoms;
                pipeline->align = align;
                pipeline->initialized = ਸੱਚ;
#ਜੇ defined(VK_EXT_shader_64bit_indexing)
                pipeline->is_64b_indexing = (i == 1);
#ਜੇ_ਅੰਤ
            }

            // We only care about the pipeline this call asked for; the rest
            // (including the 64-bit indexing variant) are handled by their
            // own request_descriptor_sets / load_shaders calls.
            ਜੇ (pipeline.get() != requested.get()) {
                ਜਾਰੀ;
            }

            ਜੇ (pipeline->compiled) {
                ਜਾਰੀ;
            }

            wait_pipeline = pipeline;

            ਜੇ (!pipeline->compile_pending) {
                pipeline->compile_pending = ਸੱਚ;
                claimed_task.pipeline = pipeline;
                claimed_task.spv_ਮਾਪ = spv_ਮਾਪ;
                claimed_task.spv_ਅੰਕੜੇ = spv_ਅੰਕੜੇ;
                claimed_task.entrypoint = entrypoint;
                claimed_task.parameter_ਗਿਣਤੀ = parameter_ਗਿਣਤੀ;
                claimed_task.wg_denoms = wg_denoms;
                claimed_task.specialization_constants = specialization_constants;
                claimed_task.disable_robustness = disable_robustness;
                claimed_task.require_ਭਰਿਆ_subgroups = require_ਭਰਿਆ_subgroups;
                claimed_task.required_subgroup_ਮਾਪ = required_subgroup_ਮਾਪ;
                has_claimed_task = ਸੱਚ;
            }
        }
    };

    ਸਵੈਚਾਲਿਤ ਅਚਲ &ggml_vk_create_pipeline2 = [&](vk_device& device, vk_pipeline& pipeline, ਅਚਲ std::string &name, size_t spv_ਮਾਪ, ਅਚਲ ਖਾਲੀ* spv_ਅੰਕੜੇ, ਅਚਲ ਅੱਖਰ *entrypoint,
                                              uint32_t parameter_ਗਿਣਤੀ, uint32_t ਧੱਕੋ_constant_ਮਾਪ, std::array<uint32_t, 3> wg_denoms, ਅਚਲ std::vector<uint32_t>& specialization_constants,
                                              uint32_t align, ਤਾਰਕਿਕ disable_robustness = ਝੂਠ, ਤਾਰਕਿਕ require_ਭਰਿਆ_subgroups = ਝੂਠ, uint32_t required_subgroup_ਮਾਪ = 0) {
        ਵਾਪਸ ggml_vk_create_pipeline(device, pipeline, name.c_str(), spv_ਮਾਪ, spv_ਅੰਕੜੇ, entrypoint,
                                       parameter_ਗਿਣਤੀ, ਧੱਕੋ_constant_ਮਾਪ, wg_denoms, specialization_constants,
                                       align, disable_robustness, require_ਭਰਿਆ_subgroups, required_subgroup_ਮਾਪ);
    };

    // FA scalar has two SPIR-V modules (MMQ vs non-MMQ); FA cm1 has one. K/V
    // quant type is selected at runtime via the FaTypeK / FaTypeV spec constants.

    ਲਈ (ਸਵੈਚਾਲਿਤ &fa : device->pipeline_flash_attn_f32_f16) {
        ਜੇ (fa.ਪਹਿਲਾ.path != FA_SCALAR) ਜਾਰੀ;
        ਅਚਲ uint32_t Br = fa.ਪਹਿਲਾ.Br;
        ਅਚਲ uint32_t Bc = fa.ਪਹਿਲਾ.Bc;
        ਅਚਲ ਤਾਰਕਿਕ aligned = fa.ਪਹਿਲਾ.aligned;
        ਅਚਲ ਤਾਰਕਿਕ f32acc = fa.ਪਹਿਲਾ.f32acc;
        ਅਚਲ uint32_t fa_sgs = fa.ਪਹਿਲਾ.subgroup_ਮਾਪ;
        ਅਚਲ ਤਾਰਕਿਕ fa_ds = fa.ਪਹਿਲਾ.subgroup_ਮਾਪ == 0;

        ਅਚਲ ਤਾਰਕਿਕ bf16_kv = fa.ਪਹਿਲਾ.k_type == GGML_TYPE_BF16;
        ਅਚਲ ਤਾਰਕਿਕ use_mmq = ggml_vk_fa_scalar_ਵਰਤਦਾ_mmq(device, fa.ਪਹਿਲਾ.k_type, fa.ਪਹਿਲਾ.v_type);
        ਅਚਲ ਖਾਲੀ * spv_ਅੰਕੜੇ = ਖਾਲੀ_ਸੂਚਕ;
        size_t spv_ਮਾਪ = 0;
        ਅਚਲ ਅੱਖਰ *name = ਖਾਲੀ_ਸੂਚਕ;
        ਜੇ (bf16_kv) {
            spv_ਅੰਕੜੇ = flash_attn_f32_f16_fp32_ਅੰਕੜੇ;
            spv_ਮਾਪ = flash_attn_f32_f16_fp32_len;
            name = aligned ? "flash_attn_f32_bf16_aligned" : "flash_attn_f32_bf16";
        } ਨਹੀਂਤਾਂ ਜੇ (use_mmq) {
#ਜੇ defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
            ਜੇ (device->fp16) {
                ਜੇ (f32acc) { spv_ਅੰਕੜੇ = flash_attn_f32_f16_int8_ਅੰਕੜੇ;        spv_ਮਾਪ = flash_attn_f32_f16_int8_len; }
                ਨਹੀਂਤਾਂ        { spv_ਅੰਕੜੇ = flash_attn_f32_f16_f16acc_int8_ਅੰਕੜੇ; spv_ਮਾਪ = flash_attn_f32_f16_f16acc_int8_len; }
            } ਨਹੀਂਤਾਂ {
                spv_ਅੰਕੜੇ = flash_attn_f32_f16_fp32_int8_ਅੰਕੜੇ;
                spv_ਮਾਪ = flash_attn_f32_f16_fp32_int8_len;
            }
#ਜੇ_ਅੰਤ
            name = aligned ? "flash_attn_f32_f16_aligned" : "flash_attn_f32_f16";
        } ਨਹੀਂਤਾਂ {
            ਜੇ (device->fp16) {
                ਜੇ (device->dot2_f16) {
                    ਜੇ (f32acc) { spv_ਅੰਕੜੇ = flash_attn_f32_f16_dot2_ਅੰਕੜੇ;        spv_ਮਾਪ = flash_attn_f32_f16_dot2_len; }
                    ਨਹੀਂਤਾਂ        { spv_ਅੰਕੜੇ = flash_attn_f32_f16_dot2_f16acc_ਅੰਕੜੇ; spv_ਮਾਪ = flash_attn_f32_f16_dot2_f16acc_len; }
                } ਨਹੀਂਤਾਂ {
                    ਜੇ (f32acc) { spv_ਅੰਕੜੇ = flash_attn_f32_f16_ਅੰਕੜੇ;        spv_ਮਾਪ = flash_attn_f32_f16_len; }
                    ਨਹੀਂਤਾਂ        { spv_ਅੰਕੜੇ = flash_attn_f32_f16_f16acc_ਅੰਕੜੇ; spv_ਮਾਪ = flash_attn_f32_f16_f16acc_len; }
                }
            } ਨਹੀਂਤਾਂ {
                spv_ਅੰਕੜੇ = flash_attn_f32_f16_fp32_ਅੰਕੜੇ;
                spv_ਮਾਪ = flash_attn_f32_f16_fp32_len;
            }
            name = aligned ? "flash_attn_f32_f16_aligned" : "flash_attn_f32_f16";
        }
        ggml_vk_create_pipeline(device, fa.second, name, spv_ਮਾਪ, spv_ਅੰਕੜੇ, "main", 8,
                                ਆਕਾਰ(vk_flash_attn_ਧੱਕੋ_constants), {Br, 1, 1},
                                get_fa_spec_constants(fa.ਪਹਿਲਾ), aligned ? Bc : 1, ਸੱਚ,
                                !fa_ds, !fa_ds ? fa_sgs : 0);
    }

#ਜੇ defined(VK_KHR_cooperative_ਮੈਟ੍ਰਿਕਸ) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    ਜੇ (device->coopmat1_fa_support) {
        ਲਈ (ਸਵੈਚਾਲਿਤ &fa : device->pipeline_flash_attn_f32_f16) {
            ਜੇ (fa.ਪਹਿਲਾ.path != FA_COOPMAT1) ਜਾਰੀ;
            ਅਚਲ uint32_t Br = fa.ਪਹਿਲਾ.Br;
            ਅਚਲ uint32_t Bc = fa.ਪਹਿਲਾ.Bc;
            ਅਚਲ ਤਾਰਕਿਕ aligned = fa.ਪਹਿਲਾ.aligned;
            ਅਚਲ ਤਾਰਕਿਕ f32acc = fa.ਪਹਿਲਾ.f32acc;
            ਅਚਲ uint32_t fa_sgs = fa.ਪਹਿਲਾ.subgroup_ਮਾਪ;
            ਅਚਲ ਤਾਰਕਿਕ fa_ds = fa.ਪਹਿਲਾ.subgroup_ਮਾਪ == 0;

            ਅਚਲ ਤਾਰਕਿਕ bf16_kv = fa.ਪਹਿਲਾ.k_type == GGML_TYPE_BF16;

            ਅਚਲ ਖਾਲੀ * spv_ਅੰਕੜੇ;
            size_t spv_ਮਾਪ;
            ਅਚਲ ਅੱਖਰ *name;
            ਜੇ (bf16_kv) {
#ਜੇ defined(VK_KHR_shader_bfloat16) && defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
                ਜੇ (!device->coopmat_bf16_support) ਜਾਰੀ;
                spv_ਅੰਕੜੇ = flash_attn_f32_f16_bf16_cm1_ਅੰਕੜੇ;
                spv_ਮਾਪ = flash_attn_f32_f16_bf16_cm1_len;
                name = aligned ? "flash_attn_f32_bf16_aligned_cm1" : "flash_attn_f32_bf16_cm1";
#ਨਹੀਂਤਾਂ
                ਜਾਰੀ;
#ਜੇ_ਅੰਤ
            } ਨਹੀਂਤਾਂ {
                ਜੇ (f32acc) { spv_ਅੰਕੜੇ = flash_attn_f32_f16_cm1_ਅੰਕੜੇ;        spv_ਮਾਪ = flash_attn_f32_f16_cm1_len; }
                ਨਹੀਂਤਾਂ        { spv_ਅੰਕੜੇ = flash_attn_f32_f16_f16acc_cm1_ਅੰਕੜੇ; spv_ਮਾਪ = flash_attn_f32_f16_f16acc_cm1_len; }
                name = aligned ? "flash_attn_f32_f16_aligned_cm1" : "flash_attn_f32_f16_cm1";
            }
            ggml_vk_create_pipeline(device, fa.second, name, spv_ਮਾਪ, spv_ਅੰਕੜੇ, "main", 8,
                                    ਆਕਾਰ(vk_flash_attn_ਧੱਕੋ_constants), {Br, 1, 1},
                                    get_fa_spec_constants(fa.ਪਹਿਲਾ), aligned ? Bc : 1, ਸੱਚ,
                                    !fa_ds, !fa_ds ? fa_sgs : 0);
        }
    }
#ਜੇ_ਅੰਤ

#ਜੇ defined(VK_NV_cooperative_matrix2) && defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
    ਜੇ (device->coopmat2) {
        ਲਈ (ਸਵੈਚਾਲਿਤ &fa : device->pipeline_flash_attn_f32_f16) {
            ਜੇ (fa.ਪਹਿਲਾ.path != FA_COOPMAT2) ਜਾਰੀ;
            ਅਚਲ uint32_t Br = fa.ਪਹਿਲਾ.Br;
            ਅਚਲ uint32_t Bc = fa.ਪਹਿਲਾ.Bc;
            ਅਚਲ ਤਾਰਕਿਕ aligned = fa.ਪਹਿਲਾ.aligned;
            ਅਚਲ ਤਾਰਕਿਕ f32acc = fa.ਪਹਿਲਾ.f32acc;

            ਅਚਲ ਤਾਰਕਿਕ bf16_kv = fa.ਪਹਿਲਾ.k_type == GGML_TYPE_BF16;
            ਅਚਲ ਖਾਲੀ * spv_ਅੰਕੜੇ;
            size_t spv_ਮਾਪ;
            ਅਚਲ ਅੱਖਰ * name;
            ਜੇ (bf16_kv) {
#ਜੇ defined(VK_KHR_shader_bfloat16) && defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
                ਜੇ (!device->coopmat2_bf16_support) ਜਾਰੀ;
                spv_ਅੰਕੜੇ = flash_attn_f32_f16_bf16_cm2_ਅੰਕੜੇ;
                spv_ਮਾਪ = flash_attn_f32_f16_bf16_cm2_len;
                name = aligned ? "flash_attn_f32_bf16_aligned_cm2" : "flash_attn_f32_bf16_cm2";
#ਨਹੀਂਤਾਂ
                ਜਾਰੀ;
#ਜੇ_ਅੰਤ
            } ਨਹੀਂਤਾਂ ਜੇ (aligned) {
                ਜੇ (f32acc) { spv_ਅੰਕੜੇ = flash_attn_f32_f16_cm2_ਅੰਕੜੇ;        spv_ਮਾਪ = flash_attn_f32_f16_cm2_len;        name = "flash_attn_f32_f16_aligned_f32acc_cm2"; }
                ਨਹੀਂਤਾਂ        { spv_ਅੰਕੜੇ = flash_attn_f32_f16_f16acc_cm2_ਅੰਕੜੇ; spv_ਮਾਪ = flash_attn_f32_f16_f16acc_cm2_len; name = "flash_attn_f32_f16_aligned_f16acc_cm2"; }
            } ਨਹੀਂਤਾਂ {
                ਜੇ (f32acc) { spv_ਅੰਕੜੇ = flash_attn_f32_f16_cm2_ਅੰਕੜੇ;        spv_ਮਾਪ = flash_attn_f32_f16_cm2_len;        name = "flash_attn_f32_f16_f32acc_cm2"; }
                ਨਹੀਂਤਾਂ        { spv_ਅੰਕੜੇ = flash_attn_f32_f16_f16acc_cm2_ਅੰਕੜੇ; spv_ਮਾਪ = flash_attn_f32_f16_f16acc_cm2_len; name = "flash_attn_f32_f16_f16acc_cm2"; }
            }
            ggml_vk_create_pipeline(device, fa.second, name, spv_ਮਾਪ, spv_ਅੰਕੜੇ, "main", 8,
                                    ਆਕਾਰ(vk_flash_attn_ਧੱਕੋ_constants), {Br, 1, 1},
                                    get_fa_spec_constants(fa.ਪਹਿਲਾ), aligned ? Bc : 1, ਸੱਚ, ਝੂਠ, 0);
        }
    }
#ਜੇ_ਅੰਤ

    ਸਵੈਚਾਲਿਤ ਅਚਲ &ggml_vk_mul_mm_spec = [&device](std::vector<uint32_t> spec, ਤਾਰਕਿਕ aligned) {
        spec.push_back(aligned ? 1u : 0u);  // constantID=11: ALIGNED
        ਜੇ (device->vendor_id == VK_VENDOR_ID_INTEL && device->coopmat_support &&
            device->driver_id == vk::DriverId::eIntelProprietaryWindows) {
            spec.push_back(0u);  // constantID=12: SHMEM_STRIDE_PAD = 0
            spec.push_back(1u);  // constantID=13: APPLY_SLM_A_RESHAPE = true
        }
        ਵਾਪਸ spec;
    };

    ਸਵੈਚਾਲਿਤ ਅਚਲ &ggml_vk_mul_mm_spec_quant = [&device](std::vector<uint32_t> spec, ਤਾਰਕਿਕ aligned, uint32_t type) {
        spec.push_back(aligned ? 1u : 0u);  // constantID=11: ALIGNED
        spec.push_back(type);               // constantID=12: MmTypeA
        ਜੇ (device->vendor_id == VK_VENDOR_ID_INTEL && device->coopmat_support &&
            device->driver_id == vk::DriverId::eIntelProprietaryWindows) {
            spec.push_back(0u);  // constantID=13: SHMEM_STRIDE_PAD = 0
            spec.push_back(1u);  // constantID=14: APPLY_SLM_A_RESHAPE = true
        }
        ਵਾਪਸ spec;
    };

    ਸਥਿਰ ਅਚਲ ggml_type non_lut_quant_types[] = {
        GGML_TYPE_Q1_0, GGML_TYPE_Q2_0, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0,
        GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0,
    };

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ FOR_EACH_LUT_TYPE_NONFP4(X) \
    X(GGML_TYPE_IQ1_S,   iq1_s)   \
    X(GGML_TYPE_IQ1_M,   iq1_m)   \
    X(GGML_TYPE_IQ2_XXS, iq2_xxs) \
    X(GGML_TYPE_IQ2_XS,  iq2_xs)  \
    X(GGML_TYPE_IQ2_S,   iq2_s)   \
    X(GGML_TYPE_IQ3_XXS, iq3_xxs) \
    X(GGML_TYPE_IQ3_S,   iq3_s)   \
    X(GGML_TYPE_IQ4_XS,  iq4_xs)  \
    X(GGML_TYPE_IQ4_NL,  iq4_nl)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ FOR_EACH_LUT_FP4_TYPE(X) \
    X(GGML_TYPE_MXFP4,   mxfp4)   \
    X(GGML_TYPE_NVFP4,   nvfp4)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ FOR_EACH_LUT_TYPE(X) \
    FOR_EACH_LUT_TYPE_NONFP4(X)  \
    FOR_EACH_LUT_FP4_TYPE(X)

    ਅਚਲ ਪੂਰਨ mul_mat_id_param_ਗਿਣਤੀ = 5;

    ਵਰਤੋਂ spec_fn_t = std::function<std::vector<uint32_t>(ਅਚਲ std::vector<uint32_t>&, ਤਾਰਕਿਕ)>;
    ਸਵੈਚਾਲਿਤ ਅਚਲ &create_mm_pipelines = [&](
        ਅਚਲ vk_matmul_pipeline_ਕੁੰਜੀ& ਕੁੰਜੀ,
        ਅਚਲ std::vector<vk_tile_config>& tile_configs,
        ਅਚਲ std::string& shader_name, size_t spv_len, ਅਚਲ ਖਾਲੀ* spv_ਅੰਕੜੇ,
        uint32_t ਧੱਕੋ_constant_ਮਾਪ, uint32_t param_ਗਿਣਤੀ,
        ਅਚਲ spec_fn_t& spec_fn,
        ਤਾਰਕਿਕ disable_robustness = ਝੂਠ, ਤਾਰਕਿਕ require_ਭਰਿਆ_subgroups = ਝੂਠ, uint32_t required_subgroup_ਮਾਪ = 0,
        ਤਾਰਕਿਕ create_aligned = ਸੱਚ, ਤਾਰਕਿਕ pin_subgroup_ਮੰਜ਼ਿਲ_warp = ਝੂਠ
    ) {
        ਸਵੈਚਾਲਿਤ& vec = device->pipeline_matmul[ਕੁੰਜੀ];
        ਅਚਲ ਤਾਰਕਿਕ ਪਹਿਲਾ_call = vec.ਖਾਲੀ_ਥਾਵਾਂ();
        ਲਈ (size_t i = 0; i < tile_configs.size(); i++) {
            ਅਚਲ ਸਵੈਚਾਲਿਤ& tc = tile_configs[i];

            // Intel coopmat1 pins the required subgroup size to each warptile's WARP element.
            ਅਚਲ uint32_t rsgs = pin_subgroup_ਮੰਜ਼ਿਲ_warp ? tc.warptile[WARP_ਮਾਪ_IDX] : required_subgroup_ਮਾਪ;
            ਅਚਲ ਤਾਰਕਿਕ     rfs  = require_ਭਰਿਆ_subgroups || pin_subgroup_ਮੰਜ਼ਿਲ_warp;

            ਜੇ (ਪਹਿਲਾ_call) {
                vk_matmul_pipeline_pair pair{};
                pair.align = tc.align;
                std::string suffix = "_" + std::to_string(i);
                pair.unaligned = std::make_shared<vk_pipeline_struct>();
                ਜੇ (create_aligned) {
                    pair.aligned = std::make_shared<vk_pipeline_struct>();
                }
                vec.push_back(pair);
            }

            ggml_vk_create_pipeline(device, vec[i].unaligned,
                vec[i].unaligned->name.ਖਾਲੀ_ਥਾਵਾਂ() ? (shader_name + "_" + std::to_string(i)).c_str() : vec[i].unaligned->name.c_str(),
                spv_len, spv_ਅੰਕੜੇ, "main", param_ਗਿਣਤੀ, ਧੱਕੋ_constant_ਮਾਪ,
                tc.wg_denoms, spec_fn(tc.warptile, ਝੂਠ), 1,
                disable_robustness, rfs, rsgs);

            ਜੇ (vec[i].aligned) {
                ggml_vk_create_pipeline(device, vec[i].aligned,
                    vec[i].aligned->name.ਖਾਲੀ_ਥਾਵਾਂ() ? (shader_name + "_aligned_" + std::to_string(i)).c_str() : vec[i].aligned->name.c_str(),
                    spv_len, spv_ਅੰਕੜੇ, "main", param_ਗਿਣਤੀ, ਧੱਕੋ_constant_ਮਾਪ,
                    tc.wg_denoms, spec_fn(tc.warptile, ਸੱਚ), tc.align,
                    disable_robustness, rfs, rsgs);
            }
        }
    };

    ਸਵੈਚਾਲਿਤ filter_tc = [&](ਅਚਲ std::vector<vk_tile_config>& configs, ggml_type type, ਤਾਰਕਿਕ is_id, ਤਾਰਕਿਕ is_int = ਝੂਠ) -> std::vector<vk_tile_config> {
        std::vector<vk_tile_config> ਨਤੀਜਾ;
        ਤਾਰਕਿਕ enabled[3];
        ਜੇ (is_int) {
            enabled[0] = is_id ? device->mul_mat_id_s_int[type] : device->mul_mat_s_int[type];
            enabled[1] = is_id ? device->mul_mat_id_m_int[type] : device->mul_mat_m_int[type];
            enabled[2] = is_id ? device->mul_mat_id_l_int[type] : device->mul_mat_l_int[type];
        } ਨਹੀਂਤਾਂ {
            enabled[0] = is_id ? device->mul_mat_id_s[type] : device->mul_mat_s[type];
            enabled[1] = is_id ? device->mul_mat_id_m[type] : device->mul_mat_m[type];
            enabled[2] = is_id ? device->mul_mat_id_l[type] : device->mul_mat_l[type];
        }
        ਲਈ (size_t i = 0; i < configs.size() && i < 3; i++) {
            ਜੇ (enabled[i]) ਨਤੀਜਾ.push_back(configs[i]);
        }
        ਵਾਪਸ ਨਤੀਜਾ;
    };

    std::vector<vk_tile_config> tc_mm = {{s_warptile, s_wg_denoms, s_align}, {m_warptile, m_wg_denoms, m_align}, {l_warptile, l_wg_denoms, l_align}};
    std::vector<vk_tile_config> tc_mmq = {{s_warptile_mmq, s_mmq_wg_denoms, s_align}, {m_warptile_mmq, m_mmq_wg_denoms, m_align}, {l_warptile_mmq, l_mmq_wg_denoms, l_align}};

#ਜੇ defined(VK_NV_cooperative_matrix2) && defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
    ਜੇ (device->coopmat2) {
        ਸਵੈਚਾਲਿਤ ਅਚਲ &ggml_vk_mul_mm_cm2_spec = [&](std::vector<uint32_t> spec, ਤਾਰਕਿਕ aligned, uint32_t type = UINT32_MAX) {
            spec.push_back(aligned ? 1u : 0u);        // ALIGNED
            spec.push_back(device->subgroup_ਮਾਪ);     // subgroup_size
            ਜੇ (type != UINT32_MAX) {
                spec.push_back(type);                  // MmTypeA
                spec.push_back((uint32_t)ggml_type_ਮਾਪ((ggml_type)type)); // MmABlockBytes
            }
            ਵਾਪਸ spec;
        };

        std::vector<vk_tile_config> tc_mmq_k = {{s_warptile_mmq_k, s_mmq_wg_denoms_k, s_align}, {m_warptile_mmq_k, m_mmq_wg_denoms_k, m_align}, {l_warptile_mmq_k, l_mmq_wg_denoms_k, l_align}};
        std::vector<vk_tile_config> tc_mmqid = {{s_warptile_mmqid, s_mmqid_wg_denoms, s_align}, {m_warptile_mmqid, m_mmqid_wg_denoms, m_align}, {l_warptile_mmqid, l_mmqid_wg_denoms, l_align}};

        spec_fn_t cm2_spec = [&](ਅਚਲ std::vector<uint32_t>& wt, ਤਾਰਕਿਕ a) { ਵਾਪਸ ggml_vk_mul_mm_cm2_spec(wt, a); };

        // F16 x F16
        create_mm_pipelines({GGML_TYPE_F16, GGML_TYPE_F16, ਝੂਠ, ਸੱਚ},  tc_mm, "matmul_f16_f16acc", matmul_f16_f16acc_cm2_len, matmul_f16_f16acc_cm2_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3, cm2_spec, ਸੱਚ);
        create_mm_pipelines({GGML_TYPE_F16, GGML_TYPE_F16, ਝੂਠ, ਝੂਠ}, tc_mm, "matmul_f16",        matmul_f16_cm2_len,        matmul_f16_cm2_ਅੰਕੜੇ,        ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3, cm2_spec, ਸੱਚ);
#ਜੇ defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        ਜੇ (device->coopmat_bf16_support) {
            create_mm_pipelines({GGML_TYPE_BF16, GGML_TYPE_BF16, ਝੂਠ, ਝੂਠ}, tc_mm, "matmul_bf16", matmul_bf16_cm2_len, matmul_bf16_cm2_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3, cm2_spec, ਸੱਚ);
        }
#ਜੇ_ਅੰਤ
        ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ type : non_lut_quant_types) {
            // regression in unified shader on Ampere
            ਜੇ (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
                ਜਾਰੀ;
            }
            ਸਵੈਚਾਲਿਤ& tc = ((type >= GGML_TYPE_Q2_K && type <= GGML_TYPE_Q6_K) || type == GGML_TYPE_TQ1_0 || type == GGML_TYPE_TQ2_0) ? tc_mmq_k : tc_mmq;
            spec_fn_t qs = [&, type](ਅਚਲ std::vector<uint32_t>& wt, ਤਾਰਕਿਕ a) { ਵਾਪਸ ggml_vk_mul_mm_cm2_spec(wt, a, (uint32_t)type); };
            create_mm_pipelines({type, GGML_TYPE_F16, ਝੂਠ, ਸੱਚ},  tc, "matmul_quant_f16_f16acc", matmul_quant_f16_f16acc_cm2_len, matmul_quant_f16_f16acc_cm2_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3, qs, ਸੱਚ);
            create_mm_pipelines({type, GGML_TYPE_F16, ਝੂਠ, ਝੂਠ}, tc, "matmul_quant_f16",        matmul_quant_f16_cm2_len,        matmul_quant_f16_cm2_ਅੰਕੜੇ,        ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3, qs, ਸੱਚ);
        }
        create_mm_pipelines({GGML_TYPE_Q4_K, GGML_TYPE_F16, ਝੂਠ, ਸੱਚ},  tc_mmq_k, "matmul_q4_k_f16_f16acc", matmul_q4_k_f16_f16acc_cm2_len, matmul_q4_k_f16_f16acc_cm2_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3, cm2_spec, ਸੱਚ);
        create_mm_pipelines({GGML_TYPE_Q4_K, GGML_TYPE_F16, ਝੂਠ, ਝੂਠ}, tc_mmq_k, "matmul_q4_k_f16",        matmul_q4_k_f16_cm2_len,        matmul_q4_k_f16_cm2_ਅੰਕੜੇ,        ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3, cm2_spec, ਸੱਚ);
        create_mm_pipelines({GGML_TYPE_Q5_K, GGML_TYPE_F16, ਝੂਠ, ਸੱਚ},  tc_mmq_k, "matmul_q5_k_f16_f16acc", matmul_q5_k_f16_f16acc_cm2_len, matmul_q5_k_f16_f16acc_cm2_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3, cm2_spec, ਸੱਚ);
        create_mm_pipelines({GGML_TYPE_Q5_K, GGML_TYPE_F16, ਝੂਠ, ਝੂਠ}, tc_mmq_k, "matmul_q5_k_f16",        matmul_q5_k_f16_cm2_len,        matmul_q5_k_f16_cm2_ਅੰਕੜੇ,        ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3, cm2_spec, ਸੱਚ);
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ X_CM2(TYPE, tstr) \
        { ਸਵੈਚਾਲਿਤ tc = filter_tc(tc_mmq, TYPE, ਝੂਠ); \
          ਜੇ (!tc.ਖਾਲੀ_ਥਾਵਾਂ()) { \
              create_mm_pipelines({TYPE, GGML_TYPE_F16, ਝੂਠ, ਸੱਚ},  tc, "matmul_" #tstr "_f16_f16acc", matmul_##tstr##_f16_f16acc_cm2_len, matmul_##tstr##_f16_f16acc_cm2_data, ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3, cm2_spec, ਸੱਚ); \
              create_mm_pipelines({TYPE, GGML_TYPE_F16, ਝੂਠ, ਝੂਠ}, tc, "matmul_" #tstr "_f16",        matmul_##tstr##_f16_cm2_len,        matmul_##tstr##_f16_cm2_data,        ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3, cm2_spec, ਸੱਚ); \
          } }
        FOR_EACH_LUT_TYPE_NONFP4(X_CM2)
#ਜੇ defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
        ਜੇ (device->ocp_fp4) {
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ X_CM2_OCP(TYPE, tstr) \
            { ਸਵੈਚਾਲਿਤ tc = filter_tc(tc_mmq, TYPE, ਝੂਠ); \
              ਜੇ (!tc.ਖਾਲੀ_ਥਾਵਾਂ()) { \
                  create_mm_pipelines({TYPE, GGML_TYPE_F16, ਝੂਠ, ਸੱਚ},  tc, "matmul_" #tstr "_f16_ocp_f16acc", matmul_##tstr##_f16_ocp_f16acc_cm2_len, matmul_##tstr##_f16_ocp_f16acc_cm2_data, ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3, cm2_spec, ਸੱਚ); \
                  create_mm_pipelines({TYPE, GGML_TYPE_F16, ਝੂਠ, ਝੂਠ}, tc, "matmul_" #tstr "_f16_ocp",        matmul_##tstr##_f16_ocp_cm2_len,        matmul_##tstr##_f16_ocp_cm2_data,        ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3, cm2_spec, ਸੱਚ); \
              } }
            FOR_EACH_LUT_FP4_TYPE(X_CM2_OCP)
#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ X_CM2_OCP
        } ਨਹੀਂਤਾਂ
#ਜੇ_ਅੰਤ
        {
            FOR_EACH_LUT_FP4_TYPE(X_CM2)
        }
#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ X_CM2

        GGML_ASSERT(device->subgroup_ballot);

        create_mm_pipelines({GGML_TYPE_F16, GGML_TYPE_F16, ਸੱਚ, ਸੱਚ},  tc_mm, "matmul_id_subgroup_f16_f16acc", matmul_id_subgroup_f16_f16acc_cm2_len, matmul_id_subgroup_f16_f16acc_cm2_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), 5, cm2_spec, ਸੱਚ);
        create_mm_pipelines({GGML_TYPE_F16, GGML_TYPE_F16, ਸੱਚ, ਝੂਠ}, tc_mm, "matmul_id_subgroup_f16",        matmul_id_subgroup_f16_cm2_len,        matmul_id_subgroup_f16_cm2_ਅੰਕੜੇ,        ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), 5, cm2_spec, ਸੱਚ);
#ਜੇ defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        ਜੇ (device->coopmat_bf16_support) {
            create_mm_pipelines({GGML_TYPE_BF16, GGML_TYPE_BF16, ਸੱਚ, ਝੂਠ}, tc_mm, "matmul_id_subgroup_bf16", matmul_id_subgroup_bf16_cm2_len, matmul_id_subgroup_bf16_cm2_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), 5, cm2_spec, ਸੱਚ);
        }
#ਜੇ_ਅੰਤ
        ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ type : non_lut_quant_types) {
            ਜੇ (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
                ਜਾਰੀ;
            }
            spec_fn_t qs_id = [&, type](ਅਚਲ std::vector<uint32_t>& wt, ਤਾਰਕਿਕ a) { ਵਾਪਸ ggml_vk_mul_mm_cm2_spec(wt, a, (uint32_t)type); };
            create_mm_pipelines({type, GGML_TYPE_F16, ਸੱਚ, ਸੱਚ},  tc_mmqid, "matmul_id_subgroup_quant_f16_f16acc", matmul_id_subgroup_quant_f16_f16acc_cm2_len, matmul_id_subgroup_quant_f16_f16acc_cm2_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), 5, qs_id, ਸੱਚ);
            create_mm_pipelines({type, GGML_TYPE_F16, ਸੱਚ, ਝੂਠ}, tc_mmqid, "matmul_id_subgroup_quant_f16",        matmul_id_subgroup_quant_f16_cm2_len,        matmul_id_subgroup_quant_f16_cm2_ਅੰਕੜੇ,        ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), 5, qs_id, ਸੱਚ);
        }
        create_mm_pipelines({GGML_TYPE_Q4_K, GGML_TYPE_F16, ਸੱਚ, ਸੱਚ},  tc_mmqid, "matmul_id_subgroup_q4_k_f16_f16acc", matmul_id_subgroup_q4_k_f16_f16acc_cm2_len, matmul_id_subgroup_q4_k_f16_f16acc_cm2_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), 5, cm2_spec, ਸੱਚ);
        create_mm_pipelines({GGML_TYPE_Q4_K, GGML_TYPE_F16, ਸੱਚ, ਝੂਠ}, tc_mmqid, "matmul_id_subgroup_q4_k_f16",        matmul_id_subgroup_q4_k_f16_cm2_len,        matmul_id_subgroup_q4_k_f16_cm2_ਅੰਕੜੇ,        ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), 5, cm2_spec, ਸੱਚ);
        create_mm_pipelines({GGML_TYPE_Q5_K, GGML_TYPE_F16, ਸੱਚ, ਸੱਚ},  tc_mmqid, "matmul_id_subgroup_q5_k_f16_f16acc", matmul_id_subgroup_q5_k_f16_f16acc_cm2_len, matmul_id_subgroup_q5_k_f16_f16acc_cm2_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), 5, cm2_spec, ਸੱਚ);
        create_mm_pipelines({GGML_TYPE_Q5_K, GGML_TYPE_F16, ਸੱਚ, ਝੂਠ}, tc_mmqid, "matmul_id_subgroup_q5_k_f16",        matmul_id_subgroup_q5_k_f16_cm2_len,        matmul_id_subgroup_q5_k_f16_cm2_ਅੰਕੜੇ,        ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), 5, cm2_spec, ਸੱਚ);
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ X_CM2_ID(TYPE, tstr) \
        { ਸਵੈਚਾਲਿਤ tc = filter_tc(tc_mmqid, TYPE, ਸੱਚ); \
          ਜੇ (!tc.ਖਾਲੀ_ਥਾਵਾਂ()) { \
              create_mm_pipelines({TYPE, GGML_TYPE_F16, ਸੱਚ, ਸੱਚ},  tc, "matmul_id_subgroup_" #tstr "_f16_f16acc", matmul_id_subgroup_##tstr##_f16_f16acc_cm2_len, matmul_id_subgroup_##tstr##_f16_f16acc_cm2_data, ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), 5, cm2_spec, ਸੱਚ); \
              create_mm_pipelines({TYPE, GGML_TYPE_F16, ਸੱਚ, ਝੂਠ}, tc, "matmul_id_subgroup_" #tstr "_f16",        matmul_id_subgroup_##tstr##_f16_cm2_len,        matmul_id_subgroup_##tstr##_f16_cm2_data,        ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), 5, cm2_spec, ਸੱਚ); \
          } }
        FOR_EACH_LUT_TYPE_NONFP4(X_CM2_ID)
#ਜੇ defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
        ਜੇ (device->ocp_fp4) {
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ X_CM2_ID_OCP(TYPE, tstr) \
            { ਸਵੈਚਾਲਿਤ tc = filter_tc(tc_mmqid, TYPE, ਸੱਚ); \
              ਜੇ (!tc.ਖਾਲੀ_ਥਾਵਾਂ()) { \
                  create_mm_pipelines({TYPE, GGML_TYPE_F16, ਸੱਚ, ਸੱਚ},  tc, "matmul_id_subgroup_" #tstr "_f16_ocp_f16acc", matmul_id_subgroup_##tstr##_f16_ocp_f16acc_cm2_len, matmul_id_subgroup_##tstr##_f16_ocp_f16acc_cm2_data, ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), 5, cm2_spec, ਸੱਚ); \
                  create_mm_pipelines({TYPE, GGML_TYPE_F16, ਸੱਚ, ਝੂਠ}, tc, "matmul_id_subgroup_" #tstr "_f16_ocp",        matmul_id_subgroup_##tstr##_f16_ocp_cm2_len,        matmul_id_subgroup_##tstr##_f16_ocp_cm2_data,        ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), 5, cm2_spec, ਸੱਚ); \
              } }
            FOR_EACH_LUT_FP4_TYPE(X_CM2_ID_OCP)
#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ X_CM2_ID_OCP
        } ਨਹੀਂਤਾਂ
#ਜੇ_ਅੰਤ
        {
            FOR_EACH_LUT_FP4_TYPE(X_CM2_ID)
        }
#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ X_CM2_ID
    } ਨਹੀਂਤਾਂ
#ਜੇ_ਅੰਤ  // defined(VK_NV_cooperative_matrix2) && defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
#ਜੇ defined(VK_KHR_cooperative_ਮੈਟ੍ਰਿਕਸ) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    ਜੇ (device->coopmat_support) {
        spec_fn_t cm1_spec = [&](ਅਚਲ std::vector<uint32_t>& wt, ਤਾਰਕਿਕ a) { ਵਾਪਸ ggml_vk_mul_mm_spec(wt, a); };

        // Intel coopmat1 pins each pipeline's required subgroup size to its warptile WARP element.
        ਅਚਲ ਤਾਰਕਿਕ cm1_pin = device->vendor_id == VK_VENDOR_ID_INTEL;

        // Intel coopmat1 uses a dedicated large-tile config for quant matmul_id.
        std::vector<vk_tile_config> tc_mmq_id = tc_mmq;
        ਜੇ (cm1_pin) {
            tc_mmq_id[2] = { { 512, 128, 128, 32, 32, 32, 2, device->coopmat_m, device->coopmat_n, device->coopmat_k, 32 }, { 128, 128, 1 }, 32 };
        }

        ਸਵੈਚਾਲਿਤ cm1_create = [&](vk_matmul_pipeline_ਕੁੰਜੀ ਕੁੰਜੀ, ਅਚਲ std::vector<vk_tile_config>& tc_ਅਧਾਰ,
                              ਅਚਲ std::string& name, size_t len, ਅਚਲ ਖਾਲੀ* ਅੰਕੜੇ, uint32_t pc_ਮਾਪ, uint32_t pc) {
            ਸਵੈਚਾਲਿਤ tc = filter_tc(tc_ਅਧਾਰ, ਕੁੰਜੀ.type_a, ਕੁੰਜੀ.mul_mat_id);
            ਜੇ (!tc.ਖਾਲੀ_ਥਾਵਾਂ()) create_mm_pipelines(ਕੁੰਜੀ, tc, name, len, ਅੰਕੜੇ, pc_ਮਾਪ, pc, cm1_spec, ਝੂਠ, ਸੱਚ, 0, ਸੱਚ, cm1_pin);
        };
        ਸਵੈਚਾਲਿਤ cm1_create_quant = [&](vk_matmul_pipeline_ਕੁੰਜੀ ਕੁੰਜੀ, ਅਚਲ std::vector<vk_tile_config>& tc_ਅਧਾਰ,
                                    ਅਚਲ std::string& name, size_t len, ਅਚਲ ਖਾਲੀ* ਅੰਕੜੇ, uint32_t pc_ਮਾਪ, uint32_t pc) {
            spec_fn_t qs = [&, type_a=ਕੁੰਜੀ.type_a](ਅਚਲ std::vector<uint32_t>& wt, ਤਾਰਕਿਕ a) { ਵਾਪਸ ggml_vk_mul_mm_spec_quant(wt, a, (uint32_t)type_a); };
            ਸਵੈਚਾਲਿਤ tc = filter_tc(tc_ਅਧਾਰ, ਕੁੰਜੀ.type_a, ਕੁੰਜੀ.mul_mat_id);
            ਜੇ (!tc.ਖਾਲੀ_ਥਾਵਾਂ()) create_mm_pipelines(ਕੁੰਜੀ, tc, name, len, ਅੰਕੜੇ, pc_ਮਾਪ, pc, qs, ਝੂਠ, ਸੱਚ, 0, ਸੱਚ, cm1_pin);
        };
        // int8 MMQ helper: per-type cm1 shader, warptile passed as-is (carries DEVICE_ARCH in
        // spec constant WARP_SIZE_IDX+1), subgroup size pinned to the warptile WARP element, no aligned variant.
        ਸਵੈਚਾਲਿਤ cm1_create_mmq = [&](vk_matmul_pipeline_ਕੁੰਜੀ ਕੁੰਜੀ, ਅਚਲ std::vector<vk_tile_config>& tc_ਅਧਾਰ,
                                  ਅਚਲ std::string& name, size_t len, ਅਚਲ ਖਾਲੀ* ਅੰਕੜੇ, uint32_t pc_ਮਾਪ, uint32_t pc) {
            spec_fn_t ਪਛਾਣ_ਤੱਤ = [](ਅਚਲ std::vector<uint32_t>& wt, ਤਾਰਕਿਕ) { ਵਾਪਸ wt; };
            ਸਵੈਚਾਲਿਤ tc = filter_tc(tc_ਅਧਾਰ, ਕੁੰਜੀ.type_a, ਕੁੰਜੀ.mul_mat_id, ਸੱਚ);
            ਜੇ (!tc.ਖਾਲੀ_ਥਾਵਾਂ()) create_mm_pipelines(ਕੁੰਜੀ, tc, name, len, ਅੰਕੜੇ, pc_ਮਾਪ, pc, ਪਛਾਣ_ਤੱਤ, ਝੂਠ, ਝੂਠ, 0, ਝੂਠ, ਸੱਚ);
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
        ਅਚਲ ਤਾਰਕਿਕ rdna3 = device->architecture == AMD_RDNA3;
        ਅਚਲ ਤਾਰਕਿਕ rdna4 = device->architecture == AMD_RDNA4;

        cm1_create({GGML_TYPE_F32, GGML_TYPE_F32, ਝੂਠ, ਝੂਠ}, tc_mm, "matmul_f32_f32",     matmul_f32_f32_cm1_len,     matmul_f32_f32_cm1_ਅੰਕੜੇ,     ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3);
        cm1_create({GGML_TYPE_F32, GGML_TYPE_F16, ਝੂਠ, ਝੂਠ}, tc_mm, "matmul_f32_f16",     matmul_f32_f16_cm1_len,     matmul_f32_f16_cm1_ਅੰਕੜੇ,     ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3);
        ਜੇ (device->coopmat_acc_f16_support) {
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F16, ਝੂਠ, ਸੱਚ},  tc_mm, "matmul_f16_f16acc", matmul_f16_f16acc_cm1_len, matmul_f16_f16acc_cm1_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3);
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F32, ਝੂਠ, ਸੱਚ},  tc_mm, "matmul_f16_f32_f16acc", matmul_f16_f32_f16acc_cm1_len, matmul_f16_f32_f16acc_cm1_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3);
        }
        ਜੇ (device->coopmat_acc_f32_support) {
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F16, ਝੂਠ, ਝੂਠ}, tc_mm, "matmul_f16",      matmul_f16_cm1_len,      matmul_f16_cm1_ਅੰਕੜੇ,      ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3);
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F32, ਝੂਠ, ਝੂਠ}, tc_mm, "matmul_f16_f32",  matmul_f16_f32_cm1_len,  matmul_f16_f32_cm1_ਅੰਕੜੇ,  ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3);
        }
#ਜੇ defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        ਜੇ (device->coopmat_bf16_support) {
            cm1_create({GGML_TYPE_BF16, GGML_TYPE_BF16, ਝੂਠ, ਝੂਠ}, tc_mm, "matmul_bf16", matmul_bf16_cm1_len, matmul_bf16_cm1_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3);
        }
#ਜੇ_ਅੰਤ
        ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ type : non_lut_quant_types) {
            ਜੇ (device->coopmat_acc_f16_support) {
                cm1_create_quant({type, GGML_TYPE_F32, ਝੂਠ, ਸੱਚ},  tc_mmq, "matmul_quant_f32_f16acc", matmul_quant_f32_f16acc_cm1_len, matmul_quant_f32_f16acc_cm1_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3);
                cm1_create_quant({type, GGML_TYPE_F16, ਝੂਠ, ਸੱਚ},  tc_mmq, "matmul_quant_f16_f16acc", matmul_quant_f16_f16acc_cm1_len, matmul_quant_f16_f16acc_cm1_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3);
            }
            ਜੇ (device->coopmat_acc_f32_support) {
                cm1_create_quant({type, GGML_TYPE_F32, ਝੂਠ, ਝੂਠ}, tc_mmq, "matmul_quant_f32",        matmul_quant_f32_cm1_len,        matmul_quant_f32_cm1_ਅੰਕੜੇ,        ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3);
                cm1_create_quant({type, GGML_TYPE_F16, ਝੂਠ, ਝੂਠ}, tc_mmq, "matmul_quant_f16",        matmul_quant_f16_cm1_len,        matmul_quant_f16_cm1_ਅੰਕੜੇ,        ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3);
            }
        }
        // The _f16 variants provide the f16 B-type pipeline used when y_non_contig converts f32->f16.
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ X_CM1(TYPE, tstr) \
        ਜੇ (device->coopmat_acc_f16_support) { \
            cm1_create({TYPE, GGML_TYPE_F32, ਝੂਠ, ਸੱਚ},  tc_mmq, "matmul_" #tstr "_f32_f16acc", matmul_##tstr##_f32_f16acc_cm1_len, matmul_##tstr##_f32_f16acc_cm1_data, ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3); \
            cm1_create({TYPE, GGML_TYPE_F16, ਝੂਠ, ਸੱਚ},  tc_mmq, "matmul_" #tstr "_f16_f16acc", matmul_##tstr##_f16_f16acc_cm1_len, matmul_##tstr##_f16_f16acc_cm1_data, ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3); \
        } \
        ਜੇ (device->coopmat_acc_f32_support) { \
            cm1_create({TYPE, GGML_TYPE_F32, ਝੂਠ, ਝੂਠ}, tc_mmq, "matmul_" #tstr "_f32",        matmul_##tstr##_f32_cm1_len,        matmul_##tstr##_f32_cm1_data,        ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3); \
            cm1_create({TYPE, GGML_TYPE_F16, ਝੂਠ, ਝੂਠ}, tc_mmq, "matmul_" #tstr "_f16",        matmul_##tstr##_f16_cm1_len,        matmul_##tstr##_f16_cm1_data,        ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3); \
        }
        FOR_EACH_LUT_TYPE_NONFP4(X_CM1)
#ਜੇ defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
        ਜੇ (device->ocp_fp4) {
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ X_CM1_OCP(TYPE, tstr) \
            ਜੇ (device->coopmat_acc_f16_support) { \
                cm1_create({TYPE, GGML_TYPE_F32, ਝੂਠ, ਸੱਚ},  tc_mmq, "matmul_" #tstr "_f32_ocp_f16acc", matmul_##tstr##_f32_ocp_f16acc_cm1_len, matmul_##tstr##_f32_ocp_f16acc_cm1_data, ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3); \
                cm1_create({TYPE, GGML_TYPE_F16, ਝੂਠ, ਸੱਚ},  tc_mmq, "matmul_" #tstr "_f16_ocp_f16acc", matmul_##tstr##_f16_ocp_f16acc_cm1_len, matmul_##tstr##_f16_ocp_f16acc_cm1_data, ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3); \
            } \
            ਜੇ (device->coopmat_acc_f32_support) { \
                cm1_create({TYPE, GGML_TYPE_F32, ਝੂਠ, ਝੂਠ}, tc_mmq, "matmul_" #tstr "_f32_ocp",        matmul_##tstr##_f32_ocp_cm1_len,        matmul_##tstr##_f32_ocp_cm1_data,        ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3); \
                cm1_create({TYPE, GGML_TYPE_F16, ਝੂਠ, ਝੂਠ}, tc_mmq, "matmul_" #tstr "_f16_ocp",        matmul_##tstr##_f16_ocp_cm1_len,        matmul_##tstr##_f16_ocp_cm1_data,        ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3); \
            }
            FOR_EACH_LUT_FP4_TYPE(X_CM1_OCP)
#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ X_CM1_OCP
        } ਨਹੀਂਤਾਂ
#ਜੇ_ਅੰਤ
        {
            FOR_EACH_LUT_FP4_TYPE(X_CM1)
        }
#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ X_CM1

        ਜੇ (device->coopmat_int_support && (rdna3 || rdna4)) {
            cm1_create_mmq({GGML_TYPE_Q4_0,   GGML_TYPE_Q8_1, ਝੂਠ, ਝੂਠ}, tc_mmq_cm1_int,   "matmul_q4_0_q8_1",   matmul_q4_0_q8_1_cm1_len,   matmul_q4_0_q8_1_cm1_ਅੰਕੜੇ,   ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3);
            ਜੇ (!rdna4) { cm1_create_mmq({GGML_TYPE_Q4_1, GGML_TYPE_Q8_1, ਝੂਠ, ਝੂਠ}, tc_mmq_cm1_int,   "matmul_q4_1_q8_1",   matmul_q4_1_q8_1_cm1_len,   matmul_q4_1_q8_1_cm1_ਅੰਕੜੇ,   ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3); }
            cm1_create_mmq({GGML_TYPE_Q5_0,   GGML_TYPE_Q8_1, ਝੂਠ, ਝੂਠ}, tc_mmq_cm1_int,   "matmul_q5_0_q8_1",   matmul_q5_0_q8_1_cm1_len,   matmul_q5_0_q8_1_cm1_ਅੰਕੜੇ,   ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3);
            ਜੇ (!rdna4) { cm1_create_mmq({GGML_TYPE_Q5_1, GGML_TYPE_Q8_1, ਝੂਠ, ਝੂਠ}, tc_mmq_cm1_int,   "matmul_q5_1_q8_1",   matmul_q5_1_q8_1_cm1_len,   matmul_q5_1_q8_1_cm1_ਅੰਕੜੇ,   ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3); }
            cm1_create_mmq({GGML_TYPE_Q8_0,   GGML_TYPE_Q8_1, ਝੂਠ, ਝੂਠ}, tc_mmq_cm1_int,   "matmul_q8_0_q8_1",   matmul_q8_0_q8_1_cm1_len,   matmul_q8_0_q8_1_cm1_ਅੰਕੜੇ,   ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3);
            cm1_create_mmq({GGML_TYPE_IQ4_NL, GGML_TYPE_Q8_1, ਝੂਠ, ਝੂਠ}, tc_mmq_cm1_int,   "matmul_iq4_nl_q8_1", matmul_iq4_nl_q8_1_cm1_len, matmul_iq4_nl_q8_1_cm1_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3);
            cm1_create_mmq({GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_1, ਝੂਠ, ਝੂਠ}, tc_mmq_cm1_int,   "matmul_iq4_xs_q8_1", matmul_iq4_xs_q8_1_cm1_len, matmul_iq4_xs_q8_1_cm1_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3);
            cm1_create_mmq({GGML_TYPE_MXFP4,  GGML_TYPE_Q8_1, ਝੂਠ, ਝੂਠ}, tc_mmq_cm1_int,   "matmul_mxfp4_q8_1",  matmul_mxfp4_q8_1_cm1_len,  matmul_mxfp4_q8_1_cm1_ਅੰਕੜੇ,  ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3);
            cm1_create_mmq({GGML_TYPE_Q3_K,   GGML_TYPE_Q8_1, ਝੂਠ, ਝੂਠ}, tc_mmq_cm1_int_k, "matmul_q3_k_q8_1",   matmul_q3_k_q8_1_cm1_len,   matmul_q3_k_q8_1_cm1_ਅੰਕੜੇ,   ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3);
            ਜੇ (!rdna4) { cm1_create_mmq({GGML_TYPE_Q4_K, GGML_TYPE_Q8_1, ਝੂਠ, ਝੂਠ}, tc_mmq_cm1_int,   "matmul_q4_k_q8_1",   matmul_q4_k_q8_1_cm1_len,   matmul_q4_k_q8_1_cm1_ਅੰਕੜੇ,   ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3); }
            ਜੇ (!rdna4) { cm1_create_mmq({GGML_TYPE_Q5_K, GGML_TYPE_Q8_1, ਝੂਠ, ਝੂਠ}, tc_mmq_cm1_int,   "matmul_q5_k_q8_1",   matmul_q5_k_q8_1_cm1_len,   matmul_q5_k_q8_1_cm1_ਅੰਕੜੇ,   ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3); }
            cm1_create_mmq({GGML_TYPE_Q6_K,   GGML_TYPE_Q8_1, ਝੂਠ, ਝੂਠ}, tc_mmq_cm1_int_k, "matmul_q6_k_q8_1",   matmul_q6_k_q8_1_cm1_len,   matmul_q6_k_q8_1_cm1_ਅੰਕੜੇ,   ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3);
            ਜੇ (!rdna4) { cm1_create_mmq({GGML_TYPE_NVFP4, GGML_TYPE_Q8_1, ਝੂਠ, ਝੂਠ}, tc_mmq_cm1_int_k, "matmul_nvfp4_q8_1",  matmul_nvfp4_q8_1_cm1_len,  matmul_nvfp4_q8_1_cm1_ਅੰਕੜੇ,  ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3); }
        }

        GGML_ASSERT(device->subgroup_ballot);

        cm1_create({GGML_TYPE_F32, GGML_TYPE_F32, ਸੱਚ, ਝੂਠ}, tc_mm, "matmul_id_subgroup_f32_f32", matmul_id_subgroup_f32_f32_cm1_len, matmul_id_subgroup_f32_f32_cm1_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ);
        ਜੇ (device->coopmat_acc_f16_support) {
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F16, ਸੱਚ, ਸੱਚ},  tc_mm, "matmul_id_subgroup_f16_f16acc",     matmul_id_subgroup_f16_f16acc_cm1_len,     matmul_id_subgroup_f16_f16acc_cm1_ਅੰਕੜੇ,     ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ);
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F32, ਸੱਚ, ਸੱਚ},  tc_mm, "matmul_id_subgroup_f16_f32_f16acc", matmul_id_subgroup_f16_f32_f16acc_cm1_len, matmul_id_subgroup_f16_f32_f16acc_cm1_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ);
        }
        ਜੇ (device->coopmat_acc_f32_support) {
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F16, ਸੱਚ, ਝੂਠ}, tc_mm, "matmul_id_subgroup_f16",     matmul_id_subgroup_f16_cm1_len,     matmul_id_subgroup_f16_cm1_ਅੰਕੜੇ,     ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ);
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F32, ਸੱਚ, ਝੂਠ}, tc_mm, "matmul_id_subgroup_f16_f32", matmul_id_subgroup_f16_f32_cm1_len, matmul_id_subgroup_f16_f32_cm1_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ);
        }
#ਜੇ defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        ਜੇ (device->coopmat_bf16_support) {
            cm1_create({GGML_TYPE_BF16, GGML_TYPE_BF16, ਸੱਚ, ਝੂਠ}, tc_mm, "matmul_id_subgroup_bf16", matmul_id_subgroup_bf16_cm1_len, matmul_id_subgroup_bf16_cm1_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ);
        }
#ਜੇ_ਅੰਤ
        ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ type : non_lut_quant_types) {
            ਜੇ (device->coopmat_acc_f16_support) {
                cm1_create_quant({type, GGML_TYPE_F32, ਸੱਚ, ਸੱਚ},  tc_mmq_id, "matmul_id_subgroup_quant_f32_f16acc", matmul_id_subgroup_quant_f32_f16acc_cm1_len, matmul_id_subgroup_quant_f32_f16acc_cm1_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ);
                cm1_create_quant({type, GGML_TYPE_F16, ਸੱਚ, ਸੱਚ},  tc_mmq_id, "matmul_id_subgroup_quant_f16_f16acc", matmul_id_subgroup_quant_f16_f16acc_cm1_len, matmul_id_subgroup_quant_f16_f16acc_cm1_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ);
            }
            ਜੇ (device->coopmat_acc_f32_support) {
                cm1_create_quant({type, GGML_TYPE_F32, ਸੱਚ, ਝੂਠ}, tc_mmq_id, "matmul_id_subgroup_quant_f32",        matmul_id_subgroup_quant_f32_cm1_len,        matmul_id_subgroup_quant_f32_cm1_ਅੰਕੜੇ,        ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ);
                cm1_create_quant({type, GGML_TYPE_F16, ਸੱਚ, ਝੂਠ}, tc_mmq_id, "matmul_id_subgroup_quant_f16",        matmul_id_subgroup_quant_f16_cm1_len,        matmul_id_subgroup_quant_f16_cm1_ਅੰਕੜੇ,        ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ);
            }
        }
        // The _f16 variants provide the f16 B-type pipeline used when y_non_contig converts f32->f16.
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ X_CM1_ID(TYPE, tstr) \
        ਜੇ (device->coopmat_acc_f16_support) { \
            cm1_create({TYPE, GGML_TYPE_F32, ਸੱਚ, ਸੱਚ},  tc_mmq_id, "matmul_id_subgroup_" #tstr "_f32_f16acc", matmul_id_subgroup_##tstr##_f32_f16acc_cm1_len, matmul_id_subgroup_##tstr##_f32_f16acc_cm1_data, ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ); \
            cm1_create({TYPE, GGML_TYPE_F16, ਸੱਚ, ਸੱਚ},  tc_mmq_id, "matmul_id_subgroup_" #tstr "_f16_f16acc", matmul_id_subgroup_##tstr##_f16_f16acc_cm1_len, matmul_id_subgroup_##tstr##_f16_f16acc_cm1_data, ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ); \
        } \
        ਜੇ (device->coopmat_acc_f32_support) { \
            cm1_create({TYPE, GGML_TYPE_F32, ਸੱਚ, ਝੂਠ}, tc_mmq_id, "matmul_id_subgroup_" #tstr "_f32",        matmul_id_subgroup_##tstr##_f32_cm1_len,        matmul_id_subgroup_##tstr##_f32_cm1_data,        ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ); \
            cm1_create({TYPE, GGML_TYPE_F16, ਸੱਚ, ਝੂਠ}, tc_mmq_id, "matmul_id_subgroup_" #tstr "_f16",        matmul_id_subgroup_##tstr##_f16_cm1_len,        matmul_id_subgroup_##tstr##_f16_cm1_data,        ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ); \
        }
        FOR_EACH_LUT_TYPE_NONFP4(X_CM1_ID)
#ਜੇ defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
        ਜੇ (device->ocp_fp4) {
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ X_CM1_ID_OCP(TYPE, tstr) \
            ਜੇ (device->coopmat_acc_f16_support) { \
                cm1_create({TYPE, GGML_TYPE_F32, ਸੱਚ, ਸੱਚ},  tc_mmq_id, "matmul_id_subgroup_" #tstr "_f32_ocp_f16acc", matmul_id_subgroup_##tstr##_f32_ocp_f16acc_cm1_len, matmul_id_subgroup_##tstr##_f32_ocp_f16acc_cm1_data, ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ); \
                cm1_create({TYPE, GGML_TYPE_F16, ਸੱਚ, ਸੱਚ},  tc_mmq_id, "matmul_id_subgroup_" #tstr "_f16_ocp_f16acc", matmul_id_subgroup_##tstr##_f16_ocp_f16acc_cm1_len, matmul_id_subgroup_##tstr##_f16_ocp_f16acc_cm1_data, ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ); \
            } \
            ਜੇ (device->coopmat_acc_f32_support) { \
                cm1_create({TYPE, GGML_TYPE_F32, ਸੱਚ, ਝੂਠ}, tc_mmq_id, "matmul_id_subgroup_" #tstr "_f32_ocp",        matmul_id_subgroup_##tstr##_f32_ocp_cm1_len,        matmul_id_subgroup_##tstr##_f32_ocp_cm1_data,        ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ); \
                cm1_create({TYPE, GGML_TYPE_F16, ਸੱਚ, ਝੂਠ}, tc_mmq_id, "matmul_id_subgroup_" #tstr "_f16_ocp",        matmul_id_subgroup_##tstr##_f16_ocp_cm1_len,        matmul_id_subgroup_##tstr##_f16_ocp_cm1_data,        ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ); \
            }
            FOR_EACH_LUT_FP4_TYPE(X_CM1_ID_OCP)
#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ X_CM1_ID_OCP
        } ਨਹੀਂਤਾਂ
#ਜੇ_ਅੰਤ
        {
            FOR_EACH_LUT_FP4_TYPE(X_CM1_ID)
        }
#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ X_CM1_ID

        ਜੇ (device->coopmat_int_support && (rdna3 || rdna4)) {
            cm1_create_mmq({GGML_TYPE_Q4_0,   GGML_TYPE_Q8_1, ਸੱਚ, ਝੂਠ}, tc_mmq_cm1_int,   "matmul_id_subgroup_q4_0_q8_1",   matmul_id_subgroup_q4_0_q8_1_cm1_len,   matmul_id_subgroup_q4_0_q8_1_cm1_ਅੰਕੜੇ,   ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ);
            cm1_create_mmq({GGML_TYPE_Q4_1,   GGML_TYPE_Q8_1, ਸੱਚ, ਝੂਠ}, tc_mmq_cm1_int,   "matmul_id_subgroup_q4_1_q8_1",   matmul_id_subgroup_q4_1_q8_1_cm1_len,   matmul_id_subgroup_q4_1_q8_1_cm1_ਅੰਕੜੇ,   ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ);
            cm1_create_mmq({GGML_TYPE_Q5_0,   GGML_TYPE_Q8_1, ਸੱਚ, ਝੂਠ}, tc_mmq_cm1_int,   "matmul_id_subgroup_q5_0_q8_1",   matmul_id_subgroup_q5_0_q8_1_cm1_len,   matmul_id_subgroup_q5_0_q8_1_cm1_ਅੰਕੜੇ,   ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ);
            cm1_create_mmq({GGML_TYPE_Q5_1,   GGML_TYPE_Q8_1, ਸੱਚ, ਝੂਠ}, tc_mmq_cm1_int,   "matmul_id_subgroup_q5_1_q8_1",   matmul_id_subgroup_q5_1_q8_1_cm1_len,   matmul_id_subgroup_q5_1_q8_1_cm1_ਅੰਕੜੇ,   ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ);
            cm1_create_mmq({GGML_TYPE_Q8_0,   GGML_TYPE_Q8_1, ਸੱਚ, ਝੂਠ}, tc_mmq_cm1_int,   "matmul_id_subgroup_q8_0_q8_1",   matmul_id_subgroup_q8_0_q8_1_cm1_len,   matmul_id_subgroup_q8_0_q8_1_cm1_ਅੰਕੜੇ,   ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ);
            cm1_create_mmq({GGML_TYPE_IQ4_NL, GGML_TYPE_Q8_1, ਸੱਚ, ਝੂਠ}, tc_mmq_cm1_int,   "matmul_id_subgroup_iq4_nl_q8_1", matmul_id_subgroup_iq4_nl_q8_1_cm1_len, matmul_id_subgroup_iq4_nl_q8_1_cm1_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ);
            cm1_create_mmq({GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_1, ਸੱਚ, ਝੂਠ}, tc_mmq_cm1_int,   "matmul_id_subgroup_iq4_xs_q8_1", matmul_id_subgroup_iq4_xs_q8_1_cm1_len, matmul_id_subgroup_iq4_xs_q8_1_cm1_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ);
            cm1_create_mmq({GGML_TYPE_MXFP4,  GGML_TYPE_Q8_1, ਸੱਚ, ਝੂਠ}, tc_mmq_cm1_int,   "matmul_id_subgroup_mxfp4_q8_1",  matmul_id_subgroup_mxfp4_q8_1_cm1_len,  matmul_id_subgroup_mxfp4_q8_1_cm1_ਅੰਕੜੇ,  ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ);
            cm1_create_mmq({GGML_TYPE_Q3_K,   GGML_TYPE_Q8_1, ਸੱਚ, ਝੂਠ}, tc_mmq_cm1_int_k, "matmul_id_subgroup_q3_k_q8_1",   matmul_id_subgroup_q3_k_q8_1_cm1_len,   matmul_id_subgroup_q3_k_q8_1_cm1_ਅੰਕੜੇ,   ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ);
            cm1_create_mmq({GGML_TYPE_Q4_K,   GGML_TYPE_Q8_1, ਸੱਚ, ਝੂਠ}, tc_mmq_cm1_int,   "matmul_id_subgroup_q4_k_q8_1",   matmul_id_subgroup_q4_k_q8_1_cm1_len,   matmul_id_subgroup_q4_k_q8_1_cm1_ਅੰਕੜੇ,   ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ);
            cm1_create_mmq({GGML_TYPE_Q5_K,   GGML_TYPE_Q8_1, ਸੱਚ, ਝੂਠ}, tc_mmq_cm1_int,   "matmul_id_subgroup_q5_k_q8_1",   matmul_id_subgroup_q5_k_q8_1_cm1_len,   matmul_id_subgroup_q5_k_q8_1_cm1_ਅੰਕੜੇ,   ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ);
            cm1_create_mmq({GGML_TYPE_Q6_K,   GGML_TYPE_Q8_1, ਸੱਚ, ਝੂਠ}, tc_mmq_cm1_int_k, "matmul_id_subgroup_q6_k_q8_1",   matmul_id_subgroup_q6_k_q8_1_cm1_len,   matmul_id_subgroup_q6_k_q8_1_cm1_ਅੰਕੜੇ,   ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ);
            ਜੇ (!rdna4) { cm1_create_mmq({GGML_TYPE_NVFP4, GGML_TYPE_Q8_1, ਸੱਚ, ਝੂਠ}, tc_mmq_cm1_int_k, "matmul_id_subgroup_nvfp4_q8_1",  matmul_id_subgroup_nvfp4_q8_1_cm1_len,  matmul_id_subgroup_nvfp4_q8_1_cm1_ਅੰਕੜੇ,  ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ); }
        }
    } ਨਹੀਂਤਾਂ
#ਜੇ_ਅੰਤ  // defined(VK_KHR_cooperative_matrix) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    {
        // Helper for subgroup path with dot2 selection and filtering
        ਸਵੈਚਾਲਿਤ sg_create = [&](vk_matmul_pipeline_ਕੁੰਜੀ ਕੁੰਜੀ, ਅਚਲ std::vector<vk_tile_config>& tc_ਅਧਾਰ,
                             ਅਚਲ std::string& name, size_t len, ਅਚਲ ਖਾਲੀ* ਅੰਕੜੇ, uint32_t pc_ਮਾਪ, uint32_t pc,
                             uint32_t rsgs = 0) {
            ਸਵੈਚਾਲਿਤ tc = filter_tc(tc_ਅਧਾਰ, ਕੁੰਜੀ.type_a, ਕੁੰਜੀ.mul_mat_id);
            ਜੇ (!tc.ਖਾਲੀ_ਥਾਵਾਂ()) create_mm_pipelines(ਕੁੰਜੀ, tc, name, len, ਅੰਕੜੇ, pc_ਮਾਪ, pc,
                [&](ਅਚਲ std::vector<uint32_t>& wt, ਤਾਰਕਿਕ a) { ਵਾਪਸ ggml_vk_mul_mm_spec(wt, a); },
                ਝੂਠ, rsgs > 0, rsgs);
        };
        ਸਵੈਚਾਲਿਤ sg_create_quant = [&](vk_matmul_pipeline_ਕੁੰਜੀ ਕੁੰਜੀ, ਅਚਲ std::vector<vk_tile_config>& tc_ਅਧਾਰ,
                                   ਅਚਲ std::string& name, size_t len, ਅਚਲ ਖਾਲੀ* ਅੰਕੜੇ, uint32_t pc_ਮਾਪ, uint32_t pc,
                                   uint32_t rsgs = 0) {
            ਸਵੈਚਾਲਿਤ tc = filter_tc(tc_ਅਧਾਰ, ਕੁੰਜੀ.type_a, ਕੁੰਜੀ.mul_mat_id);
            ਜੇ (!tc.ਖਾਲੀ_ਥਾਵਾਂ()) {
                spec_fn_t qs = [&, type_a=ਕੁੰਜੀ.type_a](ਅਚਲ std::vector<uint32_t>& wt, ਤਾਰਕਿਕ a) { ਵਾਪਸ ggml_vk_mul_mm_spec_quant(wt, a, (uint32_t)type_a); };
                create_mm_pipelines(ਕੁੰਜੀ, tc, name, len, ਅੰਕੜੇ, pc_ਮਾਪ, pc, qs, ਝੂਠ, rsgs > 0, rsgs);
            }
        };
#ਜੇ defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
        ਸਵੈਚਾਲਿਤ sg_create_mmq = [&](vk_matmul_pipeline_ਕੁੰਜੀ ਕੁੰਜੀ, ਅਚਲ std::vector<vk_tile_config>& tc_ਅਧਾਰ,
                                 ਅਚਲ std::string& name, size_t len, ਅਚਲ ਖਾਲੀ* ਅੰਕੜੇ, uint32_t pc_ਮਾਪ, uint32_t pc,
                                 uint32_t rsgs = 0) {
            ਸਵੈਚਾਲਿਤ tc = filter_tc(tc_ਅਧਾਰ, ਕੁੰਜੀ.type_a, ਕੁੰਜੀ.mul_mat_id, ਸੱਚ);
            ਜੇ (!tc.ਖਾਲੀ_ਥਾਵਾਂ()) {
                spec_fn_t ਪਛਾਣ_ਤੱਤ = [](ਅਚਲ std::vector<uint32_t>& wt, ਤਾਰਕਿਕ) { ਵਾਪਸ wt; };
                create_mm_pipelines(ਕੁੰਜੀ, tc, name, len, ਅੰਕੜੇ, pc_ਮਾਪ, pc, ਪਛਾਣ_ਤੱਤ, ਝੂਠ, rsgs > 0, rsgs, ਝੂਠ);
            }
        };
#ਜੇ_ਅੰਤ

        std::vector<vk_tile_config> tc_id = {{s_warptile_id, s_wg_denoms, s_align}, {m_warptile_id, m_wg_denoms, m_align}, {l_warptile_id, l_wg_denoms, l_align}};
        std::vector<vk_tile_config> tc_mmqid = {{s_warptile_mmqid, s_mmq_wg_denoms, s_align}, {m_warptile_mmqid, m_mmq_wg_denoms, m_align}, {l_warptile_mmqid, l_mmq_wg_denoms, l_align}};

        ਜੇ (device->fp16) {
            // FP16 subgroup path - with dot2 runtime selection
            #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SPV_DOT2(NAME) (device->dot2_f16 ? NAME ## _dot2_len : NAME ## _len), (device->dot2_f16 ? NAME ## dot2_ਅੰਕੜੇ : NAME ## ਅੰਕੜੇ_2)
            #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SPV_DOT2_F16ACC(NAME) (device->dot2_f16 ? NAME ## _dot2_f16acc_len : NAME ## _f16acc_len), (device->dot2_f16 ? NAME ## dot2_f16acc_ਅੰਕੜੇ : NAME ## f16acc_ਅੰਕੜੇ)

            sg_create({GGML_TYPE_F32, GGML_TYPE_F32, ਝੂਠ, ਝੂਠ}, tc_mm, "matmul_f32_f32", SPV_DOT2(matmul_f32_f32), ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3);
            sg_create({GGML_TYPE_F32, GGML_TYPE_F16, ਝੂਠ, ਝੂਠ}, tc_mm, "matmul_f32_f16", SPV_DOT2(matmul_f32_f16), ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3);
            sg_create({GGML_TYPE_F16, GGML_TYPE_F16, ਝੂਠ, ਸੱਚ},  tc_mm, "matmul_f16_f16acc",     SPV_DOT2_F16ACC(matmul_f16),     ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3);
            sg_create({GGML_TYPE_F16, GGML_TYPE_F16, ਝੂਠ, ਝੂਠ}, tc_mm, "matmul_f16",            SPV_DOT2(matmul_f16),            ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3);
            sg_create({GGML_TYPE_F16, GGML_TYPE_F32, ਝੂਠ, ਸੱਚ},  tc_mm, "matmul_f16_f32_f16acc", SPV_DOT2_F16ACC(matmul_f16_f32), ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3);
            sg_create({GGML_TYPE_F16, GGML_TYPE_F32, ਝੂਠ, ਝੂਠ}, tc_mm, "matmul_f16_f32",        SPV_DOT2(matmul_f16_f32),        ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3);
            // BF16 - no dot2
            sg_create({GGML_TYPE_BF16, GGML_TYPE_BF16, ਝੂਠ, ਝੂਠ}, tc_mm, "matmul_bf16", matmul_bf16_len, matmul_bf16_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3);

            ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ type : non_lut_quant_types) {
                sg_create_quant({type, GGML_TYPE_F32, ਝੂਠ, ਸੱਚ},  tc_mmq, "matmul_quant_f32_f16acc", SPV_DOT2_F16ACC(matmul_quant_f32), ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3);
                sg_create_quant({type, GGML_TYPE_F32, ਝੂਠ, ਝੂਠ}, tc_mmq, "matmul_quant_f32",        SPV_DOT2(matmul_quant_f32),        ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3);
            }
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ X_SG(TYPE, tstr) \
            sg_create({TYPE, GGML_TYPE_F32, ਝੂਠ, ਸੱਚ},  tc_mmq, "matmul_" #tstr "_f32_f16acc", SPV_DOT2_F16ACC(matmul_##tstr##_f32), ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3); \
            sg_create({TYPE, GGML_TYPE_F32, ਝੂਠ, ਝੂਠ}, tc_mmq, "matmul_" #tstr "_f32",        SPV_DOT2(matmul_##tstr##_f32),        ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3);
            FOR_EACH_LUT_TYPE(X_SG)
#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ X_SG

#ਜੇ defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
            ਜੇ (device->integer_dot_ਗੁਣਨਫਲ) {
                std::vector<vk_tile_config> tc_mmq_int = {{s_warptile_mmq_int, s_mmq_wg_denoms, s_align}, {m_warptile_mmq_int, m_mmq_wg_denoms, m_align}, {l_warptile_mmq_int, l_mmq_wg_denoms, l_align}};
                std::vector<vk_tile_config> tc_mmq_int_k = {{s_warptile_mmq_int_k, s_mmq_wg_denoms, s_align}, {m_warptile_mmq_int_k, m_mmq_wg_denoms, m_align}, {l_warptile_mmq_int_k, l_mmq_wg_denoms, l_align}};
                sg_create_mmq({GGML_TYPE_Q2_0, GGML_TYPE_Q8_1, ਝੂਠ, ਝੂਠ}, tc_mmq_int, "matmul_q2_0_q8_1", matmul_q2_0_q8_1_len, matmul_q2_0_q8_1_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3);
                sg_create_mmq({GGML_TYPE_Q4_0, GGML_TYPE_Q8_1, ਝੂਠ, ਝੂਠ}, tc_mmq_int, "matmul_q4_0_q8_1", matmul_q4_0_q8_1_len, matmul_q4_0_q8_1_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3);
                sg_create_mmq({GGML_TYPE_Q4_1, GGML_TYPE_Q8_1, ਝੂਠ, ਝੂਠ}, tc_mmq_int, "matmul_q4_1_q8_1", matmul_q4_1_q8_1_len, matmul_q4_1_q8_1_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3);
                sg_create_mmq({GGML_TYPE_Q5_0, GGML_TYPE_Q8_1, ਝੂਠ, ਝੂਠ}, tc_mmq_int, "matmul_q5_0_q8_1", matmul_q5_0_q8_1_len, matmul_q5_0_q8_1_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3);
                sg_create_mmq({GGML_TYPE_Q5_1, GGML_TYPE_Q8_1, ਝੂਠ, ਝੂਠ}, tc_mmq_int, "matmul_q5_1_q8_1", matmul_q5_1_q8_1_len, matmul_q5_1_q8_1_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3);
                sg_create_mmq({GGML_TYPE_Q8_0, GGML_TYPE_Q8_1, ਝੂਠ, ਝੂਠ}, tc_mmq_int, "matmul_q8_0_q8_1", matmul_q8_0_q8_1_len, matmul_q8_0_q8_1_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3);
                sg_create_mmq({GGML_TYPE_MXFP4, GGML_TYPE_Q8_1, ਝੂਠ, ਝੂਠ}, tc_mmq_int, "matmul_mxfp4_q8_1", matmul_mxfp4_q8_1_len, matmul_mxfp4_q8_1_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3);
                sg_create_mmq({GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_1, ਝੂਠ, ਝੂਠ}, tc_mmq_int, "matmul_iq4_xs_q8_1", matmul_iq4_xs_q8_1_len, matmul_iq4_xs_q8_1_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3);
                sg_create_mmq({GGML_TYPE_Q2_K, GGML_TYPE_Q8_1, ਝੂਠ, ਝੂਠ}, tc_mmq_int_k, "matmul_q2_k_q8_1", matmul_q2_k_q8_1_len, matmul_q2_k_q8_1_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3);
                sg_create_mmq({GGML_TYPE_Q3_K, GGML_TYPE_Q8_1, ਝੂਠ, ਝੂਠ}, tc_mmq_int_k, "matmul_q3_k_q8_1", matmul_q3_k_q8_1_len, matmul_q3_k_q8_1_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3);
                sg_create_mmq({GGML_TYPE_Q4_K, GGML_TYPE_Q8_1, ਝੂਠ, ਝੂਠ}, tc_mmq_int_k, "matmul_q4_k_q8_1", matmul_q4_k_q8_1_len, matmul_q4_k_q8_1_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3);
                sg_create_mmq({GGML_TYPE_Q5_K, GGML_TYPE_Q8_1, ਝੂਠ, ਝੂਠ}, tc_mmq_int_k, "matmul_q5_k_q8_1", matmul_q5_k_q8_1_len, matmul_q5_k_q8_1_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3);
                sg_create_mmq({GGML_TYPE_Q6_K, GGML_TYPE_Q8_1, ਝੂਠ, ਝੂਠ}, tc_mmq_int_k, "matmul_q6_k_q8_1", matmul_q6_k_q8_1_len, matmul_q6_k_q8_1_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3);
                sg_create_mmq({GGML_TYPE_IQ3_S, GGML_TYPE_Q8_1, ਝੂਠ, ਝੂਠ}, tc_mmq_int_k, "matmul_iq3_s_q8_1", matmul_iq3_s_q8_1_len, matmul_iq3_s_q8_1_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3);
            }
#ਜੇ_ਅੰਤ

            ਜੇ (device->subgroup_ballot && device->subgroup_require_ਭਰਿਆ_support && subgroup_min_ਮਾਪ_16) {
                sg_create({GGML_TYPE_F32, GGML_TYPE_F32, ਸੱਚ, ਝੂਠ}, tc_id, "matmul_id_subgroup_f32_f32", SPV_DOT2(matmul_id_subgroup_f32_f32), ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ, mul_mat_subgroup_ਮਾਪ_16);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F16, ਸੱਚ, ਸੱਚ},  tc_id, "matmul_id_subgroup_f16_f16acc",     SPV_DOT2_F16ACC(matmul_id_subgroup_f16),     ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ, mul_mat_subgroup_ਮਾਪ_16);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F16, ਸੱਚ, ਝੂਠ}, tc_id, "matmul_id_subgroup_f16",            SPV_DOT2(matmul_id_subgroup_f16),            ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ, mul_mat_subgroup_ਮਾਪ_16);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F32, ਸੱਚ, ਸੱਚ},  tc_id, "matmul_id_subgroup_f16_f32_f16acc", SPV_DOT2_F16ACC(matmul_id_subgroup_f16_f32), ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ, mul_mat_subgroup_ਮਾਪ_16);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F32, ਸੱਚ, ਝੂਠ}, tc_id, "matmul_id_subgroup_f16_f32",        SPV_DOT2(matmul_id_subgroup_f16_f32),        ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ, mul_mat_subgroup_ਮਾਪ_16);
                // BF16 id - no dot2
                sg_create({GGML_TYPE_BF16, GGML_TYPE_BF16, ਸੱਚ, ਝੂਠ}, tc_id, "matmul_id_subgroup_bf16", matmul_id_subgroup_bf16_len, matmul_id_subgroup_bf16_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ, mul_mat_subgroup_ਮਾਪ_16);
                ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ type : non_lut_quant_types) {
                    sg_create_quant({type, GGML_TYPE_F32, ਸੱਚ, ਸੱਚ},  tc_mmqid, "matmul_id_subgroup_quant_f32_f16acc", SPV_DOT2_F16ACC(matmul_id_subgroup_quant_f32), ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ, mul_mat_subgroup_ਮਾਪ);
                    sg_create_quant({type, GGML_TYPE_F32, ਸੱਚ, ਝੂਠ}, tc_mmqid, "matmul_id_subgroup_quant_f32",        SPV_DOT2(matmul_id_subgroup_quant_f32),        ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ, mul_mat_subgroup_ਮਾਪ);
                }
        #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ X_SG_ID_SUB(TYPE, tstr) \
                sg_create({TYPE, GGML_TYPE_F32, ਸੱਚ, ਸੱਚ},  tc_mmqid, "matmul_id_subgroup_" #tstr "_f32_f16acc", SPV_DOT2_F16ACC(matmul_id_subgroup_##tstr##_f32), ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ, mul_mat_subgroup_ਮਾਪ); \
                sg_create({TYPE, GGML_TYPE_F32, ਸੱਚ, ਝੂਠ}, tc_mmqid, "matmul_id_subgroup_" #tstr "_f32",        SPV_DOT2(matmul_id_subgroup_##tstr##_f32),        ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ, mul_mat_subgroup_ਮਾਪ);
                FOR_EACH_LUT_TYPE(X_SG_ID_SUB)
#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ X_SG_ID_SUB
#ਜੇ defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
                ਜੇ (device->integer_dot_ਗੁਣਨਫਲ) {
                    std::vector<vk_tile_config> tc_mmqid_int = {{s_warptile_mmqid_int, s_mmq_wg_denoms, s_align}, {m_warptile_mmqid_int, m_mmq_wg_denoms, m_align}, {l_warptile_mmqid_int, l_mmq_wg_denoms, l_align}};
                    std::vector<vk_tile_config> tc_mmqid_int_k = {{s_warptile_mmqid_int_k, s_mmq_wg_denoms, s_align}, {m_warptile_mmqid_int_k, m_mmq_wg_denoms, m_align}, {l_warptile_mmqid_int_k, l_mmq_wg_denoms, l_align}};
                    sg_create_mmq({GGML_TYPE_Q2_0, GGML_TYPE_Q8_1, ਸੱਚ, ਝੂਠ}, tc_mmqid_int, "matmul_id_subgroup_q2_0_q8_1", matmul_id_subgroup_q2_0_q8_1_len, matmul_id_subgroup_q2_0_q8_1_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ, mul_mat_subgroup_ਮਾਪ);
                    sg_create_mmq({GGML_TYPE_Q4_0, GGML_TYPE_Q8_1, ਸੱਚ, ਝੂਠ}, tc_mmqid_int, "matmul_id_subgroup_q4_0_q8_1", matmul_id_subgroup_q4_0_q8_1_len, matmul_id_subgroup_q4_0_q8_1_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ, mul_mat_subgroup_ਮਾਪ);
                    sg_create_mmq({GGML_TYPE_Q4_1, GGML_TYPE_Q8_1, ਸੱਚ, ਝੂਠ}, tc_mmqid_int, "matmul_id_subgroup_q4_1_q8_1", matmul_id_subgroup_q4_1_q8_1_len, matmul_id_subgroup_q4_1_q8_1_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ, mul_mat_subgroup_ਮਾਪ);
                    sg_create_mmq({GGML_TYPE_Q5_0, GGML_TYPE_Q8_1, ਸੱਚ, ਝੂਠ}, tc_mmqid_int, "matmul_id_subgroup_q5_0_q8_1", matmul_id_subgroup_q5_0_q8_1_len, matmul_id_subgroup_q5_0_q8_1_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ, mul_mat_subgroup_ਮਾਪ);
                    sg_create_mmq({GGML_TYPE_Q5_1, GGML_TYPE_Q8_1, ਸੱਚ, ਝੂਠ}, tc_mmqid_int, "matmul_id_subgroup_q5_1_q8_1", matmul_id_subgroup_q5_1_q8_1_len, matmul_id_subgroup_q5_1_q8_1_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ, mul_mat_subgroup_ਮਾਪ);
                    sg_create_mmq({GGML_TYPE_Q8_0, GGML_TYPE_Q8_1, ਸੱਚ, ਝੂਠ}, tc_mmqid_int, "matmul_id_subgroup_q8_0_q8_1", matmul_id_subgroup_q8_0_q8_1_len, matmul_id_subgroup_q8_0_q8_1_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ, mul_mat_subgroup_ਮਾਪ);
                    sg_create_mmq({GGML_TYPE_MXFP4, GGML_TYPE_Q8_1, ਸੱਚ, ਝੂਠ}, tc_mmqid_int, "matmul_id_subgroup_mxfp4_q8_1", matmul_id_subgroup_mxfp4_q8_1_len, matmul_id_subgroup_mxfp4_q8_1_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ, mul_mat_subgroup_ਮਾਪ);
                    sg_create_mmq({GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_1, ਸੱਚ, ਝੂਠ}, tc_mmqid_int, "matmul_id_subgroup_iq4_xs_q8_1", matmul_id_subgroup_iq4_xs_q8_1_len, matmul_id_subgroup_iq4_xs_q8_1_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ, mul_mat_subgroup_ਮਾਪ);
                    sg_create_mmq({GGML_TYPE_Q2_K, GGML_TYPE_Q8_1, ਸੱਚ, ਝੂਠ}, tc_mmqid_int_k, "matmul_id_subgroup_q2_k_q8_1", matmul_id_subgroup_q2_k_q8_1_len, matmul_id_subgroup_q2_k_q8_1_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ, mul_mat_subgroup_ਮਾਪ_16);
                    sg_create_mmq({GGML_TYPE_Q3_K, GGML_TYPE_Q8_1, ਸੱਚ, ਝੂਠ}, tc_mmqid_int_k, "matmul_id_subgroup_q3_k_q8_1", matmul_id_subgroup_q3_k_q8_1_len, matmul_id_subgroup_q3_k_q8_1_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ, mul_mat_subgroup_ਮਾਪ_16);
                    sg_create_mmq({GGML_TYPE_Q4_K, GGML_TYPE_Q8_1, ਸੱਚ, ਝੂਠ}, tc_mmqid_int_k, "matmul_id_subgroup_q4_k_q8_1", matmul_id_subgroup_q4_k_q8_1_len, matmul_id_subgroup_q4_k_q8_1_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ, mul_mat_subgroup_ਮਾਪ_16);
                    sg_create_mmq({GGML_TYPE_Q5_K, GGML_TYPE_Q8_1, ਸੱਚ, ਝੂਠ}, tc_mmqid_int_k, "matmul_id_subgroup_q5_k_q8_1", matmul_id_subgroup_q5_k_q8_1_len, matmul_id_subgroup_q5_k_q8_1_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ, mul_mat_subgroup_ਮਾਪ_16);
                    sg_create_mmq({GGML_TYPE_Q6_K, GGML_TYPE_Q8_1, ਸੱਚ, ਝੂਠ}, tc_mmqid_int_k, "matmul_id_subgroup_q6_k_q8_1", matmul_id_subgroup_q6_k_q8_1_len, matmul_id_subgroup_q6_k_q8_1_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ, mul_mat_subgroup_ਮਾਪ_16);
                    sg_create_mmq({GGML_TYPE_IQ3_S, GGML_TYPE_Q8_1, ਸੱਚ, ਝੂਠ}, tc_mmqid_int_k, "matmul_id_subgroup_iq3_s_q8_1", matmul_id_subgroup_iq3_s_q8_1_len, matmul_id_subgroup_iq3_s_q8_1_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ, mul_mat_subgroup_ਮਾਪ_16);
                }
#ਜੇ_ਅੰਤ
            } ਨਹੀਂਤਾਂ {
                sg_create({GGML_TYPE_F32, GGML_TYPE_F32, ਸੱਚ, ਝੂਠ}, tc_mm, "matmul_id_f32_f32", SPV_DOT2(matmul_id_f32_f32), ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F16, ਸੱਚ, ਸੱਚ},  tc_mm, "matmul_id_f16_f16acc",     SPV_DOT2_F16ACC(matmul_id_f16),     ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F16, ਸੱਚ, ਝੂਠ}, tc_mm, "matmul_id_f16",            SPV_DOT2(matmul_id_f16),            ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F32, ਸੱਚ, ਸੱਚ},  tc_mm, "matmul_id_f16_f32_f16acc", SPV_DOT2_F16ACC(matmul_id_f16_f32), ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F32, ਸੱਚ, ਝੂਠ}, tc_mm, "matmul_id_f16_f32",        SPV_DOT2(matmul_id_f16_f32),        ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ);
                // BF16 id - no dot2
                sg_create({GGML_TYPE_BF16, GGML_TYPE_BF16, ਸੱਚ, ਝੂਠ}, tc_mm, "matmul_id_bf16", matmul_id_bf16_len, matmul_id_bf16_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ);
                ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ type : non_lut_quant_types) {
                    sg_create_quant({type, GGML_TYPE_F32, ਸੱਚ, ਸੱਚ},  tc_mmqid, "matmul_id_quant_f32_f16acc", SPV_DOT2_F16ACC(matmul_id_quant_f32), ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ);
                    sg_create_quant({type, GGML_TYPE_F32, ਸੱਚ, ਝੂਠ}, tc_mmqid, "matmul_id_quant_f32",        SPV_DOT2(matmul_id_quant_f32),        ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ);
                }
        #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ X_SG_ID(TYPE, tstr) \
                sg_create({TYPE, GGML_TYPE_F32, ਸੱਚ, ਸੱਚ},  tc_mmqid, "matmul_id_" #tstr "_f32_f16acc", SPV_DOT2_F16ACC(matmul_id_##tstr##_f32), ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ); \
                sg_create({TYPE, GGML_TYPE_F32, ਸੱਚ, ਝੂਠ}, tc_mmqid, "matmul_id_" #tstr "_f32",        SPV_DOT2(matmul_id_##tstr##_f32),        ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ);
                FOR_EACH_LUT_TYPE(X_SG_ID)
#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ X_SG_ID
#ਜੇ defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
                ਜੇ (device->integer_dot_ਗੁਣਨਫਲ) {
                    std::vector<vk_tile_config> tc_mmqid_int = {{s_warptile_mmqid_int, s_mmq_wg_denoms, s_align}, {m_warptile_mmqid_int, m_mmq_wg_denoms, m_align}, {l_warptile_mmqid_int, l_mmq_wg_denoms, l_align}};
                    std::vector<vk_tile_config> tc_mmqid_int_k = {{s_warptile_mmqid_int_k, s_mmq_wg_denoms, s_align}, {m_warptile_mmqid_int_k, m_mmq_wg_denoms, m_align}, {l_warptile_mmqid_int_k, l_mmq_wg_denoms, l_align}};
                    sg_create_mmq({GGML_TYPE_Q2_0, GGML_TYPE_Q8_1, ਸੱਚ, ਝੂਠ}, tc_mmqid_int, "matmul_id_q2_0_q8_1", matmul_id_q2_0_q8_1_len, matmul_id_q2_0_q8_1_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ);
                    sg_create_mmq({GGML_TYPE_Q4_0, GGML_TYPE_Q8_1, ਸੱਚ, ਝੂਠ}, tc_mmqid_int, "matmul_id_q4_0_q8_1", matmul_id_q4_0_q8_1_len, matmul_id_q4_0_q8_1_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ);
                    sg_create_mmq({GGML_TYPE_Q4_1, GGML_TYPE_Q8_1, ਸੱਚ, ਝੂਠ}, tc_mmqid_int, "matmul_id_q4_1_q8_1", matmul_id_q4_1_q8_1_len, matmul_id_q4_1_q8_1_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ);
                    sg_create_mmq({GGML_TYPE_Q5_0, GGML_TYPE_Q8_1, ਸੱਚ, ਝੂਠ}, tc_mmqid_int, "matmul_id_q5_0_q8_1", matmul_id_q5_0_q8_1_len, matmul_id_q5_0_q8_1_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ);
                    sg_create_mmq({GGML_TYPE_Q5_1, GGML_TYPE_Q8_1, ਸੱਚ, ਝੂਠ}, tc_mmqid_int, "matmul_id_q5_1_q8_1", matmul_id_q5_1_q8_1_len, matmul_id_q5_1_q8_1_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ);
                    sg_create_mmq({GGML_TYPE_Q8_0, GGML_TYPE_Q8_1, ਸੱਚ, ਝੂਠ}, tc_mmqid_int, "matmul_id_q8_0_q8_1", matmul_id_q8_0_q8_1_len, matmul_id_q8_0_q8_1_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ);
                    sg_create_mmq({GGML_TYPE_MXFP4, GGML_TYPE_Q8_1, ਸੱਚ, ਝੂਠ}, tc_mmqid_int, "matmul_id_mxfp4_q8_1", matmul_id_mxfp4_q8_1_len, matmul_id_mxfp4_q8_1_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ);
                    sg_create_mmq({GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_1, ਸੱਚ, ਝੂਠ}, tc_mmqid_int, "matmul_id_iq4_xs_q8_1", matmul_id_iq4_xs_q8_1_len, matmul_id_iq4_xs_q8_1_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ);
                    sg_create_mmq({GGML_TYPE_Q2_K, GGML_TYPE_Q8_1, ਸੱਚ, ਝੂਠ}, tc_mmqid_int_k, "matmul_id_q2_k_q8_1", matmul_id_q2_k_q8_1_len, matmul_id_q2_k_q8_1_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ);
                    sg_create_mmq({GGML_TYPE_Q3_K, GGML_TYPE_Q8_1, ਸੱਚ, ਝੂਠ}, tc_mmqid_int_k, "matmul_id_q3_k_q8_1", matmul_id_q3_k_q8_1_len, matmul_id_q3_k_q8_1_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ);
                    sg_create_mmq({GGML_TYPE_Q4_K, GGML_TYPE_Q8_1, ਸੱਚ, ਝੂਠ}, tc_mmqid_int_k, "matmul_id_q4_k_q8_1", matmul_id_q4_k_q8_1_len, matmul_id_q4_k_q8_1_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ);
                    sg_create_mmq({GGML_TYPE_Q5_K, GGML_TYPE_Q8_1, ਸੱਚ, ਝੂਠ}, tc_mmqid_int_k, "matmul_id_q5_k_q8_1", matmul_id_q5_k_q8_1_len, matmul_id_q5_k_q8_1_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ);
                    sg_create_mmq({GGML_TYPE_Q6_K, GGML_TYPE_Q8_1, ਸੱਚ, ਝੂਠ}, tc_mmqid_int_k, "matmul_id_q6_k_q8_1", matmul_id_q6_k_q8_1_len, matmul_id_q6_k_q8_1_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ);
                    sg_create_mmq({GGML_TYPE_IQ3_S, GGML_TYPE_Q8_1, ਸੱਚ, ਝੂਠ}, tc_mmqid_int_k, "matmul_id_iq3_s_q8_1", matmul_id_iq3_s_q8_1_len, matmul_id_iq3_s_q8_1_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ);
                }
#ਜੇ_ਅੰਤ
            }
            #ਪਰਿਭਾਸ਼ਾ_ਰੱਦ SPV_DOT2
            #ਪਰਿਭਾਸ਼ਾ_ਰੱਦ SPV_DOT2_F16ACC
        } ਨਹੀਂਤਾਂ {
            // FP32-only fallback path
            sg_create({GGML_TYPE_F32, GGML_TYPE_F32, ਝੂਠ, ਝੂਠ}, tc_mm, "matmul_f32_f32", matmul_f32_f32_fp32_len, matmul_f32_f32_fp32_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3);
            sg_create({GGML_TYPE_F32, GGML_TYPE_F16, ਝੂਠ, ਝੂਠ}, tc_mm, "matmul_f32_f16", matmul_f32_f16_fp32_len, matmul_f32_f16_fp32_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3);
            sg_create({GGML_TYPE_F16, GGML_TYPE_F16, ਝੂਠ, ਝੂਠ}, tc_mm, "matmul_f16",     matmul_f16_fp32_len,     matmul_f16_fp32_ਅੰਕੜੇ,     ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3);
            sg_create({GGML_TYPE_F16, GGML_TYPE_F32, ਝੂਠ, ਝੂਠ}, tc_mm, "matmul_f16_f32", matmul_f16_f32_fp32_len, matmul_f16_f32_fp32_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3);
            sg_create({GGML_TYPE_BF16, GGML_TYPE_BF16, ਝੂਠ, ਝੂਠ}, tc_mm, "matmul_bf16",  matmul_bf16_fp32_len,    matmul_bf16_fp32_ਅੰਕੜੇ,    ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3);

            ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ type : non_lut_quant_types) {
                sg_create_quant({type, GGML_TYPE_F32, ਝੂਠ, ਝੂਠ}, tc_mmq, "matmul_quant_f32", matmul_quant_f32_fp32_len, matmul_quant_f32_fp32_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3);
            }
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ X_SG_FP32(TYPE, tstr) \
            sg_create({TYPE, GGML_TYPE_F32, ਝੂਠ, ਝੂਠ}, tc_mmq, "matmul_" #tstr "_f32", matmul_##tstr##_f32_fp32_len, matmul_##tstr##_f32_fp32_data, ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3);
            FOR_EACH_LUT_TYPE(X_SG_FP32)
#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ X_SG_FP32

#ਜੇ defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
            ਜੇ (device->integer_dot_ਗੁਣਨਫਲ) {
                std::vector<vk_tile_config> tc_mmq_int = {{s_warptile_mmq_int, s_mmq_wg_denoms, s_align}, {m_warptile_mmq_int, m_mmq_wg_denoms, m_align}, {l_warptile_mmq_int, l_mmq_wg_denoms, l_align}};
                std::vector<vk_tile_config> tc_mmq_int_k = {{s_warptile_mmq_int_k, s_mmq_wg_denoms, s_align}, {m_warptile_mmq_int_k, m_mmq_wg_denoms, m_align}, {l_warptile_mmq_int_k, l_mmq_wg_denoms, l_align}};
                sg_create_mmq({GGML_TYPE_Q2_0, GGML_TYPE_Q8_1, ਝੂਠ, ਝੂਠ}, tc_mmq_int, "matmul_q2_0_q8_1", matmul_q2_0_q8_1_fp32_len, matmul_q2_0_q8_1_fp32_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3);
                sg_create_mmq({GGML_TYPE_Q4_0, GGML_TYPE_Q8_1, ਝੂਠ, ਝੂਠ}, tc_mmq_int, "matmul_q4_0_q8_1", matmul_q4_0_q8_1_fp32_len, matmul_q4_0_q8_1_fp32_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3);
                sg_create_mmq({GGML_TYPE_Q4_1, GGML_TYPE_Q8_1, ਝੂਠ, ਝੂਠ}, tc_mmq_int, "matmul_q4_1_q8_1", matmul_q4_1_q8_1_fp32_len, matmul_q4_1_q8_1_fp32_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3);
                sg_create_mmq({GGML_TYPE_Q5_0, GGML_TYPE_Q8_1, ਝੂਠ, ਝੂਠ}, tc_mmq_int, "matmul_q5_0_q8_1", matmul_q5_0_q8_1_fp32_len, matmul_q5_0_q8_1_fp32_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3);
                sg_create_mmq({GGML_TYPE_Q5_1, GGML_TYPE_Q8_1, ਝੂਠ, ਝੂਠ}, tc_mmq_int, "matmul_q5_1_q8_1", matmul_q5_1_q8_1_fp32_len, matmul_q5_1_q8_1_fp32_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3);
                sg_create_mmq({GGML_TYPE_Q8_0, GGML_TYPE_Q8_1, ਝੂਠ, ਝੂਠ}, tc_mmq_int, "matmul_q8_0_q8_1", matmul_q8_0_q8_1_fp32_len, matmul_q8_0_q8_1_fp32_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3);
                sg_create_mmq({GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_1, ਝੂਠ, ਝੂਠ}, tc_mmq_int, "matmul_iq4_xs_q8_1", matmul_iq4_xs_q8_1_fp32_len, matmul_iq4_xs_q8_1_fp32_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3);
                sg_create_mmq({GGML_TYPE_Q2_K, GGML_TYPE_Q8_1, ਝੂਠ, ਝੂਠ}, tc_mmq_int_k, "matmul_q2_k_q8_1", matmul_q2_k_q8_1_fp32_len, matmul_q2_k_q8_1_fp32_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3);
                sg_create_mmq({GGML_TYPE_Q3_K, GGML_TYPE_Q8_1, ਝੂਠ, ਝੂਠ}, tc_mmq_int_k, "matmul_q3_k_q8_1", matmul_q3_k_q8_1_fp32_len, matmul_q3_k_q8_1_fp32_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3);
                sg_create_mmq({GGML_TYPE_Q4_K, GGML_TYPE_Q8_1, ਝੂਠ, ਝੂਠ}, tc_mmq_int_k, "matmul_q4_k_q8_1", matmul_q4_k_q8_1_fp32_len, matmul_q4_k_q8_1_fp32_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3);
                sg_create_mmq({GGML_TYPE_Q5_K, GGML_TYPE_Q8_1, ਝੂਠ, ਝੂਠ}, tc_mmq_int_k, "matmul_q5_k_q8_1", matmul_q5_k_q8_1_fp32_len, matmul_q5_k_q8_1_fp32_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3);
                sg_create_mmq({GGML_TYPE_Q6_K, GGML_TYPE_Q8_1, ਝੂਠ, ਝੂਠ}, tc_mmq_int_k, "matmul_q6_k_q8_1", matmul_q6_k_q8_1_fp32_len, matmul_q6_k_q8_1_fp32_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3);
                sg_create_mmq({GGML_TYPE_IQ3_S, GGML_TYPE_Q8_1, ਝੂਠ, ਝੂਠ}, tc_mmq_int_k, "matmul_iq3_s_q8_1", matmul_iq3_s_q8_1_fp32_len, matmul_iq3_s_q8_1_fp32_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3);
            }
#ਜੇ_ਅੰਤ

            ਜੇ (device->subgroup_ballot && device->subgroup_require_ਭਰਿਆ_support && subgroup_min_ਮਾਪ_16) {
                sg_create({GGML_TYPE_F32, GGML_TYPE_F32, ਸੱਚ, ਝੂਠ}, tc_id, "matmul_id_subgroup_f32_f32", matmul_id_subgroup_f32_f32_fp32_len, matmul_id_subgroup_f32_f32_fp32_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ, mul_mat_subgroup_ਮਾਪ_16);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F16, ਸੱਚ, ਝੂਠ}, tc_id, "matmul_id_subgroup_f16",     matmul_id_subgroup_f16_fp32_len,     matmul_id_subgroup_f16_fp32_ਅੰਕੜੇ,     ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ, mul_mat_subgroup_ਮਾਪ_16);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F32, ਸੱਚ, ਝੂਠ}, tc_id, "matmul_id_subgroup_f16_f32", matmul_id_subgroup_f16_f32_fp32_len, matmul_id_subgroup_f16_f32_fp32_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ, mul_mat_subgroup_ਮਾਪ_16);
                sg_create({GGML_TYPE_BF16, GGML_TYPE_BF16, ਸੱਚ, ਝੂਠ}, tc_id, "matmul_id_subgroup_bf16",  matmul_id_subgroup_bf16_fp32_len,    matmul_id_subgroup_bf16_fp32_ਅੰਕੜੇ,    ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ, mul_mat_subgroup_ਮਾਪ_16);
                ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ type : non_lut_quant_types) {
                    sg_create_quant({type, GGML_TYPE_F32, ਸੱਚ, ਝੂਠ}, tc_mmqid, "matmul_id_subgroup_quant_f32", matmul_id_subgroup_quant_f32_fp32_len, matmul_id_subgroup_quant_f32_fp32_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ, mul_mat_subgroup_ਮਾਪ);
                }
        #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ X_SG_ID_SUB_FP32(TYPE, tstr) \
                sg_create({TYPE, GGML_TYPE_F32, ਸੱਚ, ਝੂਠ}, tc_mmqid, "matmul_id_subgroup_" #tstr "_f32", matmul_id_subgroup_##tstr##_f32_fp32_len, matmul_id_subgroup_##tstr##_f32_fp32_data, ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ, mul_mat_subgroup_ਮਾਪ);
                FOR_EACH_LUT_TYPE(X_SG_ID_SUB_FP32)
#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ X_SG_ID_SUB_FP32
            } ਨਹੀਂਤਾਂ {
                sg_create({GGML_TYPE_F32, GGML_TYPE_F32, ਸੱਚ, ਝੂਠ}, tc_mm, "matmul_id_f32_f32", matmul_id_f32_f32_fp32_len, matmul_id_f32_f32_fp32_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F16, ਸੱਚ, ਝੂਠ}, tc_mm, "matmul_id_f16",     matmul_id_f16_fp32_len,     matmul_id_f16_fp32_ਅੰਕੜੇ,     ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F32, ਸੱਚ, ਝੂਠ}, tc_mm, "matmul_id_f16_f32", matmul_id_f16_f32_fp32_len, matmul_id_f16_f32_fp32_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ);
                sg_create({GGML_TYPE_BF16, GGML_TYPE_BF16, ਸੱਚ, ਝੂਠ}, tc_mm, "matmul_id_bf16",  matmul_id_bf16_fp32_len,    matmul_id_bf16_fp32_ਅੰਕੜੇ,    ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ);
                ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ type : non_lut_quant_types) {
                    sg_create_quant({type, GGML_TYPE_F32, ਸੱਚ, ਝੂਠ}, tc_mmqid, "matmul_id_quant_f32", matmul_id_quant_f32_fp32_len, matmul_id_quant_f32_fp32_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ);
                }
        #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ X_SG_ID_FP32(TYPE, tstr) \
                sg_create({TYPE, GGML_TYPE_F32, ਸੱਚ, ਝੂਠ}, tc_mmqid, "matmul_id_" #tstr "_f32", matmul_id_##tstr##_f32_fp32_len, matmul_id_##tstr##_f32_fp32_data, ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ);
                FOR_EACH_LUT_TYPE(X_SG_ID_FP32)
#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ X_SG_ID_FP32
            }
        }
    }
#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ FOR_EACH_LUT_TYPE
#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ FOR_EACH_LUT_TYPE_NONFP4
#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ FOR_EACH_LUT_FP4_TYPE
    // BF16 fallback for coopmat devices without bf16 coopmat support
    ਜੇ ((device->coopmat2 || device->coopmat_support)
#ਜੇ defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        && !device->coopmat_bf16_support
#ਜੇ_ਅੰਤ
        ) {
        ਅਚਲ uint32_t s_warptile_wm_bf16 = device->subgroup_ਮਾਪ == 8 ? 8 : 32;
        std::vector<vk_tile_config> tc_bf16_fb = {
            {{ subgroup_ਮਾਪ_32, 32, 32, 16, s_warptile_wm_bf16, 32, 2, 2, 2, 1, subgroup_ਮਾਪ_8 }, {32, 32, 1}, s_align},
            {{ 128, 64, 64, 16, mm_warp_8, 32, 2, 4, 2, 1, mm_warp_8 }, {64, 64, 1}, m_align},
            {{ 128, 128, 128, 16, mm_warp_8 * 2, 64, 2, 4, 4, 1, mm_warp_8 }, {128, 128, 1}, l_align},
        };
        ਸਵੈਚਾਲਿਤ tc_bf16_filtered = filter_tc(tc_bf16_fb, GGML_TYPE_BF16, ਝੂਠ);
        ਸਵੈਚਾਲਿਤ tc_bf16_id_filtered = filter_tc(tc_bf16_fb, GGML_TYPE_BF16, ਸੱਚ);
        spec_fn_t bf16_spec = [&](ਅਚਲ std::vector<uint32_t>& wt, ਤਾਰਕਿਕ a) { ਵਾਪਸ ggml_vk_mul_mm_spec(wt, a); };
        ਜੇ (!tc_bf16_filtered.ਖਾਲੀ_ਥਾਵਾਂ()) {
            create_mm_pipelines({GGML_TYPE_BF16, GGML_TYPE_BF16, ਝੂਠ, ਝੂਠ}, tc_bf16_filtered, "matmul_bf16", matmul_bf16_fp32_len, matmul_bf16_fp32_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_ਧੱਕੋ_constants), 3, bf16_spec);
        }
        ਜੇ (!tc_bf16_id_filtered.ਖਾਲੀ_ਥਾਵਾਂ()) {
            create_mm_pipelines({GGML_TYPE_BF16, GGML_TYPE_BF16, ਸੱਚ, ਝੂਠ}, tc_bf16_id_filtered, "matmul_id_bf16", matmul_id_bf16_fp32_len, matmul_id_bf16_fp32_ਅੰਕੜੇ, ਆਕਾਰ(vk_mat_mat_id_ਧੱਕੋ_constants), mul_mat_id_param_ਗਿਣਤੀ, bf16_spec);
        }
    }

    // Set up tile selector functions
    ਜੇ (device->coopmat2) {
        device->matmul_tile_selector = [](uint32_t m, uint32_t n, uint32_t /*k*/, uint32_t shader_core_ਗਿਣਤੀ,
                                          ਅਚਲ std::vector<vk_matmul_pipeline_pair>& configs) -> uint32_t {
            ਜੇ (configs.size() <= 1) ਵਾਪਸ 0;
            uint32_t ਆਖਰੀ = (uint32_t)configs.size() - 1;
            ਜੇ (configs.size() == 2) {
                uint32_t crossover = configs[0].unaligned->wg_denoms[1];
                ਵਾਪਸ (n > crossover) ? 1 : 0;
            }
            // 3+ configs: s=0, m=1, l=2
            ਅਚਲ uint32_t tiles_l = CEIL_DIV(m, configs[ਆਖਰੀ].unaligned->wg_denoms[0]) * CEIL_DIV(n, configs[ਆਖਰੀ].unaligned->wg_denoms[1]);
            ਅਚਲ uint32_t tiles_m = CEIL_DIV(m, configs[1].unaligned->wg_denoms[0]) * CEIL_DIV(n, configs[1].unaligned->wg_denoms[1]);
            uint32_t crossover_large = configs[1].unaligned->wg_denoms[1];
            ਤਾਰਕਿਕ prefer_large = tiles_m > shader_core_ਗਿਣਤੀ || tiles_l > shader_core_ਗਿਣਤੀ ||
                                (tiles_l <= shader_core_ਗਿਣਤੀ / 3 && tiles_m > shader_core_ਗਿਣਤੀ / 2);
            ਜੇ (n > crossover_large && prefer_large) ਵਾਪਸ ਆਖਰੀ;
            uint32_t crossover_medium_m = configs[0].unaligned->wg_denoms[0];
            uint32_t crossover_medium_n = configs[0].unaligned->wg_denoms[1];
            ਜੇ (m > crossover_medium_m && n > crossover_medium_n) ਵਾਪਸ 1;
            ਵਾਪਸ 0;
        };
        device->matmul_id_tile_selector = [](uint32_t /*m*/, uint32_t n, uint32_t /*k*/, uint32_t /*shader_core_count*/,
                                             ਅਚਲ std::vector<vk_matmul_pipeline_pair>& configs) -> uint32_t {
            ਜੇ (configs.size() <= 1) ਵਾਪਸ 0;
            uint32_t ਆਖਰੀ = (uint32_t)configs.size() - 1;
            ਜੇ (configs.size() == 2) {
                uint32_t crossover = configs[0].unaligned->wg_denoms[1];
                ਵਾਪਸ (n > crossover) ? 1 : 0;
            }
            uint32_t crossover_large = configs[1].unaligned->wg_denoms[1];
            ਜੇ (n > crossover_large) ਵਾਪਸ ਆਖਰੀ;
            uint32_t crossover_medium = configs[0].unaligned->wg_denoms[1];
            ਜੇ (n > crossover_medium) ਵਾਪਸ 1;
            ਵਾਪਸ 0;
        };
    } ਨਹੀਂਤਾਂ {
        device->matmul_tile_selector = [](uint32_t m, uint32_t n, uint32_t /*k*/, uint32_t /*shader_core_count*/,
                                          ਅਚਲ std::vector<vk_matmul_pipeline_pair>& configs) -> uint32_t {
            ਜੇ (configs.size() <= 1) ਵਾਪਸ 0;
            ਜੇ (m <= 32 || n <= 32) ਵਾਪਸ 0;
            ਜੇ (configs.size() == 2) ਵਾਪਸ 1;
            ਜੇ (m <= 64 || n <= 64) ਵਾਪਸ 1;
            ਵਾਪਸ (uint32_t)configs.size() - 1;
        };
        device->matmul_id_tile_selector = device->matmul_tile_selector;
    }

    // mul mat vec

    // the number of rows computed per shader depends on GPU model and quant
    uint32_t rm_stdq = 1;
    uint32_t rm_kq = 2;
    uint32_t rm_stdq_int = 1;
    uint32_t rm_kq_int = 1;
    ਸਵੈਚਾਲਿਤ ਅਚਲ &rm_iq_int = [](uint32_t i) { ਵਾਪਸ i == 0 ? 8u : 4u; };
    ਜੇ (device->vendor_id == VK_VENDOR_ID_AMD) {
        ਜੇ (device->architecture == AMD_GCN) {
            rm_stdq = 2;
            rm_kq = 4;
            rm_stdq_int = 4;
        }
    } ਨਹੀਂਤਾਂ ਜੇ (device->vendor_id == VK_VENDOR_ID_INTEL) {
        rm_stdq = 2;
        rm_stdq_int = 2;
    }
    // RDNA3/4: above four columns, static 4 rows for all types bench faster than the default
    ਅਚਲ ਤਾਰਕਿਕ is_rdna3_or_4 = device->vendor_id == VK_VENDOR_ID_AMD && (device->architecture == AMD_RDNA3 || device->architecture == AMD_RDNA4);
    ਸਵੈਚਾਲਿਤ ਅਚਲ &rm_int_n = [&](uint32_t ਪੰਕਤੀਆਂ, uint32_t i) { ਵਾਪਸ (is_rdna3_or_4 && i >= 4) ? 4u : ਪੰਕਤੀਆਂ; };
    // RDNA3/4 and NVIDIA except pre-Turing: use 4 rows for MUL_MAT_ID MMVQ.
    ਸਵੈਚਾਲਿਤ ਅਚਲ &rm_id = [&](uint32_t ਪੰਕਤੀਆਂ) {
        ਜੇ (device->vendor_id == VK_VENDOR_ID_NVIDIA &&
            device->architecture != vk_device_architecture::NVIDIA_PRE_TURING) {
            ਵਾਪਸ 4u;
        }
        ਵਾਪਸ is_rdna3_or_4 ? 4u : ਪੰਕਤੀਆਂ;
    };
    uint32_t rm_iq = 2 * rm_kq;

    ਅਚਲ ਤਾਰਕਿਕ use_subgroups = device->subgroup_arithmetic;
    // The Imagination proprietary compiler rejects the subgroup-only dequant mul_mat_vec
    // shaders that require a subgroup size >= 16; fall back to shared-memory reduction.
    ਅਚਲ ਤਾਰਕਿਕ is_imagination_proprietary =
        device->driver_id == vk::DriverId::eImaginationProprietary;
    // Ensure a subgroup size >= 16 is available
    ਅਚਲ ਤਾਰਕਿਕ use_subgroups16 = use_subgroups && subgroup_min_ਮਾਪ_16 && !is_imagination_proprietary;

    ਅਚਲ uint32_t subgroup_ਮਾਪ = (device->vendor_id == VK_VENDOR_ID_INTEL && device->subgroup_ਮਾਪ_control && device->subgroup_min_ਮਾਪ <= 16 && device->subgroup_max_ਮਾਪ >= 16) ? 16 : device->subgroup_ਮਾਪ;
    ਅਚਲ uint32_t subgroup_size16 = std::max(subgroup_ਮਾਪ, 16u);

    ਅਚਲ uint32_t force_subgroup_ਮਾਪ = use_subgroups ? subgroup_ਮਾਪ : 0;
    ਅਚਲ uint32_t force_subgroup_size16 = use_subgroups16 ? subgroup_size16 : 0;
    ਸਥਿਰ constexpr uint32_t mul_mat_vec_ਅੰਕ_bindings = 5;
    ਸਥਿਰ constexpr uint32_t mul_mat_vec_id_ਅੰਕ_bindings = 6;

#ਜੇ defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ OCP_DMMV_LEN(NAME, REDUC)  (device->ocp_fp4 ? NAME ## _ocp_len[REDUC]  : NAME ## _len[REDUC])
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ OCP_DMMV_ਅੰਕੜੇ(NAME, REDUC) (device->ocp_fp4 ? NAME ## ocp_ਅੰਕੜੇ[REDUC] : NAME ## ਅੰਕੜੇ_2[REDUC])
#ਨਹੀਂਤਾਂ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ OCP_DMMV_LEN(NAME, REDUC)  NAME ## _len[REDUC]
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ OCP_DMMV_ਅੰਕੜੇ(NAME, REDUC) NAME ## ਅੰਕੜੇ_2[REDUC]
#ਜੇ_ਅੰਤ

    ਲਈ (uint32_t w = 0; w < DMMV_WG_ਮਾਪ_ਗਿਣਤੀ; ++w) {
        ਅਚਲ uint32_t wg_ਮਾਪ_subgroup   = (w == DMMV_WG_ਮਾਪ_SUBGROUP) ? subgroup_ਮਾਪ : (subgroup_ਮਾਪ * 4);
        ਅਚਲ uint32_t wg_ਮਾਪ_subgroup16 = (w == DMMV_WG_ਮਾਪ_SUBGROUP) ? subgroup_size16 : (subgroup_size16 * 4);

        ਅਚਲ shader_reduction_ਬਹੁਲਕ reduc = (use_subgroups && w == DMMV_WG_ਮਾਪ_SUBGROUP) ? SHADER_REDUCTION_ਬਹੁਲਕ_SUBGROUP :
                                            (use_subgroups && w == DMMV_WG_ਮਾਪ_LARGE) ? SHADER_REDUCTION_ਬਹੁਲਕ_HYBRID :
                                            SHADER_REDUCTION_ਬਹੁਲਕ_SHMEM;

        ਅਚਲ shader_reduction_ਬਹੁਲਕ reduc16 = (use_subgroups16 && w == DMMV_WG_ਮਾਪ_SUBGROUP) ? SHADER_REDUCTION_ਬਹੁਲਕ_SUBGROUP :
                                              (use_subgroups16 && w == DMMV_WG_ਮਾਪ_LARGE) ? SHADER_REDUCTION_ਬਹੁਲਕ_HYBRID :
                                              SHADER_REDUCTION_ਬਹੁਲਕ_SHMEM;

        ਲਈ (uint32_t i = 0; i < mul_mat_vec_max_ਖਾਨੇ_ਗਿਣਤੀ; ++i) {
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_F32 ][i], "mul_mat_vec_f32_f32_f32",  arr_dmmv_f32_f32_f32_len[reduc],  arr_dmmv_f32_f32_f32_ਅੰਕੜੇ[reduc],  "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {1, 1, 1}, {wg_ਮਾਪ_subgroup, 1, i+1}, 1, ਝੂਠ, use_subgroups, force_subgroup_ਮਾਪ);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_F16 ][i], "mul_mat_vec_f16_f32_f32",  arr_dmmv_f16_f32_f32_len[reduc],  arr_dmmv_f16_f32_f32_ਅੰਕੜੇ[reduc],  "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {2, 1, 1}, {wg_ਮਾਪ_subgroup, 2, i+1}, 1, ਝੂਠ, use_subgroups, force_subgroup_ਮਾਪ);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_BF16][i], "mul_mat_vec_bf16_f32_f32", arr_dmmv_bf16_f32_f32_len[reduc], arr_dmmv_bf16_f32_f32_ਅੰਕੜੇ[reduc], "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {2, 1, 1}, {wg_ਮਾਪ_subgroup, 2, i+1}, 1, ਝੂਠ, use_subgroups, force_subgroup_ਮਾਪ);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q1_0][i], "mul_mat_vec_q1_0_f32_f32", arr_dmmv_q1_0_f32_f32_len[reduc], arr_dmmv_q1_0_f32_f32_ਅੰਕੜੇ[reduc], "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {2*rm_stdq, 1, 1}, {wg_ਮਾਪ_subgroup, 2*rm_stdq, i+1}, 1, ਸੱਚ, use_subgroups, force_subgroup_ਮਾਪ);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q2_0][i], "mul_mat_vec_q2_0_f32_f32", arr_dmmv_q2_0_f32_f32_len[reduc], arr_dmmv_q2_0_f32_f32_ਅੰਕੜੇ[reduc], "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {2*rm_stdq, 1, 1}, {wg_ਮਾਪ_subgroup, 2*rm_stdq, i+1}, 1, ਸੱਚ, use_subgroups, force_subgroup_ਮਾਪ);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q4_0][i], "mul_mat_vec_q4_0_f32_f32", arr_dmmv_q4_0_f32_f32_len[reduc], arr_dmmv_q4_0_f32_f32_ਅੰਕੜੇ[reduc], "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {2*rm_stdq, 1, 1}, {wg_ਮਾਪ_subgroup, 2*rm_stdq, i+1}, 1, ਸੱਚ, use_subgroups, force_subgroup_ਮਾਪ);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q4_1][i], "mul_mat_vec_q4_1_f32_f32", arr_dmmv_q4_1_f32_f32_len[reduc], arr_dmmv_q4_1_f32_f32_ਅੰਕੜੇ[reduc], "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {2*rm_stdq, 1, 1}, {wg_ਮਾਪ_subgroup, 2*rm_stdq, i+1}, 1, ਸੱਚ, use_subgroups, force_subgroup_ਮਾਪ);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q5_0][i], "mul_mat_vec_q5_0_f32_f32", arr_dmmv_q5_0_f32_f32_len[reduc], arr_dmmv_q5_0_f32_f32_ਅੰਕੜੇ[reduc], "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {2*rm_stdq, 1, 1}, {wg_ਮਾਪ_subgroup, 2*rm_stdq, i+1}, 1, ਸੱਚ, use_subgroups, force_subgroup_ਮਾਪ);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q5_1][i], "mul_mat_vec_q5_1_f32_f32", arr_dmmv_q5_1_f32_f32_len[reduc], arr_dmmv_q5_1_f32_f32_ਅੰਕੜੇ[reduc], "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {2*rm_stdq, 1, 1}, {wg_ਮਾਪ_subgroup, 2*rm_stdq, i+1}, 1, ਸੱਚ, use_subgroups, force_subgroup_ਮਾਪ);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q8_0][i], "mul_mat_vec_q8_0_f32_f32", arr_dmmv_q8_0_f32_f32_len[reduc], arr_dmmv_q8_0_f32_f32_ਅੰਕੜੇ[reduc], "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {1*rm_stdq, 1, 1}, {wg_ਮਾਪ_subgroup, 1*rm_stdq, i+1}, 1, ਸੱਚ, use_subgroups, force_subgroup_ਮਾਪ);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q2_K][i], "mul_mat_vec_q2_k_f32_f32", arr_dmmv_q2_k_f32_f32_len[reduc16], arr_dmmv_q2_k_f32_f32_ਅੰਕੜੇ[reduc16], "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {rm_kq, 1, 1}, {wg_ਮਾਪ_subgroup16, rm_kq, i+1}, 1, ਸੱਚ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q3_K][i], "mul_mat_vec_q3_k_f32_f32", arr_dmmv_q3_k_f32_f32_len[reduc16], arr_dmmv_q3_k_f32_f32_ਅੰਕੜੇ[reduc16], "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {rm_kq, 1, 1}, {wg_ਮਾਪ_subgroup16, rm_kq, i+1}, 1, ਸੱਚ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q4_K][i], "mul_mat_vec_q4_k_f32_f32", arr_dmmv_q4_k_f32_f32_len[reduc16], arr_dmmv_q4_k_f32_f32_ਅੰਕੜੇ[reduc16], "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {rm_kq, 1, 1}, {wg_ਮਾਪ_subgroup16, rm_kq, i+1}, 1, ਸੱਚ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q5_K][i], "mul_mat_vec_q5_k_f32_f32", arr_dmmv_q5_k_f32_f32_len[reduc16], arr_dmmv_q5_k_f32_f32_ਅੰਕੜੇ[reduc16], "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {rm_kq, 1, 1}, {wg_ਮਾਪ_subgroup16, rm_kq, i+1}, 1, ਸੱਚ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q6_K][i], "mul_mat_vec_q6_k_f32_f32", arr_dmmv_q6_k_f32_f32_len[reduc16], arr_dmmv_q6_k_f32_f32_ਅੰਕੜੇ[reduc16], "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {rm_kq, 1, 1}, {wg_ਮਾਪ_subgroup16, rm_kq, i+1}, 1, ਸੱਚ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_TQ1_0][i], "mul_mat_vec_tq1_0_f32_f32", arr_dmmv_tq1_0_f32_f32_len[reduc16], arr_dmmv_tq1_0_f32_f32_ਅੰਕੜੇ[reduc16], "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {rm_kq, 1, 1}, {wg_ਮਾਪ_subgroup16, rm_kq, i+1}, 1, ਸੱਚ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_TQ2_0][i], "mul_mat_vec_tq2_0_f32_f32", arr_dmmv_tq2_0_f32_f32_len[reduc16], arr_dmmv_tq2_0_f32_f32_ਅੰਕੜੇ[reduc16], "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {rm_kq, 1, 1}, {wg_ਮਾਪ_subgroup16, rm_kq, i+1}, 1, ਸੱਚ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ1_S][i],   "mul_mat_vec_iq1_s_f32_f32",   arr_dmmv_iq1_s_f32_f32_len[reduc16],   arr_dmmv_iq1_s_f32_f32_ਅੰਕੜੇ[reduc16],   "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {rm_iq, 1, 1}, {wg_ਮਾਪ_subgroup16, rm_iq, i+1}, 1, ਸੱਚ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ1_M][i],   "mul_mat_vec_iq1_m_f32_f32",   arr_dmmv_iq1_m_f32_f32_len[reduc16],   arr_dmmv_iq1_m_f32_f32_ਅੰਕੜੇ[reduc16],   "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {rm_iq, 1, 1}, {wg_ਮਾਪ_subgroup16, rm_iq, i+1}, 1, ਸੱਚ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ2_XXS][i], "mul_mat_vec_iq2_xxs_f32_f32", arr_dmmv_iq2_xxs_f32_f32_len[reduc16], arr_dmmv_iq2_xxs_f32_f32_ਅੰਕੜੇ[reduc16], "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {rm_iq, 1, 1}, {wg_ਮਾਪ_subgroup16, rm_iq, i+1}, 1, ਸੱਚ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ2_XS][i],  "mul_mat_vec_iq2_xs_f32_f32",  arr_dmmv_iq2_xs_f32_f32_len[reduc16],  arr_dmmv_iq2_xs_f32_f32_ਅੰਕੜੇ[reduc16],  "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {rm_iq, 1, 1}, {wg_ਮਾਪ_subgroup16, rm_iq, i+1}, 1, ਸੱਚ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ2_S][i],   "mul_mat_vec_iq2_s_f32_f32",   arr_dmmv_iq2_s_f32_f32_len[reduc16],   arr_dmmv_iq2_s_f32_f32_ਅੰਕੜੇ[reduc16],   "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {rm_iq, 1, 1}, {wg_ਮਾਪ_subgroup16, rm_iq, i+1}, 1, ਸੱਚ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ3_XXS][i], "mul_mat_vec_iq3_xxs_f32_f32", arr_dmmv_iq3_xxs_f32_f32_len[reduc16], arr_dmmv_iq3_xxs_f32_f32_ਅੰਕੜੇ[reduc16], "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {rm_iq, 1, 1}, {wg_ਮਾਪ_subgroup16, rm_iq, i+1}, 1, ਸੱਚ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ3_S][i],   "mul_mat_vec_iq3_s_f32_f32",   arr_dmmv_iq3_s_f32_f32_len[reduc16],   arr_dmmv_iq3_s_f32_f32_ਅੰਕੜੇ[reduc16],   "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {rm_iq, 1, 1}, {wg_ਮਾਪ_subgroup16, rm_iq, i+1}, 1, ਸੱਚ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ4_XS][i],  "mul_mat_vec_iq4_xs_f32_f32",  arr_dmmv_iq4_xs_f32_f32_len[reduc16],  arr_dmmv_iq4_xs_f32_f32_ਅੰਕੜੇ[reduc16],  "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {rm_iq, 1, 1}, {wg_ਮਾਪ_subgroup16, rm_iq, i+1}, 1, ਸੱਚ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ4_NL][i],  "mul_mat_vec_iq4_nl_f32_f32",  arr_dmmv_iq4_nl_f32_f32_len[reduc16],  arr_dmmv_iq4_nl_f32_f32_ਅੰਕੜੇ[reduc16],  "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {rm_iq, 1, 1}, {wg_ਮਾਪ_subgroup16, rm_iq, i+1}, 1, ਸੱਚ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_MXFP4][i],   "mul_mat_vec_mxfp4_f32_f32",   OCP_DMMV_LEN(arr_dmmv_mxfp4_f32_f32, reduc16), OCP_DMMV_ਅੰਕੜੇ(arr_dmmv_mxfp4_f32_f32, reduc16), "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {rm_iq, 1, 1}, {wg_ਮਾਪ_subgroup16, rm_iq, i+1}, 1, ਸੱਚ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_NVFP4][i],   "mul_mat_vec_nvfp4_f32_f32",   OCP_DMMV_LEN(arr_dmmv_nvfp4_f32_f32, reduc16), OCP_DMMV_ਅੰਕੜੇ(arr_dmmv_nvfp4_f32_f32, reduc16), "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {rm_iq, 1, 1}, {wg_ਮਾਪ_subgroup16, rm_iq, i+1}, 1, ਸੱਚ, use_subgroups16, force_subgroup_size16);

            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_F32 ][i], "mul_mat_vec_f32_f16_f32",  arr_dmmv_f32_f16_f32_len[reduc],  arr_dmmv_f32_f16_f32_ਅੰਕੜੇ[reduc],  "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {1, 1, 1}, {wg_ਮਾਪ_subgroup, 1, i+1}, 1, ਝੂਠ, use_subgroups, force_subgroup_ਮਾਪ);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_F16 ][i], "mul_mat_vec_f16_f16_f32",  arr_dmmv_f16_f16_f32_len[reduc],  arr_dmmv_f16_f16_f32_ਅੰਕੜੇ[reduc],  "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {2, 1, 1}, {wg_ਮਾਪ_subgroup, 2, i+1}, 1, ਝੂਠ, use_subgroups, force_subgroup_ਮਾਪ);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_BF16][i], "mul_mat_vec_bf16_f16_f32", arr_dmmv_bf16_f16_f32_len[reduc], arr_dmmv_bf16_f16_f32_ਅੰਕੜੇ[reduc], "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {2, 1, 1}, {wg_ਮਾਪ_subgroup, 2, i+1}, 1, ਝੂਠ, use_subgroups, force_subgroup_ਮਾਪ);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q1_0][i], "mul_mat_vec_q1_0_f16_f32", arr_dmmv_q1_0_f16_f32_len[reduc], arr_dmmv_q1_0_f16_f32_ਅੰਕੜੇ[reduc], "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {2*rm_stdq, 1, 1}, {wg_ਮਾਪ_subgroup, 2*rm_stdq, i+1}, 1, ਸੱਚ, use_subgroups, force_subgroup_ਮਾਪ);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q2_0][i], "mul_mat_vec_q2_0_f16_f32", arr_dmmv_q2_0_f16_f32_len[reduc], arr_dmmv_q2_0_f16_f32_ਅੰਕੜੇ[reduc], "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {2*rm_stdq, 1, 1}, {wg_ਮਾਪ_subgroup, 2*rm_stdq, i+1}, 1, ਸੱਚ, use_subgroups, force_subgroup_ਮਾਪ);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q4_0][i], "mul_mat_vec_q4_0_f16_f32", arr_dmmv_q4_0_f16_f32_len[reduc], arr_dmmv_q4_0_f16_f32_ਅੰਕੜੇ[reduc], "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {2*rm_stdq, 1, 1}, {wg_ਮਾਪ_subgroup, 2*rm_stdq, i+1}, 1, ਸੱਚ, use_subgroups, force_subgroup_ਮਾਪ);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q4_1][i], "mul_mat_vec_q4_1_f16_f32", arr_dmmv_q4_1_f16_f32_len[reduc], arr_dmmv_q4_1_f16_f32_ਅੰਕੜੇ[reduc], "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {2*rm_stdq, 1, 1}, {wg_ਮਾਪ_subgroup, 2*rm_stdq, i+1}, 1, ਸੱਚ, use_subgroups, force_subgroup_ਮਾਪ);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q5_0][i], "mul_mat_vec_q5_0_f16_f32", arr_dmmv_q5_0_f16_f32_len[reduc], arr_dmmv_q5_0_f16_f32_ਅੰਕੜੇ[reduc], "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {2*rm_stdq, 1, 1}, {wg_ਮਾਪ_subgroup, 2*rm_stdq, i+1}, 1, ਸੱਚ, use_subgroups, force_subgroup_ਮਾਪ);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q5_1][i], "mul_mat_vec_q5_1_f16_f32", arr_dmmv_q5_1_f16_f32_len[reduc], arr_dmmv_q5_1_f16_f32_ਅੰਕੜੇ[reduc], "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {2*rm_stdq, 1, 1}, {wg_ਮਾਪ_subgroup, 2*rm_stdq, i+1}, 1, ਸੱਚ, use_subgroups, force_subgroup_ਮਾਪ);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q8_0][i], "mul_mat_vec_q8_0_f16_f32", arr_dmmv_q8_0_f16_f32_len[reduc], arr_dmmv_q8_0_f16_f32_ਅੰਕੜੇ[reduc], "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {1*rm_stdq, 1, 1}, {wg_ਮਾਪ_subgroup, 1*rm_stdq, i+1}, 1, ਸੱਚ, use_subgroups, force_subgroup_ਮਾਪ);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q2_K][i], "mul_mat_vec_q2_k_f16_f32", arr_dmmv_q2_k_f16_f32_len[reduc16], arr_dmmv_q2_k_f16_f32_ਅੰਕੜੇ[reduc16], "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {rm_kq, 1, 1}, {wg_ਮਾਪ_subgroup16, rm_kq, i+1}, 1, ਸੱਚ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q3_K][i], "mul_mat_vec_q3_k_f16_f32", arr_dmmv_q3_k_f16_f32_len[reduc16], arr_dmmv_q3_k_f16_f32_ਅੰਕੜੇ[reduc16], "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {rm_kq, 1, 1}, {wg_ਮਾਪ_subgroup16, rm_kq, i+1}, 1, ਸੱਚ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q4_K][i], "mul_mat_vec_q4_k_f16_f32", arr_dmmv_q4_k_f16_f32_len[reduc16], arr_dmmv_q4_k_f16_f32_ਅੰਕੜੇ[reduc16], "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {rm_kq, 1, 1}, {wg_ਮਾਪ_subgroup16, rm_kq, i+1}, 1, ਸੱਚ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q5_K][i], "mul_mat_vec_q5_k_f16_f32", arr_dmmv_q5_k_f16_f32_len[reduc16], arr_dmmv_q5_k_f16_f32_ਅੰਕੜੇ[reduc16], "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {rm_kq, 1, 1}, {wg_ਮਾਪ_subgroup16, rm_kq, i+1}, 1, ਸੱਚ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q6_K][i], "mul_mat_vec_q6_k_f16_f32", arr_dmmv_q6_k_f16_f32_len[reduc16], arr_dmmv_q6_k_f16_f32_ਅੰਕੜੇ[reduc16], "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {rm_kq, 1, 1}, {wg_ਮਾਪ_subgroup16, rm_kq, i+1}, 1, ਸੱਚ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_TQ1_0][i], "mul_mat_vec_tq1_0_f16_f32", arr_dmmv_tq1_0_f16_f32_len[reduc16], arr_dmmv_tq1_0_f16_f32_ਅੰਕੜੇ[reduc16], "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {rm_kq, 1, 1}, {wg_ਮਾਪ_subgroup16, rm_kq, i+1}, 1, ਸੱਚ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_TQ2_0][i], "mul_mat_vec_tq2_0_f16_f32", arr_dmmv_tq2_0_f16_f32_len[reduc16], arr_dmmv_tq2_0_f16_f32_ਅੰਕੜੇ[reduc16], "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {rm_kq, 1, 1}, {wg_ਮਾਪ_subgroup16, rm_kq, i+1}, 1, ਸੱਚ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ1_S][i],   "mul_mat_vec_iq1_s_f16_f32",   arr_dmmv_iq1_s_f16_f32_len[reduc16],   arr_dmmv_iq1_s_f16_f32_ਅੰਕੜੇ[reduc16],   "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {rm_iq, 1, 1}, {wg_ਮਾਪ_subgroup16, rm_iq, i+1}, 1, ਸੱਚ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ1_M][i],   "mul_mat_vec_iq1_m_f16_f32",   arr_dmmv_iq1_m_f16_f32_len[reduc16],   arr_dmmv_iq1_m_f16_f32_ਅੰਕੜੇ[reduc16],   "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {rm_iq, 1, 1}, {wg_ਮਾਪ_subgroup16, rm_iq, i+1}, 1, ਸੱਚ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ2_XXS][i], "mul_mat_vec_iq2_xxs_f16_f32", arr_dmmv_iq2_xxs_f16_f32_len[reduc16], arr_dmmv_iq2_xxs_f16_f32_ਅੰਕੜੇ[reduc16], "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {rm_iq, 1, 1}, {wg_ਮਾਪ_subgroup16, rm_iq, i+1}, 1, ਸੱਚ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ2_XS][i],  "mul_mat_vec_iq2_xs_f16_f32",  arr_dmmv_iq2_xs_f16_f32_len[reduc16],  arr_dmmv_iq2_xs_f16_f32_ਅੰਕੜੇ[reduc16],  "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {rm_iq, 1, 1}, {wg_ਮਾਪ_subgroup16, rm_iq, i+1}, 1, ਸੱਚ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ2_S][i],   "mul_mat_vec_iq2_s_f16_f32",   arr_dmmv_iq2_s_f16_f32_len[reduc16],   arr_dmmv_iq2_s_f16_f32_ਅੰਕੜੇ[reduc16],   "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {rm_iq, 1, 1}, {wg_ਮਾਪ_subgroup16, rm_iq, i+1}, 1, ਸੱਚ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ3_XXS][i], "mul_mat_vec_iq3_xxs_f16_f32", arr_dmmv_iq3_xxs_f16_f32_len[reduc16], arr_dmmv_iq3_xxs_f16_f32_ਅੰਕੜੇ[reduc16], "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {rm_iq, 1, 1}, {wg_ਮਾਪ_subgroup16, rm_iq, i+1}, 1, ਸੱਚ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ3_S][i],   "mul_mat_vec_iq3_s_f16_f32",   arr_dmmv_iq3_s_f16_f32_len[reduc16],   arr_dmmv_iq3_s_f16_f32_ਅੰਕੜੇ[reduc16],   "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {rm_iq, 1, 1}, {wg_ਮਾਪ_subgroup16, rm_iq, i+1}, 1, ਸੱਚ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ4_XS][i],  "mul_mat_vec_iq4_xs_f16_f32",  arr_dmmv_iq4_xs_f16_f32_len[reduc16],  arr_dmmv_iq4_xs_f16_f32_ਅੰਕੜੇ[reduc16],  "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {rm_iq, 1, 1}, {wg_ਮਾਪ_subgroup16, rm_iq, i+1}, 1, ਸੱਚ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ4_NL][i],  "mul_mat_vec_iq4_nl_f16_f32",  arr_dmmv_iq4_nl_f16_f32_len[reduc16],  arr_dmmv_iq4_nl_f16_f32_ਅੰਕੜੇ[reduc16],  "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {rm_iq, 1, 1}, {wg_ਮਾਪ_subgroup16, rm_iq, i+1}, 1, ਸੱਚ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_MXFP4][i],   "mul_mat_vec_mxfp4_f16_f32",   OCP_DMMV_LEN(arr_dmmv_mxfp4_f16_f32, reduc16), OCP_DMMV_ਅੰਕੜੇ(arr_dmmv_mxfp4_f16_f32, reduc16), "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {rm_iq, 1, 1}, {wg_ਮਾਪ_subgroup16, rm_iq, i+1}, 1, ਸੱਚ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_NVFP4][i],   "mul_mat_vec_nvfp4_f16_f32",   OCP_DMMV_LEN(arr_dmmv_nvfp4_f16_f32, reduc16), OCP_DMMV_ਅੰਕੜੇ(arr_dmmv_nvfp4_f16_f32, reduc16), "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {rm_iq, 1, 1}, {wg_ਮਾਪ_subgroup16, rm_iq, i+1}, 1, ਸੱਚ, use_subgroups16, force_subgroup_size16);

#ਜੇ defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
            ਜੇ (device->integer_dot_ਗੁਣਨਫਲ) {
                ਅਚਲ uint32_t subgroup_ਮਾਪ_int = (device->vendor_id == VK_VENDOR_ID_INTEL && device->subgroup_ਮਾਪ_control) ? device->subgroup_min_ਮਾਪ : device->subgroup_ਮਾਪ;
                ਅਚਲ uint32_t wg_ਮਾਪ_subgroup_int = (w == DMMV_WG_ਮਾਪ_SUBGROUP) ? subgroup_ਮਾਪ_int : (subgroup_ਮਾਪ_int * 4);

                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q2_0][i], "mul_mat_vec_q2_0_q8_1_f32", arr_dmmv_q2_0_q8_1_f32_len[reduc], arr_dmmv_q2_0_q8_1_f32_ਅੰਕੜੇ[reduc], "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {rm_int_n(2*rm_kq_int, i), 1, 1}, {wg_ਮਾਪ_subgroup_int, rm_int_n(2*rm_kq_int, i), i+1}, 1, ਸੱਚ, use_subgroups, subgroup_ਮਾਪ_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q4_0][i], "mul_mat_vec_q4_0_q8_1_f32", arr_dmmv_q4_0_q8_1_f32_len[reduc], arr_dmmv_q4_0_q8_1_f32_ਅੰਕੜੇ[reduc], "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {rm_int_n(1*rm_stdq_int, i), 1, 1}, {wg_ਮਾਪ_subgroup_int, rm_int_n(1*rm_stdq_int, i), i+1}, 1, ਸੱਚ, use_subgroups, subgroup_ਮਾਪ_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q4_1][i], "mul_mat_vec_q4_1_q8_1_f32", arr_dmmv_q4_1_q8_1_f32_len[reduc], arr_dmmv_q4_1_q8_1_f32_ਅੰਕੜੇ[reduc], "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {rm_int_n(1*rm_stdq_int, i), 1, 1}, {wg_ਮਾਪ_subgroup_int, rm_int_n(1*rm_stdq_int, i), i+1}, 1, ਸੱਚ, use_subgroups, subgroup_ਮਾਪ_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q5_0][i], "mul_mat_vec_q5_0_q8_1_f32", arr_dmmv_q5_0_q8_1_f32_len[reduc], arr_dmmv_q5_0_q8_1_f32_ਅੰਕੜੇ[reduc], "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {rm_int_n(1*rm_stdq_int, i), 1, 1}, {wg_ਮਾਪ_subgroup_int, rm_int_n(1*rm_stdq_int, i), i+1}, 1, ਸੱਚ, use_subgroups, subgroup_ਮਾਪ_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q5_1][i], "mul_mat_vec_q5_1_q8_1_f32", arr_dmmv_q5_1_q8_1_f32_len[reduc], arr_dmmv_q5_1_q8_1_f32_ਅੰਕੜੇ[reduc], "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {rm_int_n(1*rm_stdq_int, i), 1, 1}, {wg_ਮਾਪ_subgroup_int, rm_int_n(1*rm_stdq_int, i), i+1}, 1, ਸੱਚ, use_subgroups, subgroup_ਮਾਪ_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q8_0][i], "mul_mat_vec_q8_0_q8_1_f32", arr_dmmv_q8_0_q8_1_f32_len[reduc], arr_dmmv_q8_0_q8_1_f32_ਅੰਕੜੇ[reduc], "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {rm_int_n(1*rm_stdq_int, i), 1, 1}, {wg_ਮਾਪ_subgroup_int, rm_int_n(1*rm_stdq_int, i), i+1}, 1, ਸੱਚ, use_subgroups, subgroup_ਮਾਪ_int);

                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_MXFP4][i], "mul_mat_vec_mxfp4_q8_1_f32", arr_dmmv_mxfp4_q8_1_f32_len[reduc], arr_dmmv_mxfp4_q8_1_f32_ਅੰਕੜੇ[reduc], "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {rm_int_n(2*rm_stdq_int, i), 1, 1}, {wg_ਮਾਪ_subgroup_int, rm_int_n(2*rm_stdq_int, i), i+1}, 1, ਸੱਚ, use_subgroups, subgroup_ਮਾਪ_int);

                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q2_K][i], "mul_mat_vec_q2_k_q8_1_f32", arr_dmmv_q2_k_q8_1_f32_len[reduc], arr_dmmv_q2_k_q8_1_f32_ਅੰਕੜੇ[reduc], "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {rm_int_n(2*rm_kq_int, i), 1, 1}, {wg_ਮਾਪ_subgroup_int, rm_int_n(2*rm_kq_int, i), i+1}, 1, ਸੱਚ, use_subgroups, subgroup_ਮਾਪ_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q3_K][i], "mul_mat_vec_q3_k_q8_1_f32", arr_dmmv_q3_k_q8_1_f32_len[reduc], arr_dmmv_q3_k_q8_1_f32_ਅੰਕੜੇ[reduc], "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {rm_int_n(1*rm_kq_int, i), 1, 1}, {wg_ਮਾਪ_subgroup_int, rm_int_n(1*rm_kq_int, i), i+1}, 1, ਸੱਚ, use_subgroups, subgroup_ਮਾਪ_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q4_K][i], "mul_mat_vec_q4_k_q8_1_f32", arr_dmmv_q4_k_q8_1_f32_len[reduc], arr_dmmv_q4_k_q8_1_f32_ਅੰਕੜੇ[reduc], "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {rm_int_n(1*rm_kq_int, i), 1, 1}, {wg_ਮਾਪ_subgroup_int, rm_int_n(1*rm_kq_int, i), i+1}, 1, ਸੱਚ, use_subgroups, subgroup_ਮਾਪ_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q5_K][i], "mul_mat_vec_q5_k_q8_1_f32", arr_dmmv_q5_k_q8_1_f32_len[reduc], arr_dmmv_q5_k_q8_1_f32_ਅੰਕੜੇ[reduc], "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {rm_int_n(1*rm_kq_int, i), 1, 1}, {wg_ਮਾਪ_subgroup_int, rm_int_n(1*rm_kq_int, i), i+1}, 1, ਸੱਚ, use_subgroups, subgroup_ਮਾਪ_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q6_K][i], "mul_mat_vec_q6_k_q8_1_f32", arr_dmmv_q6_k_q8_1_f32_len[reduc], arr_dmmv_q6_k_q8_1_f32_ਅੰਕੜੇ[reduc], "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {rm_int_n(1*rm_kq_int, i), 1, 1}, {wg_ਮਾਪ_subgroup_int, rm_int_n(1*rm_kq_int, i), i+1}, 1, ਸੱਚ, use_subgroups, subgroup_ਮਾਪ_int);

                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_IQ1_S][i], "mul_mat_vec_iq1_s_q8_1_f32", arr_dmmv_iq1_s_q8_1_f32_len[reduc], arr_dmmv_iq1_s_q8_1_f32_ਅੰਕੜੇ[reduc], "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {1*rm_iq_int(i), 1, 1}, {wg_ਮਾਪ_subgroup_int, 1*rm_iq_int(i), i+1}, 1, ਸੱਚ, use_subgroups, subgroup_ਮਾਪ_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_IQ1_M][i], "mul_mat_vec_iq1_m_q8_1_f32", arr_dmmv_iq1_m_q8_1_f32_len[reduc], arr_dmmv_iq1_m_q8_1_f32_ਅੰਕੜੇ[reduc], "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {1*rm_iq_int(i), 1, 1}, {wg_ਮਾਪ_subgroup_int, 1*rm_iq_int(i), i+1}, 1, ਸੱਚ, use_subgroups, subgroup_ਮਾਪ_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_IQ4_XS][i], "mul_mat_vec_iq4_xs_q8_1_f32", arr_dmmv_iq4_xs_q8_1_f32_len[reduc], arr_dmmv_iq4_xs_q8_1_f32_ਅੰਕੜੇ[reduc], "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_ਧੱਕੋ_constants), {rm_int_n(1*rm_stdq_int, i), 1, 1}, {wg_ਮਾਪ_subgroup_int, rm_int_n(1*rm_stdq_int, i), i+1}, 1, ਸੱਚ, use_subgroups, subgroup_ਮਾਪ_int);

            }
#ਜੇ_ਅੰਤ // GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT
        }

        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_F32 ], "mul_mat_vec_id_f32_f32",        arr_dmmv_id_f32_f32_f32_len[reduc],     arr_dmmv_id_f32_f32_f32_ਅੰਕੜੇ[reduc],     "main", mul_mat_vec_id_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_id_ਧੱਕੋ_constants), {1, 1, 1}, {wg_ਮਾਪ_subgroup, 1}, 1, ਝੂਠ, use_subgroups, force_subgroup_ਮਾਪ);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_F16 ], "mul_mat_vec_id_f16_f32",        arr_dmmv_id_f16_f32_f32_len[reduc],     arr_dmmv_id_f16_f32_f32_ਅੰਕੜੇ[reduc],     "main", mul_mat_vec_id_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_id_ਧੱਕੋ_constants), {2, 1, 1}, {wg_ਮਾਪ_subgroup, 2}, 1, ਝੂਠ, use_subgroups, force_subgroup_ਮਾਪ);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_BF16], "mul_mat_vec_id_bf16_f32",       arr_dmmv_id_bf16_f32_f32_len[reduc],    arr_dmmv_id_bf16_f32_f32_ਅੰਕੜੇ[reduc],    "main", mul_mat_vec_id_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_id_ਧੱਕੋ_constants), {2, 1, 1}, {wg_ਮਾਪ_subgroup, 2}, 1, ਝੂਠ, use_subgroups, force_subgroup_ਮਾਪ);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q1_0], "mul_mat_vec_id_q1_0_f32",       arr_dmmv_id_q1_0_f32_f32_len[reduc],    arr_dmmv_id_q1_0_f32_f32_ਅੰਕੜੇ[reduc],    "main", mul_mat_vec_id_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_id_ਧੱਕੋ_constants), {2*rm_stdq, 1, 1}, {wg_ਮਾਪ_subgroup, 2*rm_stdq}, 1, ਸੱਚ, use_subgroups, force_subgroup_ਮਾਪ);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q2_0], "mul_mat_vec_id_q2_0_f32",       arr_dmmv_id_q2_0_f32_f32_len[reduc],    arr_dmmv_id_q2_0_f32_f32_ਅੰਕੜੇ[reduc],    "main", mul_mat_vec_id_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_id_ਧੱਕੋ_constants), {2*rm_stdq, 1, 1}, {wg_ਮਾਪ_subgroup, 2*rm_stdq}, 1, ਸੱਚ, use_subgroups, force_subgroup_ਮਾਪ);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q4_0], "mul_mat_vec_id_q4_0_f32",       arr_dmmv_id_q4_0_f32_f32_len[reduc],    arr_dmmv_id_q4_0_f32_f32_ਅੰਕੜੇ[reduc],    "main", mul_mat_vec_id_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_id_ਧੱਕੋ_constants), {2*rm_stdq, 1, 1}, {wg_ਮਾਪ_subgroup, 2*rm_stdq}, 1, ਸੱਚ, use_subgroups, force_subgroup_ਮਾਪ);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q4_1], "mul_mat_vec_id_q4_1_f32",       arr_dmmv_id_q4_1_f32_f32_len[reduc],    arr_dmmv_id_q4_1_f32_f32_ਅੰਕੜੇ[reduc],    "main", mul_mat_vec_id_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_id_ਧੱਕੋ_constants), {2*rm_stdq, 1, 1}, {wg_ਮਾਪ_subgroup, 2*rm_stdq}, 1, ਸੱਚ, use_subgroups, force_subgroup_ਮਾਪ);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q5_0], "mul_mat_vec_id_q5_0_f32",       arr_dmmv_id_q5_0_f32_f32_len[reduc],    arr_dmmv_id_q5_0_f32_f32_ਅੰਕੜੇ[reduc],    "main", mul_mat_vec_id_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_id_ਧੱਕੋ_constants), {2*rm_stdq, 1, 1}, {wg_ਮਾਪ_subgroup, 2*rm_stdq}, 1, ਸੱਚ, use_subgroups, force_subgroup_ਮਾਪ);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q5_1], "mul_mat_vec_id_q5_1_f32",       arr_dmmv_id_q5_1_f32_f32_len[reduc],    arr_dmmv_id_q5_1_f32_f32_ਅੰਕੜੇ[reduc],    "main", mul_mat_vec_id_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_id_ਧੱਕੋ_constants), {2*rm_stdq, 1, 1}, {wg_ਮਾਪ_subgroup, 2*rm_stdq}, 1, ਸੱਚ, use_subgroups, force_subgroup_ਮਾਪ);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q8_0], "mul_mat_vec_id_q8_0_f32",       arr_dmmv_id_q8_0_f32_f32_len[reduc],    arr_dmmv_id_q8_0_f32_f32_ਅੰਕੜੇ[reduc],    "main", mul_mat_vec_id_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_id_ਧੱਕੋ_constants), {1*rm_stdq, 1, 1}, {wg_ਮਾਪ_subgroup, 1*rm_stdq}, 1, ਸੱਚ, use_subgroups, force_subgroup_ਮਾਪ);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q2_K], "mul_mat_vec_id_q2_k_f32",       arr_dmmv_id_q2_k_f32_f32_len[reduc16],    arr_dmmv_id_q2_k_f32_f32_ਅੰਕੜੇ[reduc16],    "main", mul_mat_vec_id_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_id_ਧੱਕੋ_constants), {rm_kq, 1, 1}, {wg_ਮਾਪ_subgroup16, rm_kq}, 1, ਸੱਚ, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q3_K], "mul_mat_vec_id_q3_k_f32",       arr_dmmv_id_q3_k_f32_f32_len[reduc16],    arr_dmmv_id_q3_k_f32_f32_ਅੰਕੜੇ[reduc16],    "main", mul_mat_vec_id_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_id_ਧੱਕੋ_constants), {rm_kq, 1, 1}, {wg_ਮਾਪ_subgroup16, rm_kq}, 1, ਸੱਚ, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q4_K], "mul_mat_vec_id_q4_k_f32",       arr_dmmv_id_q4_k_f32_f32_len[reduc16],    arr_dmmv_id_q4_k_f32_f32_ਅੰਕੜੇ[reduc16],    "main", mul_mat_vec_id_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_id_ਧੱਕੋ_constants), {rm_kq, 1, 1}, {wg_ਮਾਪ_subgroup16, rm_kq}, 1, ਸੱਚ, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q5_K], "mul_mat_vec_id_q5_k_f32",       arr_dmmv_id_q5_k_f32_f32_len[reduc16],    arr_dmmv_id_q5_k_f32_f32_ਅੰਕੜੇ[reduc16],    "main", mul_mat_vec_id_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_id_ਧੱਕੋ_constants), {rm_kq, 1, 1}, {wg_ਮਾਪ_subgroup16, rm_kq}, 1, ਸੱਚ, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q6_K], "mul_mat_vec_id_q6_k_f32",       arr_dmmv_id_q6_k_f32_f32_len[reduc16],    arr_dmmv_id_q6_k_f32_f32_ਅੰਕੜੇ[reduc16],    "main", mul_mat_vec_id_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_id_ਧੱਕੋ_constants), {rm_kq, 1, 1}, {wg_ਮਾਪ_subgroup16, rm_kq}, 1, ਸੱਚ, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_TQ1_0], "mul_mat_vec_id_tq1_0_f32",     arr_dmmv_id_tq1_0_f32_f32_len[reduc16],   arr_dmmv_id_tq1_0_f32_f32_ਅੰਕੜੇ[reduc16],   "main", mul_mat_vec_id_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_id_ਧੱਕੋ_constants), {rm_kq, 1, 1}, {wg_ਮਾਪ_subgroup16, rm_kq}, 1, ਸੱਚ, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_TQ2_0], "mul_mat_vec_id_tq2_0_f32",     arr_dmmv_id_tq2_0_f32_f32_len[reduc16],   arr_dmmv_id_tq2_0_f32_f32_ਅੰਕੜੇ[reduc16],   "main", mul_mat_vec_id_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_id_ਧੱਕੋ_constants), {rm_kq, 1, 1}, {wg_ਮਾਪ_subgroup16, rm_kq}, 1, ਸੱਚ, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ1_S],   "mul_mat_vec_id_iq1_s_f32",   arr_dmmv_id_iq1_s_f32_f32_len[reduc16],   arr_dmmv_id_iq1_s_f32_f32_ਅੰਕੜੇ[reduc16],   "main", mul_mat_vec_id_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_id_ਧੱਕੋ_constants), {rm_iq, 1, 1}, {wg_ਮਾਪ_subgroup16, rm_iq}, 1, ਸੱਚ, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ1_M],   "mul_mat_vec_id_iq1_m_f32",   arr_dmmv_id_iq1_m_f32_f32_len[reduc16],   arr_dmmv_id_iq1_m_f32_f32_ਅੰਕੜੇ[reduc16],   "main", mul_mat_vec_id_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_id_ਧੱਕੋ_constants), {rm_iq, 1, 1}, {wg_ਮਾਪ_subgroup16, rm_iq}, 1, ਸੱਚ, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ2_XXS], "mul_mat_vec_id_iq2_xxs_f32", arr_dmmv_id_iq2_xxs_f32_f32_len[reduc16], arr_dmmv_id_iq2_xxs_f32_f32_ਅੰਕੜੇ[reduc16], "main", mul_mat_vec_id_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_id_ਧੱਕੋ_constants), {rm_iq, 1, 1}, {wg_ਮਾਪ_subgroup16, rm_iq}, 1, ਸੱਚ, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ2_XS],  "mul_mat_vec_id_iq2_xs_f32",  arr_dmmv_id_iq2_xs_f32_f32_len[reduc16],  arr_dmmv_id_iq2_xs_f32_f32_ਅੰਕੜੇ[reduc16],  "main", mul_mat_vec_id_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_id_ਧੱਕੋ_constants), {rm_iq, 1, 1}, {wg_ਮਾਪ_subgroup16, rm_iq}, 1, ਸੱਚ, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ2_S],   "mul_mat_vec_id_iq2_s_f32",   arr_dmmv_id_iq2_s_f32_f32_len[reduc16],   arr_dmmv_id_iq2_s_f32_f32_ਅੰਕੜੇ[reduc16],   "main", mul_mat_vec_id_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_id_ਧੱਕੋ_constants), {rm_iq, 1, 1}, {wg_ਮਾਪ_subgroup16, rm_iq}, 1, ਸੱਚ, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ3_XXS], "mul_mat_vec_id_iq3_xxs_f32", arr_dmmv_id_iq3_xxs_f32_f32_len[reduc16], arr_dmmv_id_iq3_xxs_f32_f32_ਅੰਕੜੇ[reduc16], "main", mul_mat_vec_id_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_id_ਧੱਕੋ_constants), {rm_iq, 1, 1}, {wg_ਮਾਪ_subgroup16, rm_iq}, 1, ਸੱਚ, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ3_S],   "mul_mat_vec_id_iq3_s_f32",   arr_dmmv_id_iq3_s_f32_f32_len[reduc16],   arr_dmmv_id_iq3_s_f32_f32_ਅੰਕੜੇ[reduc16],   "main", mul_mat_vec_id_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_id_ਧੱਕੋ_constants), {rm_iq, 1, 1}, {wg_ਮਾਪ_subgroup16, rm_iq}, 1, ਸੱਚ, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ4_XS],  "mul_mat_vec_id_iq4_xs_f32",  arr_dmmv_id_iq4_xs_f32_f32_len[reduc16],  arr_dmmv_id_iq4_xs_f32_f32_ਅੰਕੜੇ[reduc16],  "main", mul_mat_vec_id_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_id_ਧੱਕੋ_constants), {rm_iq, 1, 1}, {wg_ਮਾਪ_subgroup16, rm_iq}, 1, ਸੱਚ, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ4_NL],  "mul_mat_vec_id_iq4_nl_f32",  arr_dmmv_id_iq4_nl_f32_f32_len[reduc16],  arr_dmmv_id_iq4_nl_f32_f32_ਅੰਕੜੇ[reduc16],  "main", mul_mat_vec_id_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_id_ਧੱਕੋ_constants), {rm_iq, 1, 1}, {wg_ਮਾਪ_subgroup16, rm_iq}, 1, ਸੱਚ, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_MXFP4],   "mul_mat_vec_id_mxfp4_f32",   OCP_DMMV_LEN(arr_dmmv_id_mxfp4_f32_f32, reduc16), OCP_DMMV_ਅੰਕੜੇ(arr_dmmv_id_mxfp4_f32_f32, reduc16), "main", mul_mat_vec_id_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_id_ਧੱਕੋ_constants), {rm_iq, 1, 1}, {wg_ਮਾਪ_subgroup16, rm_iq}, 1, ਸੱਚ, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_NVFP4],   "mul_mat_vec_id_nvfp4_f32",   OCP_DMMV_LEN(arr_dmmv_id_nvfp4_f32_f32, reduc16), OCP_DMMV_ਅੰਕੜੇ(arr_dmmv_id_nvfp4_f32_f32, reduc16), "main", mul_mat_vec_id_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_id_ਧੱਕੋ_constants), {rm_iq, 1, 1}, {wg_ਮਾਪ_subgroup16, rm_iq}, 1, ਸੱਚ, use_subgroups16, force_subgroup_size16);

#ਜੇ defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
        ਜੇ (device->integer_dot_ਗੁਣਨਫਲ) {
            ਅਚਲ uint32_t subgroup_ਮਾਪ_int = (device->vendor_id == VK_VENDOR_ID_INTEL && device->subgroup_ਮਾਪ_control) ? device->subgroup_min_ਮਾਪ : device->subgroup_ਮਾਪ;
            ਅਚਲ uint32_t wg_ਮਾਪ_subgroup_int = (w == DMMV_WG_ਮਾਪ_SUBGROUP) ? subgroup_ਮਾਪ_int : (subgroup_ਮਾਪ_int * 4);

            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q2_0], "mul_mat_vec_id_q2_0_q8_1_f32", arr_dmmv_id_q2_0_q8_1_f32_len[reduc], arr_dmmv_id_q2_0_q8_1_f32_ਅੰਕੜੇ[reduc], "main", mul_mat_vec_id_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_id_ਧੱਕੋ_constants), {rm_id(2*rm_kq_int), 1, 1}, {wg_ਮਾਪ_subgroup_int, rm_id(2*rm_kq_int)}, 1, ਸੱਚ, use_subgroups, subgroup_ਮਾਪ_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q4_0], "mul_mat_vec_id_q4_0_q8_1_f32", arr_dmmv_id_q4_0_q8_1_f32_len[reduc], arr_dmmv_id_q4_0_q8_1_f32_ਅੰਕੜੇ[reduc], "main", mul_mat_vec_id_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_id_ਧੱਕੋ_constants), {rm_id(1*rm_stdq_int), 1, 1}, {wg_ਮਾਪ_subgroup_int, rm_id(1*rm_stdq_int)}, 1, ਸੱਚ, use_subgroups, subgroup_ਮਾਪ_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q4_1], "mul_mat_vec_id_q4_1_q8_1_f32", arr_dmmv_id_q4_1_q8_1_f32_len[reduc], arr_dmmv_id_q4_1_q8_1_f32_ਅੰਕੜੇ[reduc], "main", mul_mat_vec_id_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_id_ਧੱਕੋ_constants), {rm_id(1*rm_stdq_int), 1, 1}, {wg_ਮਾਪ_subgroup_int, rm_id(1*rm_stdq_int)}, 1, ਸੱਚ, use_subgroups, subgroup_ਮਾਪ_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q5_0], "mul_mat_vec_id_q5_0_q8_1_f32", arr_dmmv_id_q5_0_q8_1_f32_len[reduc], arr_dmmv_id_q5_0_q8_1_f32_ਅੰਕੜੇ[reduc], "main", mul_mat_vec_id_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_id_ਧੱਕੋ_constants), {rm_id(1*rm_stdq_int), 1, 1}, {wg_ਮਾਪ_subgroup_int, rm_id(1*rm_stdq_int)}, 1, ਸੱਚ, use_subgroups, subgroup_ਮਾਪ_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q5_1], "mul_mat_vec_id_q5_1_q8_1_f32", arr_dmmv_id_q5_1_q8_1_f32_len[reduc], arr_dmmv_id_q5_1_q8_1_f32_ਅੰਕੜੇ[reduc], "main", mul_mat_vec_id_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_id_ਧੱਕੋ_constants), {rm_id(1*rm_stdq_int), 1, 1}, {wg_ਮਾਪ_subgroup_int, rm_id(1*rm_stdq_int)}, 1, ਸੱਚ, use_subgroups, subgroup_ਮਾਪ_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q8_0], "mul_mat_vec_id_q8_0_q8_1_f32", arr_dmmv_id_q8_0_q8_1_f32_len[reduc], arr_dmmv_id_q8_0_q8_1_f32_ਅੰਕੜੇ[reduc], "main", mul_mat_vec_id_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_id_ਧੱਕੋ_constants), {rm_id(1*rm_stdq_int), 1, 1}, {wg_ਮਾਪ_subgroup_int, rm_id(1*rm_stdq_int)}, 1, ਸੱਚ, use_subgroups, subgroup_ਮਾਪ_int);

            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_MXFP4], "mul_mat_vec_id_mxfp4_q8_1_f32", arr_dmmv_id_mxfp4_q8_1_f32_len[reduc], arr_dmmv_id_mxfp4_q8_1_f32_ਅੰਕੜੇ[reduc], "main", mul_mat_vec_id_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_id_ਧੱਕੋ_constants), {rm_id(2*rm_stdq_int), 1, 1}, {wg_ਮਾਪ_subgroup_int, rm_id(2*rm_stdq_int)}, 1, ਸੱਚ, use_subgroups, subgroup_ਮਾਪ_int);

            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q2_K], "mul_mat_vec_id_q2_k_q8_1_f32", arr_dmmv_id_q2_k_q8_1_f32_len[reduc], arr_dmmv_id_q2_k_q8_1_f32_ਅੰਕੜੇ[reduc], "main", mul_mat_vec_id_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_id_ਧੱਕੋ_constants), {rm_id(2*rm_kq_int), 1, 1}, {wg_ਮਾਪ_subgroup_int, rm_id(2*rm_kq_int)}, 1, ਸੱਚ, use_subgroups, subgroup_ਮਾਪ_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q3_K], "mul_mat_vec_id_q3_k_q8_1_f32", arr_dmmv_id_q3_k_q8_1_f32_len[reduc], arr_dmmv_id_q3_k_q8_1_f32_ਅੰਕੜੇ[reduc], "main", mul_mat_vec_id_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_id_ਧੱਕੋ_constants), {rm_id(1*rm_kq_int), 1, 1}, {wg_ਮਾਪ_subgroup_int, rm_id(1*rm_kq_int)}, 1, ਸੱਚ, use_subgroups, subgroup_ਮਾਪ_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q4_K], "mul_mat_vec_id_q4_k_q8_1_f32", arr_dmmv_id_q4_k_q8_1_f32_len[reduc], arr_dmmv_id_q4_k_q8_1_f32_ਅੰਕੜੇ[reduc], "main", mul_mat_vec_id_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_id_ਧੱਕੋ_constants), {rm_id(1*rm_kq_int), 1, 1}, {wg_ਮਾਪ_subgroup_int, rm_id(1*rm_kq_int)}, 1, ਸੱਚ, use_subgroups, subgroup_ਮਾਪ_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q5_K], "mul_mat_vec_id_q5_k_q8_1_f32", arr_dmmv_id_q5_k_q8_1_f32_len[reduc], arr_dmmv_id_q5_k_q8_1_f32_ਅੰਕੜੇ[reduc], "main", mul_mat_vec_id_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_id_ਧੱਕੋ_constants), {rm_id(1*rm_kq_int), 1, 1}, {wg_ਮਾਪ_subgroup_int, rm_id(1*rm_kq_int)}, 1, ਸੱਚ, use_subgroups, subgroup_ਮਾਪ_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q6_K], "mul_mat_vec_id_q6_k_q8_1_f32", arr_dmmv_id_q6_k_q8_1_f32_len[reduc], arr_dmmv_id_q6_k_q8_1_f32_ਅੰਕੜੇ[reduc], "main", mul_mat_vec_id_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_id_ਧੱਕੋ_constants), {rm_id(1*rm_kq_int), 1, 1}, {wg_ਮਾਪ_subgroup_int, rm_id(1*rm_kq_int)}, 1, ਸੱਚ, use_subgroups, subgroup_ਮਾਪ_int);

            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_IQ1_S], "mul_mat_vec_id_iq1_s_q8_1_f32", arr_dmmv_id_iq1_s_q8_1_f32_len[reduc], arr_dmmv_id_iq1_s_q8_1_f32_ਅੰਕੜੇ[reduc], "main", mul_mat_vec_id_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_id_ਧੱਕੋ_constants), {1*rm_iq_int(0), 1, 1}, {wg_ਮਾਪ_subgroup_int, 1*rm_iq_int(0)}, 1, ਸੱਚ, use_subgroups, subgroup_ਮਾਪ_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_IQ1_M], "mul_mat_vec_id_iq1_m_q8_1_f32", arr_dmmv_id_iq1_m_q8_1_f32_len[reduc], arr_dmmv_id_iq1_m_q8_1_f32_ਅੰਕੜੇ[reduc], "main", mul_mat_vec_id_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_id_ਧੱਕੋ_constants), {1*rm_iq_int(0), 1, 1}, {wg_ਮਾਪ_subgroup_int, 1*rm_iq_int(0)}, 1, ਸੱਚ, use_subgroups, subgroup_ਮਾਪ_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_IQ4_XS], "mul_mat_vec_id_iq4_xs_q8_1_f32", arr_dmmv_id_iq4_xs_q8_1_f32_len[reduc], arr_dmmv_id_iq4_xs_q8_1_f32_ਅੰਕੜੇ[reduc], "main", mul_mat_vec_id_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_id_ਧੱਕੋ_constants), {rm_id(1*rm_stdq_int), 1, 1}, {wg_ਮਾਪ_subgroup_int, rm_id(1*rm_stdq_int)}, 1, ਸੱਚ, use_subgroups, subgroup_ਮਾਪ_int);
        }
#ਜੇ_ਅੰਤ // GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT
    }

#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ OCP_DMMV_ਅੰਕੜੇ
#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ OCP_DMMV_LEN

#ਜੇ !defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
    GGML_ਅਣਵਰਤਿਆ(rm_stdq_int);
    GGML_ਅਣਵਰਤਿਆ(rm_kq_int);
    GGML_ਅਣਵਰਤਿਆ(is_rdna3_or_4);
    GGML_ਅਣਵਰਤਿਆ(rm_int_n);
    GGML_ਅਣਵਰਤਿਆ(rm_id);
    GGML_ਅਣਵਰਤਿਆ(rm_iq_int);
#ਜੇ_ਅੰਤ

    // dequant shaders
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_F32 ], "f32_to_f16",   dequant_f32_len,  dequant_f32_ਅੰਕੜੇ,  "main", 2, 5 * ਆਕਾਰ(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q1_0], "dequant_q1_0", dequant_q1_0_len, dequant_q1_0_ਅੰਕੜੇ, "main", 2, 5 * ਆਕਾਰ(uint32_t), {256 * 8, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q2_0], "dequant_q2_0", dequant_q2_0_len, dequant_q2_0_ਅੰਕੜੇ, "main", 2, 5 * ਆਕਾਰ(uint32_t), {256 * 4, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q4_0], "dequant_q4_0", dequant_q4_0_len, dequant_q4_0_ਅੰਕੜੇ, "main", 2, 5 * ਆਕਾਰ(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q4_1], "dequant_q4_1", dequant_q4_1_len, dequant_q4_1_ਅੰਕੜੇ, "main", 2, 5 * ਆਕਾਰ(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q5_0], "dequant_q5_0", dequant_q5_0_len, dequant_q5_0_ਅੰਕੜੇ, "main", 2, 5 * ਆਕਾਰ(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q5_1], "dequant_q5_1", dequant_q5_1_len, dequant_q5_1_ਅੰਕੜੇ, "main", 2, 5 * ਆਕਾਰ(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q8_0], "dequant_q8_0", dequant_q8_0_len, dequant_q8_0_ਅੰਕੜੇ, "main", 2, 5 * ਆਕਾਰ(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant_transpose[GGML_TYPE_Q8_0], "dequant_q8_0_transpose", dequant_q8_0_transpose_len, dequant_q8_0_transpose_ਅੰਕੜੇ, "main", 2, 5 * ਆਕਾਰ(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q2_K], "dequant_q2_k", dequant_q2_k_len, dequant_q2_k_ਅੰਕੜੇ, "main", 2, 5 * ਆਕਾਰ(uint32_t), {256 * 64, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q3_K], "dequant_q3_k", dequant_q3_k_len, dequant_q3_k_ਅੰਕੜੇ, "main", 2, 5 * ਆਕਾਰ(uint32_t), {256 * 64, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q4_K], "dequant_q4_k", dequant_q4_k_len, dequant_q4_k_ਅੰਕੜੇ, "main", 2, 5 * ਆਕਾਰ(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q5_K], "dequant_q5_k", dequant_q5_k_len, dequant_q5_k_ਅੰਕੜੇ, "main", 2, 5 * ਆਕਾਰ(uint32_t), {256 * 64, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q6_K], "dequant_q6_k", dequant_q6_k_len, dequant_q6_k_ਅੰਕੜੇ, "main", 2, 5 * ਆਕਾਰ(uint32_t), {256 * 64, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_TQ1_0], "dequant_tq1_0", dequant_tq1_0_len, dequant_tq1_0_ਅੰਕੜੇ, "main", 2, 5 * ਆਕਾਰ(uint32_t), {256 * 4, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_TQ2_0], "dequant_tq2_0", dequant_tq2_0_len, dequant_tq2_0_ਅੰਕੜੇ, "main", 2, 5 * ਆਕਾਰ(uint32_t), {256 * 64, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ1_S],   "dequant_iq1_s",   dequant_iq1_s_len,   dequant_iq1_s_ਅੰਕੜੇ,   "main", 2, 5 * ਆਕਾਰ(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ1_M],   "dequant_iq1_m",   dequant_iq1_m_len,   dequant_iq1_m_ਅੰਕੜੇ,   "main", 2, 5 * ਆਕਾਰ(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ2_XXS], "dequant_iq2_xxs", dequant_iq2_xxs_len, dequant_iq2_xxs_ਅੰਕੜੇ, "main", 2, 5 * ਆਕਾਰ(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ2_XS],  "dequant_iq2_xs",  dequant_iq2_xs_len,  dequant_iq2_xs_ਅੰਕੜੇ,  "main", 2, 5 * ਆਕਾਰ(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ2_S],   "dequant_iq2_s",   dequant_iq2_s_len,   dequant_iq2_s_ਅੰਕੜੇ,   "main", 2, 5 * ਆਕਾਰ(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ3_XXS], "dequant_iq3_xxs", dequant_iq3_xxs_len, dequant_iq3_xxs_ਅੰਕੜੇ, "main", 2, 5 * ਆਕਾਰ(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ3_S],   "dequant_iq3_s",   dequant_iq3_s_len,   dequant_iq3_s_ਅੰਕੜੇ,   "main", 2, 5 * ਆਕਾਰ(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ4_XS],  "dequant_iq4_xs",  dequant_iq4_xs_len,  dequant_iq4_xs_ਅੰਕੜੇ,  "main", 2, 5 * ਆਕਾਰ(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ4_NL],  "dequant_iq4_nl",  dequant_iq4_nl_len,  dequant_iq4_nl_ਅੰਕੜੇ,  "main", 2, 5 * ਆਕਾਰ(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_MXFP4],   "dequant_mxfp4",   dequant_mxfp4_len,   dequant_mxfp4_ਅੰਕੜੇ,   "main", 2, 5 * ਆਕਾਰ(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_NVFP4],   "dequant_nvfp4",   dequant_nvfp4_len,   dequant_nvfp4_ਅੰਕੜੇ,   "main", 2, 5 * ਆਕਾਰ(uint32_t), {256 * 16, 1, 1}, {}, 1);

    // get_rows
    ggml_vk_create_pipeline(device, device->pipeline_get_ਪੰਕਤੀਆਂ[GGML_TYPE_F32 ], "get_rows_f32",  get_ਪੰਕਤੀਆਂ_f32_len,  get_ਪੰਕਤੀਆਂ_f32_ਅੰਕੜੇ,  "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ਪੰਕਤੀਆਂ[GGML_TYPE_F16 ], "get_rows_f16",  get_ਪੰਕਤੀਆਂ_f16_len,  get_ਪੰਕਤੀਆਂ_f16_ਅੰਕੜੇ,  "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ਪੰਕਤੀਆਂ[GGML_TYPE_BF16], "get_rows_bf16", get_ਪੰਕਤੀਆਂ_bf16_len, get_ਪੰਕਤੀਆਂ_bf16_ਅੰਕੜੇ, "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ਪੰਕਤੀਆਂ[GGML_TYPE_Q1_0], "get_rows_q1_0", get_ਪੰਕਤੀਆਂ_q1_0_len, get_ਪੰਕਤੀਆਂ_q1_0_ਅੰਕੜੇ, "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ਪੰਕਤੀਆਂ[GGML_TYPE_Q2_0], "get_rows_q2_0", get_ਪੰਕਤੀਆਂ_q2_0_len, get_ਪੰਕਤੀਆਂ_q2_0_ਅੰਕੜੇ, "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ਪੰਕਤੀਆਂ[GGML_TYPE_Q4_0], "get_rows_q4_0", get_ਪੰਕਤੀਆਂ_q4_0_len, get_ਪੰਕਤੀਆਂ_q4_0_ਅੰਕੜੇ, "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ਪੰਕਤੀਆਂ[GGML_TYPE_Q4_1], "get_rows_q4_1", get_ਪੰਕਤੀਆਂ_q4_1_len, get_ਪੰਕਤੀਆਂ_q4_1_ਅੰਕੜੇ, "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ਪੰਕਤੀਆਂ[GGML_TYPE_Q5_0], "get_rows_q5_0", get_ਪੰਕਤੀਆਂ_q5_0_len, get_ਪੰਕਤੀਆਂ_q5_0_ਅੰਕੜੇ, "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ਪੰਕਤੀਆਂ[GGML_TYPE_Q5_1], "get_rows_q5_1", get_ਪੰਕਤੀਆਂ_q5_1_len, get_ਪੰਕਤੀਆਂ_q5_1_ਅੰਕੜੇ, "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ਪੰਕਤੀਆਂ[GGML_TYPE_Q8_0], "get_rows_q8_0", get_ਪੰਕਤੀਆਂ_q8_0_len, get_ਪੰਕਤੀਆਂ_q8_0_ਅੰਕੜੇ, "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ਪੰਕਤੀਆਂ[GGML_TYPE_Q2_K], "get_rows_q2_k", get_ਪੰਕਤੀਆਂ_q2_k_len, get_ਪੰਕਤੀਆਂ_q2_k_ਅੰਕੜੇ, "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ਪੰਕਤੀਆਂ[GGML_TYPE_Q3_K], "get_rows_q3_k", get_ਪੰਕਤੀਆਂ_q3_k_len, get_ਪੰਕਤੀਆਂ_q3_k_ਅੰਕੜੇ, "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ਪੰਕਤੀਆਂ[GGML_TYPE_Q4_K], "get_rows_q4_k", get_ਪੰਕਤੀਆਂ_q4_k_len, get_ਪੰਕਤੀਆਂ_q4_k_ਅੰਕੜੇ, "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ਪੰਕਤੀਆਂ[GGML_TYPE_Q5_K], "get_rows_q5_k", get_ਪੰਕਤੀਆਂ_q5_k_len, get_ਪੰਕਤੀਆਂ_q5_k_ਅੰਕੜੇ, "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ਪੰਕਤੀਆਂ[GGML_TYPE_Q6_K], "get_rows_q6_k", get_ਪੰਕਤੀਆਂ_q6_k_len, get_ਪੰਕਤੀਆਂ_q6_k_ਅੰਕੜੇ, "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ਪੰਕਤੀਆਂ[GGML_TYPE_TQ1_0], "get_rows_tq1_0", get_ਪੰਕਤੀਆਂ_tq1_0_len, get_ਪੰਕਤੀਆਂ_tq1_0_ਅੰਕੜੇ, "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ਪੰਕਤੀਆਂ[GGML_TYPE_TQ2_0], "get_rows_tq2_0", get_ਪੰਕਤੀਆਂ_tq2_0_len, get_ਪੰਕਤੀਆਂ_tq2_0_ਅੰਕੜੇ, "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ਪੰਕਤੀਆਂ[GGML_TYPE_IQ1_S],   "get_rows_iq1_s",   get_ਪੰਕਤੀਆਂ_iq1_s_len,   get_ਪੰਕਤੀਆਂ_iq1_s_ਅੰਕੜੇ,   "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ਪੰਕਤੀਆਂ[GGML_TYPE_IQ1_M],   "get_rows_iq1_m",   get_ਪੰਕਤੀਆਂ_iq1_m_len,   get_ਪੰਕਤੀਆਂ_iq1_m_ਅੰਕੜੇ,   "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ਪੰਕਤੀਆਂ[GGML_TYPE_IQ2_XXS], "get_rows_iq2_xxs", get_ਪੰਕਤੀਆਂ_iq2_xxs_len, get_ਪੰਕਤੀਆਂ_iq2_xxs_ਅੰਕੜੇ, "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ਪੰਕਤੀਆਂ[GGML_TYPE_IQ2_XS],  "get_rows_iq2_xs",  get_ਪੰਕਤੀਆਂ_iq2_xs_len,  get_ਪੰਕਤੀਆਂ_iq2_xs_ਅੰਕੜੇ,  "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ਪੰਕਤੀਆਂ[GGML_TYPE_IQ2_S],   "get_rows_iq2_s",   get_ਪੰਕਤੀਆਂ_iq2_s_len,   get_ਪੰਕਤੀਆਂ_iq2_s_ਅੰਕੜੇ,   "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ਪੰਕਤੀਆਂ[GGML_TYPE_IQ3_XXS], "get_rows_iq3_xxs", get_ਪੰਕਤੀਆਂ_iq3_xxs_len, get_ਪੰਕਤੀਆਂ_iq3_xxs_ਅੰਕੜੇ, "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ਪੰਕਤੀਆਂ[GGML_TYPE_IQ3_S],   "get_rows_iq3_s",   get_ਪੰਕਤੀਆਂ_iq3_s_len,   get_ਪੰਕਤੀਆਂ_iq3_s_ਅੰਕੜੇ,   "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ਪੰਕਤੀਆਂ[GGML_TYPE_IQ4_XS],  "get_rows_iq4_xs",  get_ਪੰਕਤੀਆਂ_iq4_xs_len,  get_ਪੰਕਤੀਆਂ_iq4_xs_ਅੰਕੜੇ,  "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ਪੰਕਤੀਆਂ[GGML_TYPE_IQ4_NL],  "get_rows_iq4_nl",  get_ਪੰਕਤੀਆਂ_iq4_nl_len,  get_ਪੰਕਤੀਆਂ_iq4_nl_ਅੰਕੜੇ,  "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ਪੰਕਤੀਆਂ[GGML_TYPE_MXFP4],   "get_rows_mxfp4",   get_ਪੰਕਤੀਆਂ_mxfp4_len,   get_ਪੰਕਤੀਆਂ_mxfp4_ਅੰਕੜੇ,   "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ਪੰਕਤੀਆਂ[GGML_TYPE_NVFP4],   "get_rows_nvfp4",   get_ਪੰਕਤੀਆਂ_nvfp4_len,   get_ਪੰਕਤੀਆਂ_nvfp4_ਅੰਕੜੇ,   "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ਪੰਕਤੀਆਂ[GGML_TYPE_I32],     "get_rows_i32",     get_ਪੰਕਤੀਆਂ_i32_len,     get_ਪੰਕਤੀਆਂ_i32_ਅੰਕੜੇ,     "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1024, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_get_ਪੰਕਤੀਆਂ_f32[GGML_TYPE_F32 ], "get_rows_f32_f32",  get_ਪੰਕਤੀਆਂ_f32_f32_len,  get_ਪੰਕਤੀਆਂ_f32_f32_ਅੰਕੜੇ,  "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ਪੰਕਤੀਆਂ_f32[GGML_TYPE_F16 ], "get_rows_f16_f32",  get_ਪੰਕਤੀਆਂ_f16_f32_len,  get_ਪੰਕਤੀਆਂ_f16_f32_ਅੰਕੜੇ,  "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ਪੰਕਤੀਆਂ_f32[GGML_TYPE_BF16], "get_rows_bf16_f32", get_ਪੰਕਤੀਆਂ_bf16_f32_len, get_ਪੰਕਤੀਆਂ_bf16_f32_ਅੰਕੜੇ, "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ਪੰਕਤੀਆਂ_f32[GGML_TYPE_Q1_0], "get_rows_q1_0_f32", get_ਪੰਕਤੀਆਂ_q1_0_f32_len, get_ਪੰਕਤੀਆਂ_q1_0_f32_ਅੰਕੜੇ, "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ਪੰਕਤੀਆਂ_f32[GGML_TYPE_Q2_0], "get_rows_q2_0_f32", get_ਪੰਕਤੀਆਂ_q2_0_f32_len, get_ਪੰਕਤੀਆਂ_q2_0_f32_ਅੰਕੜੇ, "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ਪੰਕਤੀਆਂ_f32[GGML_TYPE_Q4_0], "get_rows_q4_0_f32", get_ਪੰਕਤੀਆਂ_q4_0_f32_len, get_ਪੰਕਤੀਆਂ_q4_0_f32_ਅੰਕੜੇ, "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ਪੰਕਤੀਆਂ_f32[GGML_TYPE_Q4_1], "get_rows_q4_1_f32", get_ਪੰਕਤੀਆਂ_q4_1_f32_len, get_ਪੰਕਤੀਆਂ_q4_1_f32_ਅੰਕੜੇ, "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ਪੰਕਤੀਆਂ_f32[GGML_TYPE_Q5_0], "get_rows_q5_0_f32", get_ਪੰਕਤੀਆਂ_q5_0_f32_len, get_ਪੰਕਤੀਆਂ_q5_0_f32_ਅੰਕੜੇ, "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ਪੰਕਤੀਆਂ_f32[GGML_TYPE_Q5_1], "get_rows_q5_1_f32", get_ਪੰਕਤੀਆਂ_q5_1_f32_len, get_ਪੰਕਤੀਆਂ_q5_1_f32_ਅੰਕੜੇ, "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ਪੰਕਤੀਆਂ_f32[GGML_TYPE_Q8_0], "get_rows_q8_0_f32", get_ਪੰਕਤੀਆਂ_q8_0_f32_len, get_ਪੰਕਤੀਆਂ_q8_0_f32_ਅੰਕੜੇ, "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ਪੰਕਤੀਆਂ_f32[GGML_TYPE_Q2_K], "get_rows_q2_k_f32", get_ਪੰਕਤੀਆਂ_q2_k_f32_len, get_ਪੰਕਤੀਆਂ_q2_k_f32_ਅੰਕੜੇ, "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ਪੰਕਤੀਆਂ_f32[GGML_TYPE_Q3_K], "get_rows_q3_k_f32", get_ਪੰਕਤੀਆਂ_q3_k_f32_len, get_ਪੰਕਤੀਆਂ_q3_k_f32_ਅੰਕੜੇ, "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ਪੰਕਤੀਆਂ_f32[GGML_TYPE_Q4_K], "get_rows_q4_k_f32", get_ਪੰਕਤੀਆਂ_q4_k_f32_len, get_ਪੰਕਤੀਆਂ_q4_k_f32_ਅੰਕੜੇ, "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ਪੰਕਤੀਆਂ_f32[GGML_TYPE_Q5_K], "get_rows_q5_k_f32", get_ਪੰਕਤੀਆਂ_q5_k_f32_len, get_ਪੰਕਤੀਆਂ_q5_k_f32_ਅੰਕੜੇ, "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ਪੰਕਤੀਆਂ_f32[GGML_TYPE_Q6_K], "get_rows_q6_k_f32", get_ਪੰਕਤੀਆਂ_q6_k_f32_len, get_ਪੰਕਤੀਆਂ_q6_k_f32_ਅੰਕੜੇ, "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ਪੰਕਤੀਆਂ_f32[GGML_TYPE_TQ1_0], "get_rows_tq1_0_f32", get_ਪੰਕਤੀਆਂ_tq1_0_f32_len, get_ਪੰਕਤੀਆਂ_tq1_0_f32_ਅੰਕੜੇ, "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ਪੰਕਤੀਆਂ_f32[GGML_TYPE_TQ2_0], "get_rows_tq2_0_f32", get_ਪੰਕਤੀਆਂ_tq2_0_f32_len, get_ਪੰਕਤੀਆਂ_tq2_0_f32_ਅੰਕੜੇ, "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ਪੰਕਤੀਆਂ_f32[GGML_TYPE_IQ1_S],   "get_rows_iq1_s_f32",   get_ਪੰਕਤੀਆਂ_iq1_s_f32_len,   get_ਪੰਕਤੀਆਂ_iq1_s_f32_ਅੰਕੜੇ,   "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ਪੰਕਤੀਆਂ_f32[GGML_TYPE_IQ1_M],   "get_rows_iq1_m_f32",   get_ਪੰਕਤੀਆਂ_iq1_m_f32_len,   get_ਪੰਕਤੀਆਂ_iq1_m_f32_ਅੰਕੜੇ,   "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ਪੰਕਤੀਆਂ_f32[GGML_TYPE_IQ2_XXS], "get_rows_iq2_xxs_f32", get_ਪੰਕਤੀਆਂ_iq2_xxs_f32_len, get_ਪੰਕਤੀਆਂ_iq2_xxs_f32_ਅੰਕੜੇ, "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ਪੰਕਤੀਆਂ_f32[GGML_TYPE_IQ2_XS],  "get_rows_iq2_xs_f32",  get_ਪੰਕਤੀਆਂ_iq2_xs_f32_len,  get_ਪੰਕਤੀਆਂ_iq2_xs_f32_ਅੰਕੜੇ,  "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ਪੰਕਤੀਆਂ_f32[GGML_TYPE_IQ2_S],   "get_rows_iq2_s_f32",   get_ਪੰਕਤੀਆਂ_iq2_s_f32_len,   get_ਪੰਕਤੀਆਂ_iq2_s_f32_ਅੰਕੜੇ,   "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ਪੰਕਤੀਆਂ_f32[GGML_TYPE_IQ3_XXS], "get_rows_iq3_xxs_f32", get_ਪੰਕਤੀਆਂ_iq3_xxs_f32_len, get_ਪੰਕਤੀਆਂ_iq3_xxs_f32_ਅੰਕੜੇ, "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ਪੰਕਤੀਆਂ_f32[GGML_TYPE_IQ3_S],   "get_rows_iq3_s_f32",   get_ਪੰਕਤੀਆਂ_iq3_s_f32_len,   get_ਪੰਕਤੀਆਂ_iq3_s_f32_ਅੰਕੜੇ,   "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ਪੰਕਤੀਆਂ_f32[GGML_TYPE_IQ4_XS],  "get_rows_iq4_xs_f32",  get_ਪੰਕਤੀਆਂ_iq4_xs_f32_len,  get_ਪੰਕਤੀਆਂ_iq4_xs_f32_ਅੰਕੜੇ,  "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ਪੰਕਤੀਆਂ_f32[GGML_TYPE_IQ4_NL],  "get_rows_iq4_nl_f32",  get_ਪੰਕਤੀਆਂ_iq4_nl_f32_len,  get_ਪੰਕਤੀਆਂ_iq4_nl_f32_ਅੰਕੜੇ,  "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ਪੰਕਤੀਆਂ_f32[GGML_TYPE_MXFP4],   "get_rows_mxfp4_f32",   get_ਪੰਕਤੀਆਂ_mxfp4_f32_len,   get_ਪੰਕਤੀਆਂ_mxfp4_f32_ਅੰਕੜੇ,   "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ਪੰਕਤੀਆਂ_f32[GGML_TYPE_NVFP4],   "get_rows_nvfp4_f32",   get_ਪੰਕਤੀਆਂ_nvfp4_f32_len,   get_ਪੰਕਤੀਆਂ_nvfp4_f32_ਅੰਕੜੇ,   "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ਪੰਕਤੀਆਂ_ਪਿੱਠ_f32, "get_rows_back_f32", get_ਪੰਕਤੀਆਂ_ਪਿੱਠ_f32_len, get_ਪੰਕਤੀਆਂ_ਪਿੱਠ_f32_ਅੰਕੜੇ, "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {256, 1, 1}, {}, 1, ਸੱਚ);

    ggml_vk_create_pipeline(device, device->pipeline_matmul_ਵੰਡੋ_k_reduce, "split_k_reduce", ਵੰਡੋ_k_reduce_len, ਵੰਡੋ_k_reduce_ਅੰਕੜੇ, "main", 2, 2 * ਆਕਾਰ(uint32_t), {256 * 4, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_flash_attn_ਵੰਡੋ_k_reduce, "fa_split_k_reduce", fa_ਵੰਡੋ_k_reduce_len, fa_ਵੰਡੋ_k_reduce_ਅੰਕੜੇ, "main", 3, ਆਕਾਰ(vk_op_flash_attn_ਵੰਡੋ_k_reduce_ਧੱਕੋ_constants), {1, device->subgroup_ਮਾਪ, 1}, {device->subgroup_ਮਾਪ}, 1, ਸੱਚ);

#ਜੇ defined(VK_KHR_cooperative_ਮੈਟ੍ਰਿਕਸ) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    ਜੇ (device->vendor_id == VK_VENDOR_ID_INTEL && (device->architecture == INTEL_XE2 || (device->architecture == INTEL_XE1 && device->coopmat_support && device->uma))) {
        ਸਵੈਚਾਲਿਤ ਉੱਤਲਾ_power_ਦਾ_2 = [&](uint32_t ਵਿੱਚ) {
            GGML_ASSERT(ਵਿੱਚ != 0);
            ਜੇ (ਵਿੱਚ <= 1) ਵਾਪਸ 1u;
            uint32_t ret = ਵਿੱਚ - 1;
            ret |= ret >> 1;
            ret |= ret >> 2;
            ret |= ret >> 4;
            ret |= ret >> 8;
            ret |= ret >> 16;
            ਵਾਪਸ ret + 1;
            };

        uint32_t xe_native_sub_ਸਮੂਹ_ਮਾਪ = 16;
        ਜੇ (device->architecture == INTEL_XE1) {
            xe_native_sub_ਸਮੂਹ_ਮਾਪ = 8;
        }

        ਲਈ (ਸਵੈਚਾਲਿਤ& it : device->pipeline_xe_fa_decode_dual_phases) {
            ਅਚਲ uint32_t ਵੰਡੋ_p_chunk = 32;
            ਸਵੈਚਾਲਿਤ HdQk = it.ਪਹਿਲਾ;
            ਸਵੈਚਾਲਿਤ& pipelines = it.second;
            uint32_t ਸਿਰ_dim_qk = std::get<0>(HdQk);
            uint32_t ਸਿਰ_dim_pv = std::get<1>(HdQk);
            uint32_t gqa_ਅਨੁਪਾਤ = std::get<2>(HdQk);
            uint32_t q_len = std::get<3>(HdQk);
            ਅਚਲ uint32_t ਬਾਹਰਲਾ_dim_ਪ੍ਰਤੀ_wg = gqa_ਅਨੁਪਾਤ > 16 ? 8 : 16;
            uint32_t aligned_q_len = ਉੱਤਲਾ_power_ਦਾ_2(q_len);
            uint32_t ਸਮੂਹ_sz_ph1 = std::min(std::max(aligned_q_len * xe_native_sub_ਸਮੂਹ_ਮਾਪ, 64u), 256u);
            uint32_t ਬਾਹਰਲਾ_ਪ੍ਰਤੀ_wg_ph1 = std::min(q_len, 256u / xe_native_sub_ਸਮੂਹ_ਮਾਪ);
            uint32_t aligned_gqa_ਅਨੁਪਾਤ = ਉੱਤਲਾ_power_ਦਾ_2(gqa_ਅਨੁਪਾਤ);
            uint32_t ਵੰਡੋ_p_ਪ੍ਰਤੀ_iter_ph2 = 256;
            uint32_t ਵੰਡੋ_p_ਪ੍ਰਤੀ_warp = 16;
            uint32_t ਸਮੂਹ_sz_ph2 = (ਵੰਡੋ_p_ਪ੍ਰਤੀ_iter_ph2 / ਵੰਡੋ_p_ਪ੍ਰਤੀ_warp) * xe_native_sub_ਸਮੂਹ_ਮਾਪ;
            uint32_t ਬਾਹਰਲਾ_ਪ੍ਰਤੀ_wg_ph2 = std::min(std::max(16u / aligned_gqa_ਅਨੁਪਾਤ, 1u), q_len);
            ggml_vk_create_pipeline(device, pipelines.ਪਹਿਲਾ, "xe_fa_decode_ph1", fa_decode_ph1_cm1_len, fa_decode_ph1_cm1_ਅੰਕੜੇ, "main", 5, ਆਕਾਰ(vk_fa_xe_opt_ਧੱਕੋ_constants), { 1, 32, 1 }, { ਸਮੂਹ_sz_ph1, gqa_ਅਨੁਪਾਤ, ਸਿਰ_dim_qk, xe_native_sub_ਸਮੂਹ_ਮਾਪ, ਵੰਡੋ_p_chunk, ਬਾਹਰਲਾ_ਪ੍ਰਤੀ_wg_ph1 }, 1, ਝੂਠ, ਸੱਚ, xe_native_sub_ਸਮੂਹ_ਮਾਪ);
            ggml_vk_create_pipeline(device, pipelines.second, "xe_fa_decode_ph2", fa_decode_ph2_cm1_len, fa_decode_ph2_cm1_ਅੰਕੜੇ, "main", 5, ਆਕਾਰ(vk_fa_xe_opt_ਧੱਕੋ_constants), { 1, 1, 1 }, { ਸਮੂਹ_sz_ph2, gqa_ਅਨੁਪਾਤ, ਸਿਰ_dim_pv, ਬਾਹਰਲਾ_ਪ੍ਰਤੀ_wg_ph2, xe_native_sub_ਸਮੂਹ_ਮਾਪ, ਵੰਡੋ_p_ਪ੍ਰਤੀ_iter_ph2, ਵੰਡੋ_p_chunk, ਬਾਹਰਲਾ_dim_ਪ੍ਰਤੀ_wg }, 1, ਝੂਠ, ਸੱਚ, xe_native_sub_ਸਮੂਹ_ਮਾਪ);
        }
    }
#ਜੇ_ਅੰਤ

    ਲਈ (ਸਵੈਚਾਲਿਤ &it : device->pipeline_fa_ਨਕਾਬ_opt) {
        ਸਵੈਚਾਲਿਤ BrBc = it.ਪਹਿਲਾ;
        ggml_vk_create_pipeline(device, it.second, "fa_mask_opt", fa_ਨਕਾਬ_opt_len, fa_ਨਕਾਬ_opt_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_flash_attn_ਨਕਾਬ_opt_ਧੱਕੋ_constants), {1, 1, 1}, {128, 128 / device->subgroup_ਮਾਪ, BrBc.ਪਹਿਲਾ, BrBc.second}, 1, ਸੱਚ, ਸੱਚ, device->subgroup_ਮਾਪ);
    }

    {
        // Large workgroup so the per-row KV scan parallelizes; capped to device limits.
        ਅਚਲ uint32_t compact_max = std::min({1024u, device->properties.limits.max_Compute_Work_ਸਮੂਹ_Invocations, device->properties.limits.max_Compute_Work_ਸਮੂਹ_ਮਾਪ[0]});

        // Fast ballot prefix-sum path when the device supports full subgroups; otherwise
        // a shared-memory prefix-sum fallback. Both emit a deterministic ascending list.
        device->fa_sparse_compact_use_subgroups = device->subgroup_ballot && device->subgroup_require_ਭਰਿਆ_support;
        ਜੇ (device->fa_sparse_compact_use_subgroups) {
            ਅਚਲ uint32_t compact_wg = std::max(device->subgroup_ਮਾਪ, (compact_max / device->subgroup_ਮਾਪ) * device->subgroup_ਮਾਪ);
            ਅਚਲ uint32_t compact_ਅੰਕ_sg = compact_wg / device->subgroup_ਮਾਪ;
            ggml_vk_create_pipeline(device, device->pipeline_fa_sparse_compact_subgroup, "fa_sparse_compact_subgroup", fa_sparse_compact_subgroup_len, fa_sparse_compact_subgroup_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_flash_attn_sparse_compact_ਧੱਕੋ_constants), {1, 1, 1}, {compact_wg, compact_ਅੰਕ_sg}, 1, ਸੱਚ, ਸੱਚ, device->subgroup_ਮਾਪ);
        } ਨਹੀਂਤਾਂ {
            ggml_vk_create_pipeline(device, device->pipeline_fa_sparse_compact, "fa_sparse_compact", fa_sparse_compact_len, fa_sparse_compact_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_flash_attn_sparse_compact_ਧੱਕੋ_constants), {1, 1, 1}, {compact_max}, 1, ਸੱਚ);
        }
    }

    ਜੇ (device->subgroup_clustered && device->subgroup_require_ਭਰਿਆ_support) {
        ggml_vk_create_pipeline(device, device->pipeline_quantize_q8_1_x4, "quantize_q8_1_x4", quantize_q8_1_x4_subgroup_len, quantize_q8_1_x4_subgroup_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_quantize_q8_1_ਧੱਕੋ_constants), {32 * device->subgroup_ਮਾਪ / 8, 1, 1}, { device->subgroup_ਮਾਪ }, 1, ਸੱਚ, ਸੱਚ);
    } ਨਹੀਂਤਾਂ {
        ggml_vk_create_pipeline(device, device->pipeline_quantize_q8_1_x4, "quantize_q8_1_x4", quantize_q8_1_x4_len, quantize_q8_1_x4_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_quantize_q8_1_ਧੱਕੋ_constants), {32 * device->subgroup_ਮਾਪ / 8, 1, 1}, { device->subgroup_ਮਾਪ }, 1);
    }

    ਲਈ (uint32_t i = 0; i < p021_max_gqa_ਅਨੁਪਾਤ; ++i) {
        ਜੇ (device->subgroup_arithmetic && device->subgroup_require_ਭਰਿਆ_support) {
            ggml_vk_create_pipeline2(device, device->pipeline_mul_mat_vec_p021_f16_f32[i], "mul_mat_vec_p021_f16_f32"+std::to_string(i+1), mul_mat_vec_p021_f16_f32_subgroup_ਜੋੜੋ_len, mul_mat_vec_p021_f16_f32_subgroup_ਜੋੜੋ_ਅੰਕੜੇ, "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_p021_ਧੱਕੋ_constants), {1, 1, 1}, {device->subgroup_ਮਾਪ, i + 1}, 1, ਸੱਚ, ਸੱਚ);
        } ਨਹੀਂਤਾਂ {
            ggml_vk_create_pipeline2(device, device->pipeline_mul_mat_vec_p021_f16_f32[i], "mul_mat_vec_p021_f16_f32"+std::to_string(i+1), mul_mat_vec_p021_f16_f32_len,              mul_mat_vec_p021_f16_f32_ਅੰਕੜੇ,              "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_p021_ਧੱਕੋ_constants), {1, 1, 1}, {device->subgroup_ਮਾਪ, i + 1}, 1, ਸੱਚ);
        }
    }
    ggml_vk_create_pipeline(device, device->pipeline_mul_mat_vec_nc_f16_f32, "mul_mat_vec_nc_f16_f32", mul_mat_vec_nc_f16_f32_len, mul_mat_vec_nc_f16_f32_ਅੰਕੜੇ, "main", mul_mat_vec_ਅੰਕ_bindings, ਆਕਾਰ(vk_mat_vec_nc_ਧੱਕੋ_constants), {1, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_norm_f32, "norm_f32", norm_f32_len, norm_f32_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_unary_ਧੱਕੋ_constants), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_ਸਮੂਹ_norm_f32, "group_norm_f32", ਸਮੂਹ_norm_f32_len, ਸਮੂਹ_norm_f32_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_ਧੱਕੋ_constants), {1, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_f32, "rms_norm_f32", rms_norm_f32_len, rms_norm_f32_ਅੰਕੜੇ, "main", 4, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1, 1, 1}, {0, 0}, 1, ਸੱਚ);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_f32, "rms_norm_mul_f32", rms_norm_f32_len, rms_norm_f32_ਅੰਕੜੇ, "main", 4, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1, 1, 1}, {0, 1}, 1, ਸੱਚ);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_ਜੋੜੋ_f32, "rms_norm_mul_add_f32", rms_norm_mul_ਜੋੜੋ_f32_len, rms_norm_mul_ਜੋੜੋ_f32_ਅੰਕੜੇ, "main", 5, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1, 1, 1}, {0, 1, 0}, 1, ਸੱਚ);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_ਜੋੜੋ_mul_f32, "rms_norm_mul_add_mul_f32", rms_norm_mul_ਜੋੜੋ_f32_len, rms_norm_mul_ਜੋੜੋ_f32_ਅੰਕੜੇ, "main", 5, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1, 1, 1}, {0, 1, 1}, 1, ਸੱਚ);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_ਜੋੜੋ_partials_f32, "rms_norm_mul_add_partials_f32", rms_norm_mul_ਜੋੜੋ_partials_f32_len, rms_norm_mul_ਜੋੜੋ_partials_f32_ਅੰਕੜੇ, "main", 6, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1, 1, 1}, {0, 1, 0}, 1, ਸੱਚ);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_ਜੋੜੋ_mul_partials_f32, "rms_norm_mul_add_mul_partials_f32", rms_norm_mul_ਜੋੜੋ_partials_f32_len, rms_norm_mul_ਜੋੜੋ_partials_f32_ਅੰਕੜੇ, "main", 6, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1, 1, 1}, {0, 1, 1}, 1, ਸੱਚ);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_ਸੈੱਟ_ਪੰਕਤੀਆਂ_f32_f32, "rms_norm_set_rows_f32_f32", rms_norm_ਸੈੱਟ_ਪੰਕਤੀਆਂ_f32_f32_len, rms_norm_ਸੈੱਟ_ਪੰਕਤੀਆਂ_f32_f32_ਅੰਕੜੇ, "main", 4, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1, 1, 1}, {0, 0}, 1, ਸੱਚ);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_ਸੈੱਟ_ਪੰਕਤੀਆਂ_f32_f16, "rms_norm_set_rows_f32_f16", rms_norm_ਸੈੱਟ_ਪੰਕਤੀਆਂ_f32_f16_len, rms_norm_ਸੈੱਟ_ਪੰਕਤੀਆਂ_f32_f16_ਅੰਕੜੇ, "main", 4, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1, 1, 1}, {0, 0}, 1, ਸੱਚ);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_partials_f32, "rms_norm_partials_f32", rms_norm_partials_f32_len, rms_norm_partials_f32_ਅੰਕੜੇ, "main", 4, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1, 1, 1}, {0, 0}, 1, ਸੱਚ);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_partials_f32, "rms_norm_mul_partials_f32", rms_norm_partials_f32_len, rms_norm_partials_f32_ਅੰਕੜੇ, "main", 4, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1, 1, 1}, {0, 1}, 1, ਸੱਚ);

    ਜੇ (ਆਕਾਰ(vk_op_rms_norm_mul_rope_ਧੱਕੋ_constants) <= device->properties.limits.max_ਧੱਕੋ_Constants_ਮਾਪ) {
        ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_rope_f32_f32, "rms_norm_mul_rope_f32_f32", rms_norm_mul_rope_f32_f32_len, rms_norm_mul_rope_f32_f32_ਅੰਕੜੇ, "main", 7, ਆਕਾਰ(vk_op_rms_norm_mul_rope_ਧੱਕੋ_constants), {1, 1, 1}, {0, 1}, 1, ਸੱਚ);
        ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_rope_f32_f16, "rms_norm_mul_rope_f32_f16", rms_norm_mul_rope_f32_f16_len, rms_norm_mul_rope_f32_f16_ਅੰਕੜੇ, "main", 7, ਆਕਾਰ(vk_op_rms_norm_mul_rope_ਧੱਕੋ_constants), {1, 1, 1}, {0, 1}, 1, ਸੱਚ);
    }

    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_ਪਿੱਠ_f32, "rms_norm_back_f32", rms_norm_ਪਿੱਠ_f32_len, rms_norm_ਪਿੱਠ_f32_ਅੰਕੜੇ, "main", 3, ਆਕਾਰ(vk_op_ਧੱਕੋ_constants), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_l2_norm_f32, "l2_norm_f32", l2_norm_f32_len, l2_norm_f32_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_unary_ਧੱਕੋ_constants), {1, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_cpy_32, "cpy_32", cpy_32_len, cpy_32_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_unary_ਧੱਕੋ_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_f16, "cpy_f32_f16", cpy_f32_f16_len, cpy_f32_f16_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_unary_ਧੱਕੋ_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_16, "cpy_16", cpy_16_len, cpy_16_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_unary_ਧੱਕੋ_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f16_f32, "cpy_f16_f32", cpy_f16_f32_len, cpy_f16_f32_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_unary_ਧੱਕੋ_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_bf16,"cpy_f32_bf16",cpy_f32_bf16_len,cpy_f32_bf16_ਅੰਕੜੇ,"main", 2, ਆਕਾਰ(vk_op_unary_ਧੱਕੋ_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_bf16_f32,"cpy_bf16_f32",cpy_bf16_f32_len,cpy_bf16_f32_ਅੰਕੜੇ,"main", 2, ਆਕਾਰ(vk_op_unary_ਧੱਕੋ_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_bf16_f16,"cpy_bf16_f16",cpy_bf16_f16_len,cpy_bf16_f16_ਅੰਕੜੇ,"main", 2, ਆਕਾਰ(vk_op_unary_ਧੱਕੋ_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_i32_f32, "cpy_i32_f32", cpy_i32_f32_len, cpy_i32_f32_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_unary_ਧੱਕੋ_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_i32, "cpy_f32_i32", cpy_f32_i32_len, cpy_f32_i32_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_unary_ਧੱਕੋ_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_32, "contig_cpy_32", contig_cpy_32_len, contig_cpy_32_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_unary_ਧੱਕੋ_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_f32_f16, "contig_cpy_f32_f16", contig_cpy_f32_f16_len, contig_cpy_f32_f16_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_unary_ਧੱਕੋ_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_16, "contig_cpy_16", contig_cpy_16_len, contig_cpy_16_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_unary_ਧੱਕੋ_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_f16_f32, "contig_cpy_f16_f32", contig_cpy_f16_f32_len, contig_cpy_f16_f32_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_unary_ਧੱਕੋ_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_f32_bf16,"contig_cpy_f32_bf16",contig_cpy_f32_bf16_len,contig_cpy_f32_bf16_ਅੰਕੜੇ,"main", 2, ਆਕਾਰ(vk_op_unary_ਧੱਕੋ_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_bf16_f32,"contig_cpy_bf16_f32",contig_cpy_bf16_f32_len,contig_cpy_bf16_f32_ਅੰਕੜੇ,"main", 2, ਆਕਾਰ(vk_op_unary_ਧੱਕੋ_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_bf16_f16,"contig_cpy_bf16_f16",contig_cpy_bf16_f16_len,contig_cpy_bf16_f16_ਅੰਕੜੇ,"main", 2, ਆਕਾਰ(vk_op_unary_ਧੱਕੋ_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_i32_f32, "contig_cpy_i32_f32", contig_cpy_i32_f32_len, contig_cpy_i32_f32_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_unary_ਧੱਕੋ_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_f32_i32, "contig_cpy_f32_i32", contig_cpy_f32_i32_len, contig_cpy_f32_i32_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_unary_ਧੱਕੋ_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_cpy_transpose_32, "cpy_transpose_32", cpy_transpose_32_len, cpy_transpose_32_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_unary_ਧੱਕੋ_constants), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_transpose_16, "cpy_transpose_16", cpy_transpose_16_len, cpy_transpose_16_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_unary_ਧੱਕੋ_constants), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_transpose_02_32, "cpy_transpose_02_32", cpy_transpose_02_32_len, cpy_transpose_02_32_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_unary_ਧੱਕੋ_constants), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_transpose_02_16, "cpy_transpose_02_16", cpy_transpose_02_16_len, cpy_transpose_02_16_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_unary_ਧੱਕੋ_constants), {1, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q1_0], "cpy_f32_q1_0", cpy_f32_q1_0_len, cpy_f32_q1_0_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_unary_ਧੱਕੋ_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q2_0], "cpy_f32_q2_0", cpy_f32_q2_0_len, cpy_f32_q2_0_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_unary_ਧੱਕੋ_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q4_0], "cpy_f32_q4_0", cpy_f32_q4_0_len, cpy_f32_q4_0_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_unary_ਧੱਕੋ_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q4_1], "cpy_f32_q4_1", cpy_f32_q4_1_len, cpy_f32_q4_1_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_unary_ਧੱਕੋ_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q5_0], "cpy_f32_q5_0", cpy_f32_q5_0_len, cpy_f32_q5_0_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_unary_ਧੱਕੋ_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q5_1], "cpy_f32_q5_1", cpy_f32_q5_1_len, cpy_f32_q5_1_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_unary_ਧੱਕੋ_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q8_0], "cpy_f32_q8_0", cpy_f32_q8_0_len, cpy_f32_q8_0_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_unary_ਧੱਕੋ_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_IQ4_NL], "cpy_f32_iq4_nl", cpy_f32_iq4_nl_len, cpy_f32_iq4_nl_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_unary_ਧੱਕੋ_constants), {32, 1, 1}, {}, 1);

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ਸੈੱਟ_ਪੰਕਤੀਆਂ_2(src_idx, src, itype) \
        ggml_vk_create_pipeline(device, device->pipeline_ਸੈੱਟ_ਪੰਕਤੀਆਂ ## itype [src_idx][GGML_TYPE_F32],  "set_rows_" #src "_f32" #itype,  ਸੈੱਟ_ਪੰਕਤੀਆਂ_3 ## src ## _f32 ## itype ## _len,  ਸੈੱਟ_ਪੰਕਤੀਆਂ_3 ## src ## _f32 ## itype ## ਅੰਕੜੇ_2,  "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1, 1, 1}, {1}, 1, ਸੱਚ); \
        ggml_vk_create_pipeline(device, device->pipeline_ਸੈੱਟ_ਪੰਕਤੀਆਂ ## itype [src_idx][GGML_TYPE_F16],  "set_rows_" #src "_f16" #itype,  ਸੈੱਟ_ਪੰਕਤੀਆਂ_3 ## src ## _f16 ## itype ## _len,  ਸੈੱਟ_ਪੰਕਤੀਆਂ_3 ## src ## _f16 ## itype ## ਅੰਕੜੇ_2,  "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1, 1, 1}, {1}, 1, ਸੱਚ); \
        ggml_vk_create_pipeline(device, device->pipeline_ਸੈੱਟ_ਪੰਕਤੀਆਂ ## itype [src_idx][GGML_TYPE_BF16], "set_rows_" #src "_bf16" #itype, ਸੈੱਟ_ਪੰਕਤੀਆਂ_3 ## src ## _bf16 ## itype ## _len, ਸੈੱਟ_ਪੰਕਤੀਆਂ_3 ## src ## _bf16 ## itype ## ਅੰਕੜੇ_2, "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1, 1, 1}, {1}, 1, ਸੱਚ); \
        ggml_vk_create_pipeline(device, device->pipeline_ਸੈੱਟ_ਪੰਕਤੀਆਂ ## itype [src_idx][GGML_TYPE_Q1_0], "set_rows_" #src "_q1_0" #itype, ਸੈੱਟ_ਪੰਕਤੀਆਂ_3 ## src ## _q1_0 ## itype ## _len, ਸੈੱਟ_ਪੰਕਤੀਆਂ_3 ## src ## _q1_0 ## itype ## ਅੰਕੜੇ_2, "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1, 1, 1}, {1}, 1, ਸੱਚ); \
        ggml_vk_create_pipeline(device, device->pipeline_ਸੈੱਟ_ਪੰਕਤੀਆਂ ## itype [src_idx][GGML_TYPE_Q2_0], "set_rows_" #src "_q2_0" #itype, ਸੈੱਟ_ਪੰਕਤੀਆਂ_3 ## src ## _q2_0 ## itype ## _len, ਸੈੱਟ_ਪੰਕਤੀਆਂ_3 ## src ## _q2_0 ## itype ## ਅੰਕੜੇ_2, "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1, 1, 1}, {1}, 1, ਸੱਚ); \
        ggml_vk_create_pipeline(device, device->pipeline_ਸੈੱਟ_ਪੰਕਤੀਆਂ ## itype [src_idx][GGML_TYPE_Q4_0], "set_rows_" #src "_q4_0" #itype, ਸੈੱਟ_ਪੰਕਤੀਆਂ_3 ## src ## _q4_0 ## itype ## _len, ਸੈੱਟ_ਪੰਕਤੀਆਂ_3 ## src ## _q4_0 ## itype ## ਅੰਕੜੇ_2, "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1, 1, 1}, {1}, 1, ਸੱਚ); \
        ggml_vk_create_pipeline(device, device->pipeline_ਸੈੱਟ_ਪੰਕਤੀਆਂ ## itype [src_idx][GGML_TYPE_Q4_1], "set_rows_" #src "_q4_1" #itype, ਸੈੱਟ_ਪੰਕਤੀਆਂ_3 ## src ## _q4_1 ## itype ## _len, ਸੈੱਟ_ਪੰਕਤੀਆਂ_3 ## src ## _q4_1 ## itype ## ਅੰਕੜੇ_2, "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1, 1, 1}, {1}, 1, ਸੱਚ); \
        ggml_vk_create_pipeline(device, device->pipeline_ਸੈੱਟ_ਪੰਕਤੀਆਂ ## itype [src_idx][GGML_TYPE_Q5_0], "set_rows_" #src "_q5_0" #itype, ਸੈੱਟ_ਪੰਕਤੀਆਂ_3 ## src ## _q5_0 ## itype ## _len, ਸੈੱਟ_ਪੰਕਤੀਆਂ_3 ## src ## _q5_0 ## itype ## ਅੰਕੜੇ_2, "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1, 1, 1}, {1}, 1, ਸੱਚ); \
        ggml_vk_create_pipeline(device, device->pipeline_ਸੈੱਟ_ਪੰਕਤੀਆਂ ## itype [src_idx][GGML_TYPE_Q5_1], "set_rows_" #src "_q5_1" #itype, ਸੈੱਟ_ਪੰਕਤੀਆਂ_3 ## src ## _q5_1 ## itype ## _len, ਸੈੱਟ_ਪੰਕਤੀਆਂ_3 ## src ## _q5_1 ## itype ## ਅੰਕੜੇ_2, "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1, 1, 1}, {1}, 1, ਸੱਚ); \
        ggml_vk_create_pipeline(device, device->pipeline_ਸੈੱਟ_ਪੰਕਤੀਆਂ ## itype [src_idx][GGML_TYPE_Q8_0], "set_rows_" #src "_q8_0" #itype, ਸੈੱਟ_ਪੰਕਤੀਆਂ_3 ## src ## _q8_0 ## itype ## _len, ਸੈੱਟ_ਪੰਕਤੀਆਂ_3 ## src ## _q8_0 ## itype ## ਅੰਕੜੇ_2, "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1, 1, 1}, {1}, 1, ਸੱਚ); \
        ggml_vk_create_pipeline(device, device->pipeline_ਸੈੱਟ_ਪੰਕਤੀਆਂ ## itype [src_idx][GGML_TYPE_IQ4_NL], "set_rows_" #src "_iq4_nl" #itype, ਸੈੱਟ_ਪੰਕਤੀਆਂ_3 ## src ## _iq4_nl ## itype ## _len, ਸੈੱਟ_ਪੰਕਤੀਆਂ_3 ## src ## _iq4_nl ## itype ## ਅੰਕੜੇ_2, "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1, 1, 1}, {1}, 1, ਸੱਚ);

    ਸੈੱਟ_ਪੰਕਤੀਆਂ_2(0, f32, _i32)
    ਸੈੱਟ_ਪੰਕਤੀਆਂ_2(0, f32, _i64)
    ਸੈੱਟ_ਪੰਕਤੀਆਂ_2(1, f16, _i32)
    ਸੈੱਟ_ਪੰਕਤੀਆਂ_2(1, f16, _i64)
#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ ਸੈੱਟ_ਪੰਕਤੀਆਂ_2


    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q1_0], "cpy_q1_0_f32", cpy_q1_0_f32_len, cpy_q1_0_f32_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_unary_ਧੱਕੋ_constants), {(uint32_t)ggml_blck_ਮਾਪ(GGML_TYPE_Q1_0), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q2_0], "cpy_q2_0_f32", cpy_q2_0_f32_len, cpy_q2_0_f32_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_unary_ਧੱਕੋ_constants), {(uint32_t)ggml_blck_ਮਾਪ(GGML_TYPE_Q2_0), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q4_0], "cpy_q4_0_f32", cpy_q4_0_f32_len, cpy_q4_0_f32_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_unary_ਧੱਕੋ_constants), {(uint32_t)ggml_blck_ਮਾਪ(GGML_TYPE_Q4_0), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q4_1], "cpy_q4_1_f32", cpy_q4_1_f32_len, cpy_q4_1_f32_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_unary_ਧੱਕੋ_constants), {(uint32_t)ggml_blck_ਮਾਪ(GGML_TYPE_Q4_1), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q5_0], "cpy_q5_0_f32", cpy_q5_0_f32_len, cpy_q5_0_f32_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_unary_ਧੱਕੋ_constants), {(uint32_t)ggml_blck_ਮਾਪ(GGML_TYPE_Q5_0), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q5_1], "cpy_q5_1_f32", cpy_q5_1_f32_len, cpy_q5_1_f32_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_unary_ਧੱਕੋ_constants), {(uint32_t)ggml_blck_ਮਾਪ(GGML_TYPE_Q5_1), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q8_0], "cpy_q8_0_f32", cpy_q8_0_f32_len, cpy_q8_0_f32_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_unary_ਧੱਕੋ_constants), {(uint32_t)ggml_blck_ਮਾਪ(GGML_TYPE_Q8_0), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_IQ4_NL], "cpy_iq4_nl_f32", cpy_iq4_nl_f32_len, cpy_iq4_nl_f32_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_unary_ਧੱਕੋ_constants), {(uint32_t)ggml_blck_ਮਾਪ(GGML_TYPE_IQ4_NL), 1, 1}, {}, 1);

    ਸਵੈਚਾਲਿਤ get_suffix = [](ਤਾਰਕਿਕ src0_f16, ਤਾਰਕਿਕ src1_f16, ਤਾਰਕਿਕ dst_f16) {
        std::string s;
        s += std::string(src0_f16 ? "_f16" : "_f32");
        s += std::string(src1_f16 ? "_f16" : "_f32");
        s += std::string(dst_f16 ? "_f16" : "_f32");
        ਵਾਪਸ s;
    };

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CREATE_ਦੁਆਧਾਰੀ(name, namemod, spec, bindings) \
    ਲਈ (ਪੂਰਨ s0 : {0,1}) ਲਈ (ਪੂਰਨ s1 : {0,1}) ਲਈ (ਪੂਰਨ d : {0,1}) \
        ggml_vk_create_pipeline2(device, device->pipeline_ ## name ## namemod[s0][s1][d], \
                                #name + get_suffix(s0, s1, d) + #namemod, name ## _len[s0][s1][d], name ## ਅੰਕੜੇ_2[s0][s1][d], \
                                "main", (bindings), ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {512, 1, 1}, spec, 1);

    CREATE_ਦੁਆਧਾਰੀ(ਜੋੜੋ_2, , {0}, 4)
    CREATE_ਦੁਆਧਾਰੀ(ਜੋੜੋ_2, _norepeat, {1}, 4)
    CREATE_ਦੁਆਧਾਰੀ(sub, , {0}, 3)
    CREATE_ਦੁਆਧਾਰੀ(sub, _norepeat, {1}, 3)
    CREATE_ਦੁਆਧਾਰੀ(mul, , {0}, 3)
    CREATE_ਦੁਆਧਾਰੀ(mul, _norepeat, {1}, 3)
    CREATE_ਦੁਆਧਾਰੀ(div, , {0}, 3)
    CREATE_ਦੁਆਧਾਰੀ(div, _norepeat, {1}, 3)
    CREATE_ਦੁਆਧਾਰੀ(ਜੋੜੋ_rms, , {0}, 4)
    CREATE_ਦੁਆਧਾਰੀ(ਜੋੜੋ_rms, _norepeat, {1}, 4)
#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ CREATE_ਦੁਆਧਾਰੀ

    ਜੇ (device->multi_ਜੋੜੋ) {
        ਲਈ (uint32_t i = 0; i < MAX_FUSED_ADDS; ++i) {
            ggml_vk_create_pipeline2(device, device->pipeline_multi_ਜੋੜੋ[i],     "multi_add_f32_"     + std::to_string(i+1), multi_ਜੋੜੋ_f32_len,     multi_ਜੋੜੋ_f32_ਅੰਕੜੇ,     "main", MAX_PARAMETER_ਗਿਣਤੀ, ਆਕਾਰ(vk_op_multi_ਜੋੜੋ_ਧੱਕੋ_constants), {512, 1, 1}, {i+2}, 1);
            ggml_vk_create_pipeline2(device, device->pipeline_multi_ਜੋੜੋ_rms[i], "multi_add_rms_f32_" + std::to_string(i+1), multi_ਜੋੜੋ_rms_f32_len, multi_ਜੋੜੋ_rms_f32_ਅੰਕੜੇ, "main", MAX_PARAMETER_ਗਿਣਤੀ, ਆਕਾਰ(vk_op_multi_ਜੋੜੋ_ਧੱਕੋ_constants), {512, 1, 1}, {i+2}, 1);
        }
    }

    ggml_vk_create_pipeline(device, device->pipeline_ਜੋੜੋ_id_f32, "add_id_f32", ਜੋੜੋ_id_f32_len, ਜੋੜੋ_id_f32_ਅੰਕੜੇ, "main", 4, ਆਕਾਰ(vk_op_ਜੋੜੋ_id_ਧੱਕੋ_constants), {1, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_acc_f32, "acc_f32", acc_f32_len, acc_f32_ਅੰਕੜੇ, "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {512, 1, 1}, {0, 1}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_ਸੈੱਟ_f32, "set_f32", acc_f32_len, acc_f32_ਅੰਕੜੇ, "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {512, 1, 1}, {0, 0}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_concat_i8, "concat_i8", concat_i8_len, concat_i8_ਅੰਕੜੇ, "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_concat_i16, "concat_i16", concat_i16_len, concat_i16_ਅੰਕੜੇ, "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_concat_i32, "concat_i32", concat_i32_len, concat_i32_ਅੰਕੜੇ, "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_concat_i64, "concat_i64", concat_i64_len, concat_i64_ਅੰਕੜੇ, "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_upscale_ਸਭ_ਤੋਂ_ਨੇੜੇ_f32, "upscale_f32", upscale_f32_len, upscale_f32_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_upscale_ਧੱਕੋ_constants), {512, 1, 1}, {GGML_SCALE_ਬਹੁਲਕ_ਸਭ_ਤੋਂ_ਨੇੜੇ}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_upscale_bilinear_f32, "upscale_f32", upscale_f32_len, upscale_f32_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_upscale_ਧੱਕੋ_constants), {512, 1, 1}, {GGML_SCALE_ਬਹੁਲਕ_BILINEAR}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_upscale_bicubic_f32, "upscale_f32", upscale_f32_len, upscale_f32_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_upscale_ਧੱਕੋ_constants), {512, 1, 1}, {GGML_SCALE_ਬਹੁਲਕ_BICUBIC}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_upscale_bilinear_antialias_f32, "upscale_f32", upscale_f32_len, upscale_f32_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_upscale_ਧੱਕੋ_constants), {512, 1, 1}, {GGML_SCALE_ਬਹੁਲਕ_BILINEAR | GGML_SCALE_FLAG_ANTIALIAS}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_scale_f32, "scale_f32", scale_f32_len, scale_f32_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_unary_ਧੱਕੋ_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_log[0], "log_f32", log_f32_len, log_f32_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_unary_ਧੱਕੋ_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_log[1], "log_f16", log_f16_len, log_f16_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_unary_ਧੱਕੋ_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_tri[0], "tri_f32", tri_f32_len, tri_f32_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_unary_ਧੱਕੋ_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_tri[1], "tri_f16", tri_f16_len, tri_f16_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_unary_ਧੱਕੋ_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_diag[0], "diag_f32", diag_f32_len, diag_f32_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_unary_ਧੱਕੋ_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_diag[1], "diag_f16", diag_f16_len, diag_f16_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_unary_ਧੱਕੋ_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_pad_f32, "pad_f32", pad_f32_len, pad_f32_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_pad_ਧੱਕੋ_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_pad_reflect_1d_f32, "pad_reflect_1d_f32", pad_reflect_1d_f32_len, pad_reflect_1d_f32_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_unary_ਧੱਕੋ_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_roll_f32, "roll_f32", roll_f32_len, roll_f32_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_unary_ਧੱਕੋ_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_repeat_i32, "repeat_i32", repeat_i32_len, repeat_i32_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_unary_ਧੱਕੋ_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_repeat_ਪਿੱਠ_f32, "repeat_back_f32", repeat_ਪਿੱਠ_f32_len, repeat_ਪਿੱਠ_f32_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_unary_ਧੱਕੋ_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_repeat_i16, "repeat_i16", repeat_i16_len, repeat_i16_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_unary_ਧੱਕੋ_constants), {512, 1, 1}, {}, 1);

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CREATE_UNARY(name)  \
    ggml_vk_create_pipeline(device, device->pipeline_ ## name [0], #name "_f32", name ## _f32_len, name ## f32_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_unary_ਧੱਕੋ_constants), {512, 1, 1}, {}, 1);  \
    ggml_vk_create_pipeline(device, device->pipeline_ ## name [1], #name "_f16", name ## _f16_len, name ## f16_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_unary_ਧੱਕੋ_constants), {512, 1, 1}, {}, 1);

    CREATE_UNARY(elu)
    CREATE_UNARY(gelu)
    CREATE_UNARY(gelu_erf)
    CREATE_UNARY(gelu_ਤੇਜ਼)
    CREATE_UNARY(silu)
    CREATE_UNARY(relu)
    CREATE_UNARY(sqr)
    CREATE_UNARY(ਵਰਗਮੂਲ)
    CREATE_UNARY(ਜੀਵਾ)
    CREATE_UNARY(ਕੋਜੀਵਾ)
    CREATE_UNARY(clamp)
    CREATE_UNARY(leaky_relu)
    CREATE_UNARY(xielu)
    CREATE_UNARY(neg)
    CREATE_UNARY(tanh)
    CREATE_UNARY(sigmoid)
    CREATE_UNARY(hardsigmoid)
    CREATE_UNARY(hardswish)
    CREATE_UNARY(ਨਿਰਪੇਖ)
    CREATE_UNARY(softplus)
    CREATE_UNARY(ਕਦਮ)
    CREATE_UNARY(round)
    CREATE_UNARY(ceil)
    CREATE_UNARY(floor)
    CREATE_UNARY(trunc)
    CREATE_UNARY(sgn)
    CREATE_UNARY(ਘਾਤਾਂਕ)
    CREATE_UNARY(expm1)
#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ CREATE_UNARY

// spec constants: {norepeat, op_on_b}
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CREATE_UNARY_MUL(name, idx) \
    ਲਈ (ਪੂਰਨ dt = 0; dt < 2; ++dt) { \
        ਅਚਲ size_t len_ = dt ? name ## _mul_f16_len : name ## _mul_f32_len; \
        ਅਚਲ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ * ਅੰਕੜੇ_3 = dt ? name ## mul_f16_ਅੰਕੜੇ : name ## mul_f32_ਅੰਕੜੇ; \
        ਅਚਲ std::string dts_ = dt ? "f16" : "f32"; \
        ਲਈ (ਪੂਰਨ ob = 0; ob < 2; ++ob) \
            ਲਈ (ਪੂਰਨ nr = 0; nr < 2; ++nr) \
                ggml_vk_create_pipeline(device, device->pipeline_unary_mul[(idx)][dt][nr][ob], \
                    (#name "_mul" + std::string(ob ? "_b" : "") + "_" + dts_ + (nr ? "_norepeat" : "")).c_str(), \
                    len_, ਅੰਕੜੇ_3, "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {512, 1, 1}, \
                    { (uint32_t) nr, (uint32_t) ob }, 1); \
    }

    CREATE_UNARY_MUL(gelu, 0)
    CREATE_UNARY_MUL(sigmoid, 1)
    CREATE_UNARY_MUL(silu, 2)
    CREATE_UNARY_MUL(softplus, 3)
#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ CREATE_UNARY_MUL

    ggml_vk_create_pipeline(device, device->pipeline_add1_f16_f16, "add1_f16_f16", add1_f16_f16_len, add1_f16_f16_ਅੰਕੜੇ, "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_add1_f16_f32, "add1_f16_f32", add1_f16_f32_len, add1_f16_f32_ਅੰਕੜੇ, "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_add1_f32_f32, "add1_f32_f32", add1_f32_f32_len, add1_f32_f32_ਅੰਕੜੇ, "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_arange_f32, "arange_f32", arange_f32_len, arange_f32_ਅੰਕੜੇ, "main", 1, ਆਕਾਰ(vk_op_ਧੱਕੋ_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_fill_f32, "fill_f32", fill_f32_len, fill_f32_ਅੰਕੜੇ, "main", 1, ਆਕਾਰ(vk_op_ਧੱਕੋ_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_fill_f16, "fill_f16", fill_f16_len, fill_f16_ਅੰਕੜੇ, "main", 1, ਆਕਾਰ(vk_op_ਧੱਕੋ_constants), {512, 1, 1}, {}, 1);

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CREATE_GLU(name)  \
    ggml_vk_create_pipeline(device, device->pipeline_ ## name [0], #name "_f32", name ## _f32_len, name ## f32_ਅੰਕੜੇ, "main", 3, ਆਕਾਰ(vk_op_glu_ਧੱਕੋ_constants), {512, 1, 1}, {}, 1, ਸੱਚ);   \
    ggml_vk_create_pipeline(device, device->pipeline_ ## name [1], #name "_f16", name ## _f16_len, name ## f16_ਅੰਕੜੇ, "main", 3, ਆਕਾਰ(vk_op_glu_ਧੱਕੋ_constants), {512, 1, 1}, {}, 1, ਸੱਚ);

    CREATE_GLU(geglu)
    CREATE_GLU(reglu)
    CREATE_GLU(swiglu)
    CREATE_GLU(swiglu_oai)
    CREATE_GLU(swiglu_clamp)
    CREATE_GLU(geglu_erf)
    CREATE_GLU(geglu_ਤੇਜ਼)
#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ CREATE_GLU

    ggml_vk_create_pipeline(device, device->pipeline_silu_ਪਿੱਠ_f32, "silu_back_f32", silu_ਪਿੱਠ_f32_len, silu_ਪਿੱਠ_f32_ਅੰਕੜੇ, "main", 3, ਆਕਾਰ(vk_op_ਧੱਕੋ_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_diag_ਨਕਾਬ_inf_f32, "diag_mask_inf_f32", diag_ਨਕਾਬ_inf_f32_len, diag_ਨਕਾਬ_inf_f32_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_diag_ਨਕਾਬ_ਧੱਕੋ_constants), {1, 512, 1}, {}, 1, ਸੱਚ);

    ggml_vk_create_pipeline(device, device->pipeline_soft_max_f32, "soft_max_f32", soft_max_f32_len, soft_max_f32_ਅੰਕੜੇ, "main", 4, ਆਕਾਰ(vk_op_soft_max_ਧੱਕੋ_constants), {1, 1, 1}, { device->subgroup_ਮਾਪ }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_f32_wg512, "soft_max_f32_wg512", soft_max_f32_len, soft_max_f32_ਅੰਕੜੇ, "main", 4, ਆਕਾਰ(vk_op_soft_max_ਧੱਕੋ_constants), {1, 1, 1}, { 512 }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_f32_f16, "soft_max_f32_f16", soft_max_f32_f16_len, soft_max_f32_f16_ਅੰਕੜੇ, "main", 4, ਆਕਾਰ(vk_op_soft_max_ਧੱਕੋ_constants), {1, 1, 1}, { device->subgroup_ਮਾਪ }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_f32_f16_wg512, "soft_max_f32_f16_wg512", soft_max_f32_f16_len, soft_max_f32_f16_ਅੰਕੜੇ, "main", 4, ਆਕਾਰ(vk_op_soft_max_ਧੱਕੋ_constants), {1, 1, 1}, { 512 }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_ਪਿੱਠ_f32, "soft_max_back_f32", soft_max_ਪਿੱਠ_f32_len, soft_max_ਪਿੱਠ_f32_ਅੰਕੜੇ, "main", 3, ਆਕਾਰ(vk_op_ਧੱਕੋ_constants), {1, 1, 1}, { device->subgroup_ਮਾਪ }, 1, ਸੱਚ);

    ggml_vk_create_pipeline(device, device->pipeline_soft_max_large1_f32,     "soft_max_large1_f32",     soft_max_large1_f32_len,     soft_max_large1_f32_ਅੰਕੜੇ,     "main", 6, ਆਕਾਰ(vk_op_soft_max_ਧੱਕੋ_constants), {1, 1, 1}, { 128, 4 }, 1, ਸੱਚ);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_large2_f32,     "soft_max_large2_f32",     soft_max_large2_f32_len,     soft_max_large2_f32_ਅੰਕੜੇ,     "main", 6, ਆਕਾਰ(vk_op_soft_max_ਧੱਕੋ_constants), {1, 1, 1}, { 128, 4 }, 1, ਸੱਚ);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_large3_f32,     "soft_max_large3_f32",     soft_max_large3_f32_len,     soft_max_large3_f32_ਅੰਕੜੇ,     "main", 6, ਆਕਾਰ(vk_op_soft_max_ਧੱਕੋ_constants), {1, 1, 1}, { 128, 4 }, 1, ਸੱਚ);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_large1_f32_f16, "soft_max_large1_f32_f16", soft_max_large1_f32_f16_len, soft_max_large1_f32_f16_ਅੰਕੜੇ, "main", 6, ਆਕਾਰ(vk_op_soft_max_ਧੱਕੋ_constants), {1, 1, 1}, { 128, 4 }, 1, ਸੱਚ);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_large2_f32_f16, "soft_max_large2_f32_f16", soft_max_large2_f32_f16_len, soft_max_large2_f32_f16_ਅੰਕੜੇ, "main", 6, ਆਕਾਰ(vk_op_soft_max_ਧੱਕੋ_constants), {1, 1, 1}, { 128, 4 }, 1, ਸੱਚ);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_large3_f32_f16, "soft_max_large3_f32_f16", soft_max_large3_f32_f16_len, soft_max_large3_f32_f16_ਅੰਕੜੇ, "main", 6, ਆਕਾਰ(vk_op_soft_max_ਧੱਕੋ_constants), {1, 1, 1}, { 128, 4 }, 1, ਸੱਚ);

    ggml_vk_create_pipeline(device, device->pipeline_rope_norm_f32, "rope_norm_f32", rope_norm_f32_len, rope_norm_f32_ਅੰਕੜੇ, "main", 5, ਆਕਾਰ(vk_op_rope_ਧੱਕੋ_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_neox_f32, "rope_neox_f32", rope_neox_f32_len, rope_neox_f32_ਅੰਕੜੇ, "main", 5, ਆਕਾਰ(vk_op_rope_ਧੱਕੋ_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_multi_f32, "rope_multi_f32", rope_multi_f32_len, rope_multi_f32_ਅੰਕੜੇ, "main", 5, ਆਕਾਰ(vk_op_rope_ਧੱਕੋ_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_vision_f32, "rope_vision_f32", rope_vision_f32_len, rope_vision_f32_ਅੰਕੜੇ, "main", 5, ਆਕਾਰ(vk_op_rope_ਧੱਕੋ_constants), {1, 512, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_rope_norm_f16, "rope_norm_f16", rope_norm_f16_len, rope_norm_f16_ਅੰਕੜੇ, "main", 5, ਆਕਾਰ(vk_op_rope_ਧੱਕੋ_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_neox_f16, "rope_neox_f16", rope_neox_f16_len, rope_neox_f16_ਅੰਕੜੇ, "main", 5, ਆਕਾਰ(vk_op_rope_ਧੱਕੋ_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_multi_f16, "rope_multi_f16", rope_multi_f16_len, rope_multi_f16_ਅੰਕੜੇ, "main", 5, ਆਕਾਰ(vk_op_rope_ਧੱਕੋ_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_vision_f16, "rope_vision_f16", rope_vision_f16_len, rope_vision_f16_ਅੰਕੜੇ, "main", 5, ਆਕਾਰ(vk_op_rope_ਧੱਕੋ_constants), {1, 512, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_rope_norm_f32_f16, "rope_norm_f32_f16", rope_norm_f32_f16_len, rope_norm_f32_f16_ਅੰਕੜੇ, "main", 5, ਆਕਾਰ(vk_op_rope_ਧੱਕੋ_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_neox_f32_f16, "rope_neox_f32_f16", rope_neox_f32_f16_len, rope_neox_f32_f16_ਅੰਕੜੇ, "main", 5, ਆਕਾਰ(vk_op_rope_ਧੱਕੋ_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_multi_f32_f16, "rope_multi_f32_f16", rope_multi_f32_f16_len, rope_multi_f32_f16_ਅੰਕੜੇ, "main", 5, ਆਕਾਰ(vk_op_rope_ਧੱਕੋ_constants), {1, 512, 1}, {}, 1);

    ਲਈ (uint32_t i = 0; i < ਅੰਕ_argsort_pipelines; ++i) {
        uint32_t ਖੰਡ_ਮਾਪ_2 = 1u << std::min(i, device->max_workgroup_ਮਾਪ_log2);
        ਜੇ (i <= device->max_workgroup_ਮਾਪ_log2 &&
            2 * ਆਕਾਰ(ਪੂਰਨ) * ਖੰਡ_ਮਾਪ_2 <= device->properties.limits.max_Compute_Shared_ਸਮ੍ਰਿਤੀ_ਮਾਪ) {
            ਅਚਲ uint32_t NCOLS_PADDED_LOG2 = i;
            ggml_vk_create_pipeline2(device, device->pipeline_argsort_f32[i], "argsort_f32_"+std::to_string(i), argsort_f32_len, argsort_f32_ਅੰਕੜੇ, "main", 3, ਆਕਾਰ(vk_op_argsort_ਧੱਕੋ_constants), {ਖੰਡ_ਮਾਪ_2, 1, 1}, {ਖੰਡ_ਮਾਪ_2, NCOLS_PADDED_LOG2}, 1, ਸੱਚ);
        }
        ਅਚਲ uint32_t WG_UNROLL_ਗੁਣਕ = ਖੰਡ_ਮਾਪ_2 > 1 ? 2 : 1;
        ਖੰਡ_ਮਾਪ_2 /= WG_UNROLL_ਗੁਣਕ;
        ggml_vk_create_pipeline2(device, device->pipeline_argsort_large_f32[i], "argsort_large_f32_"+std::to_string(i), argsort_large_f32_len, argsort_large_f32_ਅੰਕੜੇ, "main", 3, ਆਕਾਰ(vk_op_argsort_ਧੱਕੋ_constants), {ਖੰਡ_ਮਾਪ_2 * WG_UNROLL_ਗੁਣਕ, 1, 1}, {ਖੰਡ_ਮਾਪ_2, WG_UNROLL_ਗੁਣਕ}, 1, ਸੱਚ);
    }

    ਲਈ (uint32_t i = 0; i < ਅੰਕ_topk_pipelines; ++i) {
        ਅਚਲ uint32_t ਖੰਡ_ਮਾਪ_2 = 1u << i;
        ਅਚਲ uint32_t NCOLS_PADDED_LOG2 = i;
        ਜੇ (i <= device->max_workgroup_ਮਾਪ_log2) {
            uint32_t nary_shmem = 2 * ਆਕਾਰ(ਪੂਰਨ) * ਖੰਡ_ਮਾਪ_2 +
                                  ਆਕਾਰ(ਪੂਰਨ) * device->subgroup_ਮਾਪ +
                                  2 * ਆਕਾਰ(ਪੂਰਨ) +
                                  2 * (ਖੰਡ_ਮਾਪ_2 / device->subgroup_ਮਾਪ) * ਆਕਾਰ(ਪੂਰਨ);
            ਜੇ (device->subgroup_arithmetic && device->subgroup_require_ਭਰਿਆ_support && device->subgroup_shuffle && device->subgroup_ballot &&
                nary_shmem <= device->properties.limits.max_Compute_Shared_ਸਮ੍ਰਿਤੀ_ਮਾਪ) {
                ggml_vk_create_pipeline2(device, device->pipeline_topk_f32[i], "topk_f32_"+std::to_string(i), topk_nary_ਖੋਜ_f32_len, topk_nary_ਖੋਜ_f32_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_topk_ਧੱਕੋ_constants), {ਖੰਡ_ਮਾਪ_2, 1, 1}, {ਖੰਡ_ਮਾਪ_2, device->subgroup_ਮਾਪ, device->subgroup_ਮਾਪ_log2}, 1, ਸੱਚ, ਸੱਚ, device->subgroup_ਮਾਪ);
            } ਨਹੀਂਤਾਂ ਜੇ (2 * ਆਕਾਰ(ਪੂਰਨ) * ਖੰਡ_ਮਾਪ_2 <= device->properties.limits.max_Compute_Shared_ਸਮ੍ਰਿਤੀ_ਮਾਪ) {
                ggml_vk_create_pipeline2(device, device->pipeline_topk_f32[i], "topk_f32_"+std::to_string(i), topk_argsort_f32_len, topk_argsort_f32_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_topk_ਧੱਕੋ_constants), {ਖੰਡ_ਮਾਪ_2, 1, 1}, {ਖੰਡ_ਮਾਪ_2, NCOLS_PADDED_LOG2}, 1, ਸੱਚ);
            }
        }
    }

    // large-k fallback: one workgroup per row, radix-select instead of a full sort. The QSA
    // variant (spec constant 1) additionally gathers the qwen4 indexer input on the fly.
    {
        ਅਚਲ uint32_t ਖੰਡ_ਮਾਪ_2 = 1u << std::min(10u, device->max_workgroup_ਮਾਪ_log2);
        ggml_vk_create_pipeline2(device, device->pipeline_topk_radix_f32, "topk_radix_f32", topk_radix_select_f32_len, topk_radix_select_f32_ਅੰਕੜੇ, "main", 5, ਆਕਾਰ(vk_op_topk_radix_ਧੱਕੋ_constants), {ਖੰਡ_ਮਾਪ_2, 1, 1}, {ਖੰਡ_ਮਾਪ_2, 0}, 1, ਸੱਚ);
        ggml_vk_create_pipeline2(device, device->pipeline_topk_radix_qsa, "topk_radix_qsa", topk_radix_select_f32_len, topk_radix_select_f32_ਅੰਕੜੇ, "main", 5, ਆਕਾਰ(vk_op_topk_radix_ਧੱਕੋ_constants), {ਖੰਡ_ਮਾਪ_2, 1, 1}, {ਖੰਡ_ਮਾਪ_2, 1}, 1, ਸੱਚ);
    }

    ggml_vk_create_pipeline(device, device->pipeline_argmax_f32, "argmax_f32", argmax_f32_len, argmax_f32_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_ਧੱਕੋ_constants), {1, 1, 1}, { device->subgroup_ਮਾਪ }, 1);

    ggml_vk_create_pipeline(device, device->pipeline_ਜੋੜ_ਪੰਕਤੀਆਂ_f32, "sum_rows_f32", ਜੋੜ_ਪੰਕਤੀਆਂ_f32_len, ਜੋੜ_ਪੰਕਤੀਆਂ_f32_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_ਜੋੜ_ਪੰਕਤੀਆਂ_ਧੱਕੋ_constants), {1, 1, 1}, { device->subgroup_ਮਾਪ }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cross_entropy_loss_f32, "cross_entropy_loss_f32", cross_entropy_loss_f32_len, cross_entropy_loss_f32_ਅੰਕੜੇ, "main", 3, ਆਕਾਰ(vk_op_ਧੱਕੋ_constants), {1, 1, 1}, { device->subgroup_ਮਾਪ }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cross_entropy_loss_f32_wg512, "cross_entropy_loss_f32_wg512", cross_entropy_loss_f32_len, cross_entropy_loss_f32_ਅੰਕੜੇ, "main", 3, ਆਕਾਰ(vk_op_ਧੱਕੋ_constants), {1, 1, 1}, { 512 }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cross_entropy_loss_ਪਿੱਠ_f32, "cross_entropy_loss_back_f32", cross_entropy_loss_ਪਿੱਠ_f32_len, cross_entropy_loss_ਪਿੱਠ_f32_ਅੰਕੜੇ, "main", 4, ਆਕਾਰ(vk_op_ਧੱਕੋ_constants), {1, 1, 1}, { device->subgroup_ਮਾਪ }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cross_entropy_loss_ਪਿੱਠ_f32_wg512, "cross_entropy_loss_back_f32_wg512", cross_entropy_loss_ਪਿੱਠ_f32_len, cross_entropy_loss_ਪਿੱਠ_f32_ਅੰਕੜੇ, "main", 4, ਆਕਾਰ(vk_op_ਧੱਕੋ_constants), {1, 1, 1}, { 512 }, 1);
    // Intel Windows driver in range [32.0.101.8509, 32.0.101.8860) will crash when using fwht kernels so we gate that here
    ਅਚਲ ਤਾਰਕਿਕ can_use_fwht = device->driver_id != vk::DriverId::eIntelProprietaryWindows ||
        !ggml_vk_intel_windows_driver_ਵਿੱਚ_range(device->properties.driver_ਸੰਸਕਰਣ_2, 101, 8509, 101, 8860);
    ਜੇ (can_use_fwht && device->subgroup_basic && device->subgroup_shuffle) {
        ਪੂਰਨ idx = 0;
        ਲਈ (uint32_t n : {64, 128, 256, 512}) {
            ਜੇ (device->subgroup_ਮਾਪ <= n) {
                ggml_vk_create_pipeline(device, device->pipeline_fwht_f32[idx], "fwht_f32", fwht_f32_len, fwht_f32_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_fwht_ਧੱਕੋ_constants), {1, 1, 1}, { device->subgroup_ਮਾਪ, n }, 1, ਸੱਚ, ਸੱਚ, device->subgroup_ਮਾਪ);
            }
            ++idx;
        }
    } ਨਹੀਂਤਾਂ ਜੇ (can_use_fwht) {
        ਪੂਰਨ idx = 0;
        ਲਈ (uint32_t n : {64, 128, 256, 512}) {
            ਅਚਲ uint32_t ਖੰਡ_ਮਾਪ_3 = std::min(device->subgroup_ਮਾਪ, n);
            ggml_vk_create_pipeline(device, device->pipeline_fwht_f32[idx], "fwht_shmem_f32", fwht_shmem_f32_len, fwht_shmem_f32_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_fwht_ਧੱਕੋ_constants), {1, 1, 1}, { ਖੰਡ_ਮਾਪ_3, n }, 1);
            ++idx;
        }
    }

    ਅਚਲ uint32_t cumsum_elem_ਪ੍ਰਤੀ_thread = (device->vendor_id == VK_VENDOR_ID_AMD || device->vendor_id == VK_VENDOR_ID_INTEL) ? 2 : 4;
    ggml_vk_create_pipeline(device, device->pipeline_cumsum_f32,       "cumsum_f32", cumsum_f32_len, cumsum_f32_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_ਜੋੜ_ਪੰਕਤੀਆਂ_ਧੱਕੋ_constants), {1, 1, 1}, { 256, device->subgroup_ਮਾਪ, cumsum_elem_ਪ੍ਰਤੀ_thread }, 1, ਸੱਚ, ਸੱਚ, device->subgroup_ਮਾਪ);
    ggml_vk_create_pipeline(device, device->pipeline_cumsum_small_f32, "cumsum_f32", cumsum_f32_len, cumsum_f32_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_ਜੋੜ_ਪੰਕਤੀਆਂ_ਧੱਕੋ_constants), {1, 1, 1}, { 128, device->subgroup_ਮਾਪ, 1 }, 1, ਸੱਚ, ਸੱਚ, device->subgroup_ਮਾਪ);
    ggml_vk_create_pipeline(device, device->pipeline_cumsum_multipass1_f32, "cumsum_multipass1_f32", cumsum_multipass1_f32_len, cumsum_multipass1_f32_ਅੰਕੜੇ, "main", 3, ਆਕਾਰ(vk_op_ਜੋੜ_ਪੰਕਤੀਆਂ_ਧੱਕੋ_constants), {256, 1, 1}, { 256, device->subgroup_ਮਾਪ }, 1, ਸੱਚ, ਸੱਚ, device->subgroup_ਮਾਪ);
    ggml_vk_create_pipeline(device, device->pipeline_cumsum_multipass2_f32, "cumsum_multipass2_f32", cumsum_multipass2_f32_len, cumsum_multipass2_f32_ਅੰਕੜੇ, "main", 3, ਆਕਾਰ(vk_op_ਜੋੜ_ਪੰਕਤੀਆਂ_ਧੱਕੋ_constants), {256, 1, 1}, { 256, device->subgroup_ਮਾਪ }, 1, ਸੱਚ, ਸੱਚ, device->subgroup_ਮਾਪ);

    ggml_vk_create_pipeline(device, device->pipeline_ਗਿਣਤੀ_equal_i32, "count_equal_i32", ਗਿਣਤੀ_equal_i32_len, ਗਿਣਤੀ_equal_i32_ਅੰਕੜੇ, "main", 3, ਆਕਾਰ(vk_op_ਧੱਕੋ_constants), {512, 1, 1}, { device->subgroup_ਮਾਪ }, 1);

    ਜੇ (device->subgroup_arithmetic && device->subgroup_vote && device->subgroup_require_ਭਰਿਆ_support) {
        ggml_vk_create_pipeline(device, device->pipeline_ਗਿਣਤੀ_experts, "count_experts", ਗਿਣਤੀ_experts_subgroup_len, ਗਿਣਤੀ_experts_subgroup_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_ਗਿਣਤੀ_experts_ਧੱਕੋ_constants), {1, 1, 1}, {}, 1, ਸੱਚ, ਸੱਚ);
    } ਨਹੀਂਤਾਂ {
        ggml_vk_create_pipeline(device, device->pipeline_ਗਿਣਤੀ_experts, "count_experts", ਗਿਣਤੀ_experts_len, ਗਿਣਤੀ_experts_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_ਗਿਣਤੀ_experts_ਧੱਕੋ_constants), {1, 1, 1}, {}, 1, ਸੱਚ);
    }

    // comb holds a token's 4x4 matrix in one 16-lane slice of a subgroup, so it
    // needs at least 16 lanes, pinned to a known size.
    ਜੇ (device->subgroup_basic && device->subgroup_shuffle && device->subgroup_require_ਭਰਿਆ_support && device->subgroup_ਮਾਪ >= 16) {
        ਅਚਲ uint32_t ਟੋਕਨ_ਪ੍ਰਤੀ_workgroup = 4 * (device->subgroup_ਮਾਪ / 16);
        ggml_vk_create_pipeline(device, device->pipeline_dsv4_hc_comb_f32, "dsv4_hc_comb_f32", dsv4_hc_comb_f32_len, dsv4_hc_comb_f32_ਅੰਕੜੇ, "main", 4, ਆਕਾਰ(vk_op_dsv4_hc_comb_ਧੱਕੋ_constants), {ਟੋਕਨ_ਪ੍ਰਤੀ_workgroup, 1, 1}, { device->subgroup_ਮਾਪ }, 1, ਸੱਚ, ਸੱਚ, device->subgroup_ਮਾਪ);
    }

    ggml_vk_create_pipeline(device, device->pipeline_dsv4_hc_pre_f32,        "dsv4_hc_pre_f32",        dsv4_hc_pre_f32_len,  dsv4_hc_pre_f32_ਅੰਕੜੇ,  "main", 3, ਆਕਾਰ(vk_op_dsv4_hc_pre_ਧੱਕੋ_constants),  {256, 1, 1}, { 256, 0 }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dsv4_hc_pre_gated_f32,  "dsv4_hc_pre_gated_f32",  dsv4_hc_pre_f32_len,  dsv4_hc_pre_f32_ਅੰਕੜੇ,  "main", 3, ਆਕਾਰ(vk_op_dsv4_hc_pre_ਧੱਕੋ_constants),  {256, 1, 1}, { 256, 1 }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dsv4_hc_post_f32,       "dsv4_hc_post_f32",       dsv4_hc_post_f32_len, dsv4_hc_post_f32_ਅੰਕੜੇ, "main", 5, ਆਕਾਰ(vk_op_dsv4_hc_post_ਧੱਕੋ_constants), {256, 1, 1}, { 256, 1 }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dsv4_hc_post_nocomb_f32,"dsv4_hc_post_nocomb_f32",dsv4_hc_post_f32_len, dsv4_hc_post_f32_ਅੰਕੜੇ, "main", 5, ਆਕਾਰ(vk_op_dsv4_hc_post_ਧੱਕੋ_constants), {256, 1, 1}, { 256, 0 }, 1);

    ਲਈ (ਸਵੈਚਾਲਿਤ &s : device->pipeline_solve_tri_f32) {
        ਅਚਲ vk_solve_tri_pipeline_state &state = s.ਪਹਿਲਾ;

        // Max number of rows to load at a time, limited by shared memory
        ਅਚਲ uint32_t batch_N = device->properties.limits.max_Compute_Shared_ਸਮ੍ਰਿਤੀ_ਮਾਪ / ((state.N + state.K) * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
        // Need at least K invocations, and prefer a minimum of 128 to spread out loading shared memory
        ਅਚਲ uint32_t ਖੰਡ_ਮਾਪ_3 = std::max(128u, 1u << (uint32_t)ceilf(log2f(ਦਸ਼ਮਲਵ(state.K))));

        ggml_vk_create_pipeline(
            device, s.second, "solve_tri_f32",
            solve_tri_f32_len, solve_tri_f32_ਅੰਕੜੇ, "main", 3,
            ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {1, 1, 1}, { 0, state.N, state.K, batch_N, ਖੰਡ_ਮਾਪ_3 }, 1, ਸੱਚ);
    }

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ IM2_ਖਾਨਾ(bda) \
    ggml_vk_create_pipeline(device, device->pipeline_im2col_f32, "im2col_f32", im2col_f32 ## bda ## _len, im2col_f32 ## bda ## ਅੰਕੜੇ_2, "main", 2, ਆਕਾਰ(vk_op_im2col_ਧੱਕੋ_constants), {512, 1, 1}, { device->subgroup_ਮਾਪ }, 1, ਸੱਚ);   \
    ggml_vk_create_pipeline(device, device->pipeline_im2col_3d_f32, "im2col_3d_f32", im2col_3d_f32 ## bda ## _len, im2col_3d_f32 ## bda ## ਅੰਕੜੇ_2, "main", 2, ਆਕਾਰ(vk_op_im2col_3d_ਧੱਕੋ_constants), {512, 1, 1}, { 512 }, 1, ਸੱਚ);      \
    ggml_vk_create_pipeline(device, device->pipeline_im2col_f32_f16, "im2col_f32_f16", im2col_f32_f16 ## bda ## _len, im2col_f32_f16 ## bda ## ਅੰਕੜੇ_2, "main", 2, ਆਕਾਰ(vk_op_im2col_ਧੱਕੋ_constants), {512, 1, 1}, { device->subgroup_ਮਾਪ }, 1, ਸੱਚ);   \
    ggml_vk_create_pipeline(device, device->pipeline_im2col_3d_f32_f16, "im2col_3d_f32_f16", im2col_3d_f32_f16 ## bda ## _len, im2col_3d_f32_f16 ## bda ## ਅੰਕੜੇ_2, "main", 2, ਆਕਾਰ(vk_op_im2col_3d_ਧੱਕੋ_constants), {512, 1, 1}, { 512 }, 1, ਸੱਚ);
    ਜੇ (device->shader_int64 && device->ਬਫਰ_device_ਪਤਾ) {
        IM2_ਖਾਨਾ(_bda)
    } ਨਹੀਂਤਾਂ {
        IM2_ਖਾਨਾ()
    }

    ggml_vk_create_pipeline(device, device->pipeline_timestep_embedding_f32, "timestep_embedding_f32", timestep_embedding_f32_len, timestep_embedding_f32_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_timestep_embedding_ਧੱਕੋ_constants), {256, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_conv_transpose_1d_f32, "conv_transpose_1d_f32", conv_transpose_1d_f32_len, conv_transpose_1d_f32_ਅੰਕੜੇ, "main", 3, ਆਕਾਰ(vk_op_conv_transpose_1d_ਧੱਕੋ_constants), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_col2im_1d_f32,  "col2im_1d_f32",  col2im_1d_f32_len,  col2im_1d_f32_ਅੰਕੜੇ,  "main", 2, ਆਕਾਰ(vk_op_col2im_1d_ਧੱਕੋ_constants), {256, 1, 1}, {}, 1, ਸੱਚ);
    ggml_vk_create_pipeline(device, device->pipeline_col2im_1d_f16,  "col2im_1d_f16",  col2im_1d_f16_len,  col2im_1d_f16_ਅੰਕੜੇ,  "main", 2, ਆਕਾਰ(vk_op_col2im_1d_ਧੱਕੋ_constants), {256, 1, 1}, {}, 1, ਸੱਚ);
    ggml_vk_create_pipeline(device, device->pipeline_col2im_1d_bf16, "col2im_1d_bf16", col2im_1d_bf16_len, col2im_1d_bf16_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_col2im_1d_ਧੱਕੋ_constants), {256, 1, 1}, {}, 1, ਸੱਚ);

    ggml_vk_create_pipeline(device, device->pipeline_ਬਾਹਰਲਾ_prod_f32, "out_prod_f32", ਬਾਹਰਲਾ_prod_f32_len, ਬਾਹਰਲਾ_prod_f32_ਅੰਕੜੇ, "main", 3, ਆਕਾਰ(vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants), {256, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_snake_f32,  "snake_f32",  snake_f32_len,  snake_f32_ਅੰਕੜੇ,  "main", 4, ਆਕਾਰ(vk_op_snake_ਧੱਕੋ_constants), {256, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_snake_f16,  "snake_f16",  snake_f16_len,  snake_f16_ਅੰਕੜੇ,  "main", 4, ਆਕਾਰ(vk_op_snake_ਧੱਕੋ_constants), {256, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_snake_bf16, "snake_bf16", snake_bf16_len, snake_bf16_ਅੰਕੜੇ, "main", 4, ਆਕਾਰ(vk_op_snake_ਧੱਕੋ_constants), {256, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_pool1d_f32, "pool1d_f32", pool1d_f32_len, pool1d_f32_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_pool1d_ਧੱਕੋ_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_pool2d_f32, "pool2d_f32", pool2d_f32_len, pool2d_f32_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_pool2d_ਧੱਕੋ_constants), {1, 1, 1}, {512}, 1);
    ਜੇ (device->subgroup_arithmetic && device->subgroup_require_ਭਰਿਆ_support) {
        ggml_vk_create_pipeline(device, device->pipeline_pool2d_subgroup_f32, "pool2d_subgroup_f32", pool2d_subgroup_f32_len, pool2d_subgroup_f32_ਅੰਕੜੇ, "main", 2, ਆਕਾਰ(vk_op_pool2d_ਧੱਕੋ_constants), {1, 1, 1}, {128}, 1, ਝੂਠ, ਸੱਚ, device->subgroup_ਮਾਪ);
    }

    ggml_vk_create_pipeline(device, device->pipeline_rwkv_wkv6_f32, "rwkv_wkv6_f32", rwkv_wkv6_f32_len, rwkv_wkv6_f32_ਅੰਕੜੇ, "main", 7, ਆਕਾਰ(vk_op_rwkv_wkv6_ਧੱਕੋ_constants), {1, 1, 1}, {device->subgroup_ਮਾਪ}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_rwkv_wkv7_f32, "rwkv_wkv7_f32", rwkv_wkv7_f32_len, rwkv_wkv7_f32_ਅੰਕੜੇ, "main", 8, ਆਕਾਰ(vk_op_rwkv_wkv7_ਧੱਕੋ_constants), {1, 1, 1}, {device->subgroup_ਮਾਪ}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_gated_ਰੇਖਿਕ_attn_f32, "gated_linear_attn_f32", gated_ਰੇਖਿਕ_attn_f32_len, gated_ਰੇਖਿਕ_attn_f32_ਅੰਕੜੇ, "main", 6, ਆਕਾਰ(vk_op_gated_ਰੇਖਿਕ_attn_ਧੱਕੋ_constants), {1, 1, 1}, {}, 1);

    ਲਈ (ggml_type k_type : lightning_indexer_k_types) {
        ਅਚਲ std::string name = "lightning_indexer_" + std::string(ggml_type_name(k_type)) + "_k_f32";
        ggml_vk_create_pipeline(device, device->pipeline_lightning_indexer_f32[k_type], name.c_str(), lightning_indexer_f32_len, lightning_indexer_f32_ਅੰਕੜੇ, "main", 5, ਆਕਾਰ(vk_op_lightning_indexer_ਧੱਕੋ_constants), {1, 1, 1}, {(uint32_t)k_type, fa_ਖੰਡ_ਬਾਈਟਾਂ(k_type)}, 1, ਸੱਚ);
    }

    {
        ਅਚਲ uint32_t gdn_sizes[] = {16, 32, 64, 128};
        ਅਚਲ ਅੱਖਰ * gdn_ਨਾਂ[][2] = {
            {"gated_delta_net_f32_d16",     "gated_delta_net_f32_d16_kda"},
            {"gated_delta_net_f32_d32",     "gated_delta_net_f32_d32_kda"},
            {"gated_delta_net_f32_d64",     "gated_delta_net_f32_d64_kda"},
            {"gated_delta_net_f32_d128",    "gated_delta_net_f32_d128_kda"},
        };
        ਲਈ (uint32_t si = 0; si < 4; si++) {
            ਅਚਲ uint32_t S_V = gdn_sizes[si];
            GGML_ASSERT(is_pow2(S_V));

            // Intel Xe regresses at SIMD32 for this scan; prefer a narrower subgroup.
            uint32_t gdn_subgroup_ਮਾਪ = device->subgroup_ਮਾਪ;
            ਜੇ (device->vendor_id == VK_VENDOR_ID_INTEL && device->subgroup_ਮਾਪ_control &&
                device->subgroup_min_ਮਾਪ <= 16u && device->subgroup_max_ਮਾਪ >= 16u) {
                gdn_subgroup_ਮਾਪ = 16u;
            }
            uint32_t lanes_ਪ੍ਰਤੀ_ਕਾਲਮ;
            ਜੇ (device->vendor_id == VK_VENDOR_ID_INTEL) {
                // Intel Xe: full-width reduction (min rows/lane) ~10x over the COLS_PER_WG=8 rule.
                lanes_ਪ੍ਰਤੀ_ਕਾਲਮ = std::min(gdn_subgroup_ਮਾਪ, S_V);
            } ਨਹੀਂਤਾਂ ਜੇ (S_V >= 128u && device->subgroup_clustered) {
                // COLS_PER_WG=8: measured optimum (Ampere sg32->4, Vega20 sg64->8).
                lanes_ਪ੍ਰਤੀ_ਕਾਲਮ = std::max(1u, gdn_subgroup_ਮਾਪ / 8u);
            } ਨਹੀਂਤਾਂ {
                // Use largest power-of-two that divides both S_V and subgroup_size so that
                // (1) S_V % lanes_per_column == 0 and (2) S_V % (subgroup_size / lanes_per_column) == 0.
                // This means we don't need extra bounds checking logic in the shader.
                lanes_ਪ੍ਰਤੀ_ਕਾਲਮ = std::min(S_V, gdn_subgroup_ਮਾਪ);
            }

            // gated_delta_net.comp relies on S_V % COLS_PER_WG == 0 and
            // S_V % LANES_PER_COLUMN == 0 to avoid bounds checks.
            ਜਦੋਂ (lanes_ਪ੍ਰਤੀ_ਕਾਲਮ > 1u) {
                ਅਚਲ ਤਾਰਕਿਕ valid_lanes = (gdn_subgroup_ਮਾਪ % lanes_ਪ੍ਰਤੀ_ਕਾਲਮ) == 0 &&
                                         (S_V % lanes_ਪ੍ਰਤੀ_ਕਾਲਮ) == 0;
                ਅਚਲ uint32_t ਖਾਨੇ_ਗਿਣਤੀ_ਪ੍ਰਤੀ_wg = valid_lanes ? gdn_subgroup_ਮਾਪ / lanes_ਪ੍ਰਤੀ_ਕਾਲਮ : 0;
                ਜੇ (valid_lanes && ਖਾਨੇ_ਗਿਣਤੀ_ਪ੍ਰਤੀ_wg > 0 && (S_V % ਖਾਨੇ_ਗਿਣਤੀ_ਪ੍ਰਤੀ_wg) == 0) {
                    ਰੋਕੋ;
                }
                lanes_ਪ੍ਰਤੀ_ਕਾਲਮ >>= 1u;
            }

            GGML_ASSERT((gdn_subgroup_ਮਾਪ % lanes_ਪ੍ਰਤੀ_ਕਾਲਮ) == 0);
            GGML_ASSERT((S_V % lanes_ਪ੍ਰਤੀ_ਕਾਲਮ) == 0);
            GGML_ASSERT((S_V % (gdn_subgroup_ਮਾਪ / lanes_ਪ੍ਰਤੀ_ਕਾਲਮ)) == 0);

            ਅਚਲ ਤਾਰਕਿਕ need_partial_subgroup_reduce = lanes_ਪ੍ਰਤੀ_ਕਾਲਮ != 1u && lanes_ਪ੍ਰਤੀ_ਕਾਲਮ < gdn_subgroup_ਮਾਪ;
            ਅਚਲ ਤਾਰਕਿਕ use_clustered_reduce = device->subgroup_arithmetic && device->subgroup_clustered && need_partial_subgroup_reduce;
            ਅਚਲ ਤਾਰਕਿਕ use_subgroup_reduce = device->subgroup_arithmetic && !need_partial_subgroup_reduce;
            ਅਚਲ ਤਾਰਕਿਕ use_subgroup_ops = use_clustered_reduce || use_subgroup_reduce;
            size_t gdn_len;
            ਅਚਲ ਖਾਲੀ * gdn_ਅੰਕੜੇ;
            ਜੇ (use_clustered_reduce) {
                gdn_len = gated_delta_net_f32_len;
                gdn_ਅੰਕੜੇ = (ਅਚਲ ਖਾਲੀ *)gated_delta_net_f32_ਅੰਕੜੇ;
            } ਨਹੀਂਤਾਂ ਜੇ (use_subgroup_reduce) {
                gdn_len = gated_delta_net_f32_nocluster_len;
                gdn_ਅੰਕੜੇ = (ਅਚਲ ਖਾਲੀ *)gated_delta_net_f32_nocluster_ਅੰਕੜੇ;
            } ਨਹੀਂਤਾਂ {
                gdn_len = gated_delta_net_f32_shmem_len;
                gdn_ਅੰਕੜੇ = (ਅਚਲ ਖਾਲੀ *)gated_delta_net_f32_shmem_ਅੰਕੜੇ;
            }

            ਅਚਲ uint32_t ਖਾਨੇ_ਗਿਣਤੀ_ਪ੍ਰਤੀ_wg = gdn_subgroup_ਮਾਪ / lanes_ਪ੍ਰਤੀ_ਕਾਲਮ;
            ਅਚਲ std::array<uint32_t, 3> wg_denoms = {1u, 1u, ਖਾਨੇ_ਗਿਣਤੀ_ਪ੍ਰਤੀ_wg};

            ਲਈ (uint32_t kda = 0; kda < 2; kda++) {
                ggml_vk_create_pipeline(device, device->pipeline_gated_delta_net[si][kda],
                    gdn_ਨਾਂ[si][kda], gdn_len, gdn_ਅੰਕੜੇ, "main", 7, ਆਕਾਰ(vk_op_gated_delta_net_ਧੱਕੋ_constants),
                    wg_denoms, {S_V, kda, gdn_subgroup_ਮਾਪ, lanes_ਪ੍ਰਤੀ_ਕਾਲਮ}, 1, ਸੱਚ, use_subgroup_ops, gdn_subgroup_ਮਾਪ);
            }
        }
    }

    ਜੇ (device->subgroup_arithmetic && device->subgroup_require_ਭਰਿਆ_support) {
        ggml_vk_create_pipeline(device, device->pipeline_ssm_ਜਾਂਚ_f32_d128, "ssm_scan_128_f32", ssm_ਜਾਂਚ_subgroup_f32_len, ssm_ਜਾਂਚ_subgroup_f32_ਅੰਕੜੇ, "main", 8, ਆਕਾਰ(vk_op_ssm_ਜਾਂਚ_ਧੱਕੋ_constants), {1, 1, 1}, {128, device->subgroup_ਮਾਪ}, 1, ਸੱਚ, ਸੱਚ);
        ggml_vk_create_pipeline(device, device->pipeline_ssm_ਜਾਂਚ_f32_d256, "ssm_scan_256_f32", ssm_ਜਾਂਚ_subgroup_f32_len, ssm_ਜਾਂਚ_subgroup_f32_ਅੰਕੜੇ, "main", 8, ਆਕਾਰ(vk_op_ssm_ਜਾਂਚ_ਧੱਕੋ_constants), {1, 1, 1}, {256, device->subgroup_ਮਾਪ}, 1, ਸੱਚ, ਸੱਚ);
    } ਨਹੀਂਤਾਂ {
        ggml_vk_create_pipeline(device, device->pipeline_ssm_ਜਾਂਚ_f32_d128, "ssm_scan_128_f32", ssm_ਜਾਂਚ_f32_len, ssm_ਜਾਂਚ_f32_ਅੰਕੜੇ, "main", 8, ਆਕਾਰ(vk_op_ssm_ਜਾਂਚ_ਧੱਕੋ_constants), {1, 1, 1}, {128, device->subgroup_ਮਾਪ, 16}, 1, ਸੱਚ, ਸੱਚ);
        ggml_vk_create_pipeline(device, device->pipeline_ssm_ਜਾਂਚ_f32_d256, "ssm_scan_256_f32", ssm_ਜਾਂਚ_f32_len, ssm_ਜਾਂਚ_f32_ਅੰਕੜੇ, "main", 8, ਆਕਾਰ(vk_op_ssm_ਜਾਂਚ_ਧੱਕੋ_constants), {1, 1, 1}, {256, device->subgroup_ਮਾਪ, 16}, 1, ਸੱਚ, ਸੱਚ);
    }

    ggml_vk_create_pipeline(device, device->pipeline_ssm_conv_f32,           "ssm_conv_f32",           ssm_conv_f32_len, ssm_conv_f32_ਅੰਕੜੇ, "main", 4, ਆਕਾਰ(vk_op_ssm_conv_ਧੱਕੋ_constants), {32, 16, 1}, {32, 16, 0, 0}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_ssm_conv_silu_f32,      "ssm_conv_silu_f32",      ssm_conv_f32_len, ssm_conv_f32_ਅੰਕੜੇ, "main", 4, ਆਕਾਰ(vk_op_ssm_conv_ਧੱਕੋ_constants), {32, 16, 1}, {32, 16, 0, 1}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_ssm_conv_bias_silu_f32, "ssm_conv_bias_silu_f32", ssm_conv_f32_len, ssm_conv_f32_ਅੰਕੜੇ, "main", 4, ਆਕਾਰ(vk_op_ssm_conv_ਧੱਕੋ_constants), {32, 16, 1}, {32, 16, 1, 1}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_opt_ਕਦਮ_adamw_f32, "opt_step_adamw_f32", opt_ਕਦਮ_adamw_f32_len, opt_ਕਦਮ_adamw_f32_ਅੰਕੜੇ, "main", 5, ਆਕਾਰ(vk_op_ਧੱਕੋ_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_opt_ਕਦਮ_sgd_f32, "opt_step_sgd_f32", opt_ਕਦਮ_sgd_f32_len, opt_ਕਦਮ_sgd_f32_ਅੰਕੜੇ, "main", 3, ਆਕਾਰ(vk_op_ਧੱਕੋ_constants), {512, 1, 1}, {}, 1);

    // conv2d, conv_transpose_2d, conv3d
    ਲਈ (uint32_t s = 0; s < CONV_SHAPE_ਗਿਣਤੀ; ++s) {
        // smaller WG for the small-tile fallback gives more concurrent WGs per SM
        uint32_t conv2d_WG_ਮਾਪ  = (s == CONV_SHAPE_64x32) ? 128 : 256;
        uint32_t use_collectives = 0;  // Enables subgroup ops for preventing the re-calculation of indices.
        uint32_t conv2d_TS_K     = (s == CONV_SHAPE_64x32) ? 4 : 8;
        uint32_t conv2d_SHMEM_PAD = 4;
        vk_conv_ਖੰਡ_ਮਾਪ conv2d_BS = vk_conv_ਖੰਡ_sizes[s];
        ਤਾਰਕਿਕ conv2d_UNROLL = ਸੱਚ;

#ਜੇ defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
        ਜੇ (device->coopmat2) {
            conv2d_SHMEM_PAD = 8; // 8 float16_t
        }
#ਜੇ_ਅੰਤ

        ਜੇ (device->vendor_id == VK_VENDOR_ID_INTEL) {
            conv2d_SHMEM_PAD = 0;
            conv2d_UNROLL = ਝੂਠ;
        } ਨਹੀਂਤਾਂ ਜੇ (device->vendor_id == VK_VENDOR_ID_AMD) {
            conv2d_SHMEM_PAD = device->architecture == vk_device_architecture::AMD_GCN ? 1 : 4;
            ਜੇ (s == CONV_SHAPE_128x128 && device->architecture != vk_device_architecture::AMD_GCN) {
                conv2d_UNROLL = ਝੂਠ;
            }
        }

        // Use collectives on pre-Turing NVIDIA GPUs and GCN AMD cards, which had slower integer math.
        ਤਾਰਕਿਕ allow_collectives_nv = device->vendor_id != VK_VENDOR_ID_NVIDIA ||
                                    device->architecture == vk_device_architecture::NVIDIA_PRE_TURING;
        ਤਾਰਕਿਕ allow_collectives_amd = device->vendor_id != VK_VENDOR_ID_AMD ||
                                     device->architecture == vk_device_architecture::AMD_GCN;

        ਜੇ (device->subgroup_shuffle &&
            device->vendor_id != VK_VENDOR_ID_INTEL &&   // Do not enable collectives on Intel, see PR 14316.
            allow_collectives_nv &&
            allow_collectives_amd) {
            use_collectives = 1;
            conv2d_BS.CRS   = std::min(
                device->subgroup_ਮਾਪ,
                conv2d_BS.CRS);  // CRS block size should be capped at subgroup size for correctness when shuffle is used.
        }

        // cm1 is used only when cm2 is unavailable; capped at 64x128 (due to shared memory size).
        // Requires 16x16x16 f16-acc since that's the fragment shape hard-coded in the shader.
        // Subgroup size must be 32 or 64 (to keep WG_SIZE sane) and we need
        // subgroup_size_control to force the driver to actually use it.
        ਤਾਰਕਿਕ conv2d_use_cm1 = ਝੂਠ;
#ਜੇ defined(VK_KHR_cooperative_ਮੈਟ੍ਰਿਕਸ) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
        conv2d_use_cm1 = !device->coopmat2 &&
                         device->coopmat_support && device->coopmat_support_16x16x16_f16acc &&
                         device->subgroup_ਮਾਪ_control &&
                         (device->subgroup_ਮਾਪ == 32 || device->subgroup_ਮਾਪ == 64) &&
                         s != CONV_SHAPE_128x128;
#ਜੇ_ਅੰਤ

        ਅਚਲ uint32_t conv2d_cm1_shmem_pad = 8;

        ਸਵੈਚਾਲਿਤ shmem_req = [&](uint32_t pad, ਤਾਰਕਿਕ csh_store, ਤਾਰਕਿਕ fp16_shmem) {
            ਅਚਲ uint32_t elem_ਮਾਪ = fp16_shmem ? (uint32_t)ਆਕਾਰ(uint16_t) : (uint32_t)ਆਕਾਰ(ਦਸ਼ਮਲਵ);
            ਅਚਲ uint32_t csh_elems = csh_store ? conv2d_BS.K * conv2d_BS.NPQ : 0u;
            ਵਾਪਸ (conv2d_BS.K * (conv2d_BS.CRS + pad) + conv2d_BS.CRS * (conv2d_BS.NPQ + pad) + csh_elems) * elem_ਮਾਪ;
        };

        // 2D, transpose-2D, and 3D conv use the same KxCRS @ CRSxNPQ shmem
        // layout. cm1 needs Csh for output, so check before applying cm1 params.
        ਜੇ (conv2d_use_cm1 && device->properties.limits.max_Compute_Shared_ਸਮ੍ਰਿਤੀ_ਮਾਪ < shmem_req(conv2d_cm1_shmem_pad, ਸੱਚ, ਸੱਚ)) {
            conv2d_use_cm1 = ਝੂਠ;
        }

        uint32_t conv2d_WM = 16, conv2d_WN = 16;  // cm1 subgroup tile, ignored otherwise
        ਜੇ (conv2d_use_cm1) {
            conv2d_SHMEM_PAD = conv2d_cm1_shmem_pad;
            // 16x16x16 fragments; pick WM/WN to keep WG_SIZE at 256
            // (i.e. 8 subgroups for sg=32, 4 subgroups for sg=64).
            ਅਚਲ ਤਾਰਕਿਕ sg64 = (device->subgroup_ਮਾਪ == 64);
            ਚੋਣ (s) {
                ਹਾਲਤ CONV_SHAPE_64x32:   conv2d_WM = sg64 ? 32 : 16; conv2d_WN = 16; ਰੋਕੋ;
                ਹਾਲਤ CONV_SHAPE_64x128:  conv2d_WM = 32; conv2d_WN = sg64 ? 64 : 32; ਰੋਕੋ;
                ਹਾਲਤ CONV_SHAPE_32x256:  conv2d_WM = sg64 ? 16 : 32; conv2d_WN = sg64 ? 128 : 32; ਰੋਕੋ;
                ਮੂਲ: ਰੋਕੋ;
            }
            ਅਚਲ uint32_t warps_M = conv2d_BS.K / conv2d_WM;
            ਅਚਲ uint32_t warps_N = conv2d_BS.NPQ / conv2d_WN;
            conv2d_WG_ਮਾਪ         = warps_M * warps_N * device->subgroup_ਮਾਪ;
        }

        // stage cm2 accumulator through shmem for coalesced global stores;
        // skipped on 128x128 where the extra Csh footprint hurts occupancy.
        // cm1 always uses the staged path.
        uint32_t conv2d_csh_store = (device->coopmat2 && s != CONV_SHAPE_128x128) ? 1u : 0u;
        ਜੇ (conv2d_use_cm1) {
            conv2d_csh_store = 1;
        }

        // shmem is fp16 on cm2/cm1 (matches Csh), fp32 on scalar
        ਅਚਲ ਤਾਰਕਿਕ conv2d_use_fp16_shmem = device->coopmat2 || conv2d_use_cm1;

        // shrink CRS if the non-cm1 config still doesn't fit
        ਜੇ (device->properties.limits.max_Compute_Shared_ਸਮ੍ਰਿਤੀ_ਮਾਪ < shmem_req(conv2d_SHMEM_PAD, conv2d_csh_store, conv2d_use_fp16_shmem)) {
            GGML_ASSERT(!conv2d_use_cm1);
            conv2d_BS.CRS = 8;
            ਜੇ (use_collectives) {
                conv2d_BS.CRS = std::min(device->subgroup_ਮਾਪ, conv2d_BS.CRS);
            }
            conv2d_csh_store = 0;
        }

        std::array<uint32_t, 3> wg_denoms = { conv2d_BS.K, 1, 1 };
        std::vector<uint32_t> spec_constants = { conv2d_WG_ਮਾਪ, conv2d_BS.K, conv2d_BS.CRS, conv2d_BS.NPQ, conv2d_TS_K, use_collectives, conv2d_SHMEM_PAD };

        // cm1 needs a fixed subgroup width to match the WG_SIZE we computed
        ਅਚਲ uint32_t conv2d_required_subgroup_ਮਾਪ = conv2d_use_cm1 ? device->subgroup_ਮਾਪ : 0;

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CREATE_CONV(name, type_suffix, spv_suffix) \
        ਲਈ (ਸਵੈਚਾਲਿਤ &c : device->pipeline_##name##type_suffix[s]) { \
            ਅਚਲ vk_conv2d_pipeline_state &state = c.ਪਹਿਲਾ;  \
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
                ਆਕਾਰ(vk_op_conv2d_ਧੱਕੋ_constants), wg_denoms, spec_constants_cpy, 1, ਸੱਚ, use_collectives || conv2d_required_subgroup_ਮਾਪ, conv2d_required_subgroup_ਮਾਪ);    \
        }
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CREATE_CONVS(spv_suffix) \
        CREATE_CONV(conv2d, _f32, spv_suffix) \
        CREATE_CONV(conv2d, _f16_f32, spv_suffix) \
        CREATE_CONV(conv_transpose_2d, _f32, spv_suffix) \
        CREATE_CONV(conv_transpose_2d, _f16_f32, spv_suffix)
#ਜੇ defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
        ਜੇ (device->coopmat2) {
            CREATE_CONVS(_cm2)
        } ਨਹੀਂਤਾਂ
#ਜੇ_ਅੰਤ
#ਜੇ defined(VK_KHR_cooperative_ਮੈਟ੍ਰਿਕਸ) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
        ਜੇ (conv2d_use_cm1) {
            CREATE_CONVS(_cm1)
        } ਨਹੀਂਤਾਂ
#ਜੇ_ਅੰਤ
        ਜੇ (conv2d_UNROLL) {
            CREATE_CONVS(_unroll)
        } ਨਹੀਂਤਾਂ {
            CREATE_CONVS( )
        }
#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ CREATE_CONV
#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ CREATE_CONVS

        std::vector<uint32_t> conv3d_spec_constants = { conv2d_WG_ਮਾਪ, conv2d_BS.K, conv2d_BS.CRS, conv2d_BS.NPQ, conv2d_TS_K, conv2d_SHMEM_PAD };
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CREATE_CONV3D(type_suffix, spv_suffix) \
        ਲਈ (ਸਵੈਚਾਲਿਤ &c : device->pipeline_conv3d##type_suffix[s]) { \
            ਅਚਲ vk_conv3d_pipeline_state &state = c.ਪਹਿਲਾ; \
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
                ਆਕਾਰ(vk_op_conv3d_ਧੱਕੋ_constants), wg_denoms, spec_constants_cpy, 1, ਸੱਚ, conv2d_required_subgroup_ਮਾਪ != 0, conv2d_required_subgroup_ਮਾਪ); \
        }
#ਜੇ defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
        ਜੇ (device->coopmat2) {
            CREATE_CONV3D(_f32, _cm2)
            CREATE_CONV3D(_f16_f32, _cm2)
        } ਨਹੀਂਤਾਂ
#ਜੇ_ਅੰਤ
#ਜੇ defined(VK_KHR_cooperative_ਮੈਟ੍ਰਿਕਸ) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
        ਜੇ (conv2d_use_cm1) {
            CREATE_CONV3D(_f32, _cm1)
            CREATE_CONV3D(_f16_f32, _cm1)
        } ਨਹੀਂਤਾਂ
#ਜੇ_ਅੰਤ
        ਜੇ (conv2d_UNROLL) {
            CREATE_CONV3D(_f32, _unroll)
            CREATE_CONV3D(_f16_f32, _unroll)
        } ਨਹੀਂਤਾਂ {
            CREATE_CONV3D(_f32, )
            CREATE_CONV3D(_f16_f32, )
        }
#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ CREATE_CONV3D
    }

    ggml_vk_create_pipeline(device, device->pipeline_conv2d_dw_whcn_f32, "conv2d_dw_whcn_f32", conv2d_dw_whcn_f32_len, conv2d_dw_whcn_f32_ਅੰਕੜੇ, "main", 3, ਆਕਾਰ(vk_op_conv2d_dw_ਧੱਕੋ_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_conv2d_dw_cwhn_f32, "conv2d_dw_cwhn_f32", conv2d_dw_cwhn_f32_len, conv2d_dw_cwhn_f32_ਅੰਕੜੇ, "main", 3, ਆਕਾਰ(vk_op_conv2d_dw_ਧੱਕੋ_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_conv2d_dw_whcn_f16_f32, "conv2d_dw_whcn_f16_f32", conv2d_dw_whcn_f16_f32_len, conv2d_dw_whcn_f16_f32_ਅੰਕੜੇ, "main", 3, ਆਕਾਰ(vk_op_conv2d_dw_ਧੱਕੋ_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_conv2d_dw_cwhn_f16_f32, "conv2d_dw_cwhn_f16_f32", conv2d_dw_cwhn_f16_f32_len, conv2d_dw_cwhn_f16_f32_ਅੰਕੜੇ, "main", 3, ਆਕਾਰ(vk_op_conv2d_dw_ਧੱਕੋ_constants), {512, 1, 1}, {}, 1);

    ਲਈ (uint32_t use_ਧੱਕੋ = 0; use_ਧੱਕੋ < 2; ++use_ਧੱਕੋ) {
        ਲਈ (uint32_t i = 0; i < ਅੰਕ_topk_moe_pipelines; ++i) {
            ggml_vk_create_pipeline2(device, device->pipeline_topk_moe[i][use_ਧੱਕੋ], "topk_moe_f32_"+std::to_string(i), topk_moe_f32_len, topk_moe_f32_ਅੰਕੜੇ, "main", 4, ਆਕਾਰ(vk_op_topk_moe_ਧੱਕੋ_constants), {1, 1, 1}, {device->subgroup_ਮਾਪ, 1u<<i, use_ਧੱਕੋ}, 1, ਸੱਚ, ਸੱਚ, device->subgroup_ਮਾਪ);
        }
    }

    // Drop compile_mutex so other threads can walk while we compile.
    compile_ਤਾਲਾ.unlock();

    // Compile what we claimed; create_pipeline_func reacquires compile_mutex
    // at the end to flip compile_pending/compiled and notify waiters.
    ਜੇ (has_claimed_task) {
        ਸਵੈਚਾਲਿਤ & task = claimed_task;
        ggml_vk_create_pipeline_func(device, task.pipeline, task.spv_ਮਾਪ, task.spv_ਅੰਕੜੇ,
                                     task.entrypoint, task.parameter_ਗਿਣਤੀ, task.wg_denoms,
                                     task.specialization_constants, task.disable_robustness,
                                     task.require_ਭਰਿਆ_subgroups, task.required_subgroup_ਮਾਪ);
    }

    // Another thread may be compiling the pipeline we need; block on it here.
    ਜੇ (wait_pipeline) {
        std::unique_lock<std::mutex> wait_ਤਾਲਾ(device->compile_mutex);
        device->compile_cv.wait(wait_ਤਾਲਾ, [&] {
            ਵਾਪਸ wait_pipeline->compiled.load();
        });
    }
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_vk_external_ਸਮ੍ਰਿਤੀ_host_usable(vk::DriverId driver_id) {
    // Disable external_memory_host until https://github.com/KhronosGroup/MoltenVK/pull/2622
    // is available in the Vulkan SDK.
    ਵਾਪਸ driver_id != vk::DriverId::eMoltenvk;
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_vk_host_ਬਫਰ_import_usable(uint32_t vendor_id) {
    // amdgpu re-faults all pages of every imported host allocation on each submission
    ਵਾਪਸ vendor_id != VK_VENDOR_ID_AMD;
}

vk_device ggml_vk_get_device(size_t idx) {
    VK_LOG_DEBUG("ggml_vk_get_device(" << idx << ")");

    ਸਥਿਰ std::mutex mutex;
    std::lock_guard<std::mutex> ਤਾਲਾ(mutex);

    ਜੇ (vk_instance.devices[idx] == ਖਾਲੀ_ਸੂਚਕ) {
        VK_LOG_DEBUG("Initializing new vk_device");
        vk_device device = std::make_shared<vk_device_struct>();

        device->ਸਮ੍ਰਿਤੀ_logger = std::unique_ptr<vk_ਸਮ੍ਰਿਤੀ_logger>(ਨਵਾਂ vk_ਸਮ੍ਰਿਤੀ_logger());

        size_t dev_ਅੰਕ = vk_instance.device_indices[idx];

        std::vector<vk::PhysicalDevice> physical_devices = vk_instance.instance.enumeratePhysicalDevices();

        ਜੇ (dev_ਅੰਕ >= physical_devices.size()) {
            std::cerr << "ggml_vulkan: Device with index " << dev_ਅੰਕ << " does not exist." << std::ਸਤਰ_ਅੰਤ;
            ਸੁੱਟੋ std::runtime_error("Device not found");
        }

        device->physical_device = physical_devices[dev_ਅੰਕ];
        ਅਚਲ std::vector<vk::ExtensionProperties> ext_props = device->physical_device.enumerateDeviceExtensionProperties();

        device->architecture = get_device_architecture(device->physical_device);

        ਅਚਲ ਅੱਖਰ* GGML_VK_PREFER_HOST_ਸਮ੍ਰਿਤੀ = getenv("GGML_VK_PREFER_HOST_MEMORY");
        device->prefer_host_ਸਮ੍ਰਿਤੀ = GGML_VK_PREFER_HOST_ਸਮ੍ਰਿਤੀ != ਖਾਲੀ_ਸੂਚਕ;

        ਅਚਲ ਅੱਖਰ* GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM = getenv("GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM");
        device->disable_host_visible_vidmem = GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM != ਖਾਲੀ_ਸੂਚਕ;

        ਅਚਲ ਅੱਖਰ* GGML_VK_ALLOW_SYSMEM_FALLBACK = getenv("GGML_VK_ALLOW_SYSMEM_FALLBACK");
        device->allow_sysmem_fallback = GGML_VK_ALLOW_SYSMEM_FALLBACK != ਖਾਲੀ_ਸੂਚਕ;

        ਅਚਲ ਅੱਖਰ* GGML_VK_DISABLE_ਗ੍ਰਾਫ_OPTIMIZE = getenv("GGML_VK_DISABLE_GRAPH_OPTIMIZE");
        device->disable_ਗ੍ਰਾਫ_optimize = GGML_VK_DISABLE_ਗ੍ਰਾਫ_OPTIMIZE != ਖਾਲੀ_ਸੂਚਕ;

        ਤਾਰਕਿਕ fp16_storage = ਝੂਠ;
        ਤਾਰਕਿਕ fp16_compute = ਝੂਠ;
        ਤਾਰਕਿਕ maintenance4_support = ਝੂਠ;
        ਤਾਰਕਿਕ sm_builtins = ਝੂਠ;
        ਤਾਰਕਿਕ amd_shader_core_properties2 = ਝੂਠ;
        ਤਾਰਕਿਕ pipeline_robustness = ਝੂਠ;
        ਤਾਰਕਿਕ coopmat2_support = ਝੂਠ;
        ਤਾਰਕਿਕ coopmat2_decode_ਸਦਿਸ਼_support = ਝੂਠ;
        ਤਾਰਕਿਕ pipeline_executable_properties_support = ਝੂਠ;
        ਤਾਰਕਿਕ internally_sync_support = ਝੂਠ;
        device->coopmat_support = ਝੂਠ;
        device->integer_dot_ਗੁਣਨਫਲ = ਝੂਠ;
        device->shader_64b_indexing = ਝੂਠ;
        ਤਾਰਕਿਕ bfloat16_support = ਝੂਠ;
        ਤਾਰਕਿਕ dot2_f16_support = ਝੂਠ;
        ਤਾਰਕਿਕ ocp_microscaling_extension = ਝੂਠ;
        ਤਾਰਕਿਕ shader_float8_extension = ਝੂਠ;

        ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ& properties : ext_props) {
            ਜੇ (strcmp("VK_KHR_maintenance4", properties.extensionName) == 0) {
                maintenance4_support = ਸੱਚ;
            } ਨਹੀਂਤਾਂ ਜੇ (strcmp("VK_KHR_16bit_storage", properties.extensionName) == 0) {
                fp16_storage = ਸੱਚ;
            } ਨਹੀਂਤਾਂ ਜੇ (strcmp("VK_KHR_shader_float16_int8", properties.extensionName) == 0) {
                fp16_compute = ਸੱਚ;
            } ਨਹੀਂਤਾਂ ਜੇ (strcmp("VK_NV_shader_sm_builtins", properties.extensionName) == 0) {
                sm_builtins = ਸੱਚ;
            } ਨਹੀਂਤਾਂ ਜੇ (strcmp("VK_AMD_shader_core_properties2", properties.extensionName) == 0) {
                amd_shader_core_properties2 = ਸੱਚ;
            } ਨਹੀਂਤਾਂ ਜੇ (strcmp("VK_EXT_pipeline_robustness", properties.extensionName) == 0) {
                pipeline_robustness = ਸੱਚ;
            } ਨਹੀਂਤਾਂ ਜੇ (strcmp("VK_EXT_subgroup_size_control", properties.extensionName) == 0) {
                device->subgroup_ਮਾਪ_control = ਸੱਚ;
#ਜੇ defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
            } ਨਹੀਂਤਾਂ ਜੇ (strcmp("VK_KHR_cooperative_matrix", properties.extensionName) == 0 &&
                       !getenv("GGML_VK_DISABLE_COOPMAT")) {
                device->coopmat_support = ਸੱਚ;
                device->coopmat_m = 0;
                device->coopmat_n = 0;
                device->coopmat_k = 0;
#ਜੇ_ਅੰਤ
#ਜੇ defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
            } ਨਹੀਂਤਾਂ ਜੇ (strcmp("VK_NV_cooperative_matrix2", properties.extensionName) == 0 &&
                       !getenv("GGML_VK_DISABLE_COOPMAT2")) {
                coopmat2_support = ਸੱਚ;
#ਜੇ_ਅੰਤ
            } ਨਹੀਂਤਾਂ ਜੇ (strcmp(VK_NV_COOPERATIVE_ਮੈਟ੍ਰਿਕਸ_DECODE_ਸਦਿਸ਼_EXTENSION_NAME, properties.extensionName) == 0 &&
                       !getenv("GGML_VK_DISABLE_COOPMAT2_DECODE_VECTOR")) {
                coopmat2_decode_ਸਦਿਸ਼_support = ਸੱਚ;
#ਜੇ defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
            } ਨਹੀਂਤਾਂ ਜੇ (strcmp("VK_KHR_shader_integer_dot_product", properties.extensionName) == 0 &&
                       !getenv("GGML_VK_DISABLE_INTEGER_DOT_PRODUCT")) {
                device->integer_dot_ਗੁਣਨਫਲ = ਸੱਚ;
#ਜੇ_ਅੰਤ
#ਜੇ defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
            } ਨਹੀਂਤਾਂ ਜੇ (strcmp("VK_KHR_shader_bfloat16", properties.extensionName) == 0 &&
                       !getenv("GGML_VK_DISABLE_BFLOAT16")) {
                bfloat16_support = ਸੱਚ;
#ਜੇ_ਅੰਤ
#ਜੇ defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT)
            } ਨਹੀਂਤਾਂ ਜੇ (strcmp(VK_EXT_SHADER_OCP_MICROSCALING_TYPES_EXTENSION_NAME, properties.extensionName) == 0) {
                ocp_microscaling_extension = ਸੱਚ;
#ਜੇ_ਅੰਤ
#ਜੇ defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
            } ਨਹੀਂਤਾਂ ਜੇ (strcmp(VK_EXT_SHADER_FLOAT8_EXTENSION_NAME, properties.extensionName) == 0) {
                shader_float8_extension = ਸੱਚ;
#ਜੇ_ਅੰਤ
            } ਨਹੀਂਤਾਂ ਜੇ (strcmp("VK_VALVE_shader_mixed_float_dot_product", properties.extensionName) == 0 &&
                       !getenv("GGML_VK_DISABLE_DOT2")) {
                dot2_f16_support = ਸੱਚ;
            } ਨਹੀਂਤਾਂ ਜੇ (strcmp("VK_KHR_pipeline_executable_properties", properties.extensionName) == 0) {
                pipeline_executable_properties_support = ਸੱਚ;
            } ਨਹੀਂਤਾਂ ਜੇ (strcmp("VK_EXT_memory_priority", properties.extensionName) == 0 &&
                       getenv("GGML_VK_ENABLE_MEMORY_PRIORITY")) {
                device->ਸਮ੍ਰਿਤੀ_ਤਰਜੀਹ = ਸੱਚ;
            } ਨਹੀਂਤਾਂ ਜੇ (strcmp("VK_EXT_external_memory_host", properties.extensionName) == 0) {
                device->external_ਸਮ੍ਰਿਤੀ_host = ਸੱਚ;
#ਜੇ defined(VK_EXT_shader_64bit_indexing)
            } ਨਹੀਂਤਾਂ ਜੇ (strcmp("VK_EXT_shader_64bit_indexing", properties.extensionName) == 0) {
                device->shader_64b_indexing = ਸੱਚ;
#ਜੇ_ਅੰਤ
            } ਨਹੀਂਤਾਂ ਜੇ (strcmp(VK_KHR_INTERNALLY_SYNCHRONIZED_QUEUES_EXTENSION_NAME, properties.extensionName) == 0) {
                internally_sync_support = ਸੱਚ;
            } ਨਹੀਂਤਾਂ ਜੇ (strcmp("VK_EXT_device_fault", properties.extensionName) == 0) {
                device->device_fault = ਸੱਚ;
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
        vk::Physical_Device_Subgroup_ਮਾਪ_Control_Properties_EXT subgroup_ਮਾਪ_control_props;
        vk::Physical_Device_Shader_Integer_Dot_ਗੁਣਨਫਲ_Properties_KHR shader_integer_dot_ਗੁਣਨਫਲ_props;
        vk::Physical_Device_External_ਸਮ੍ਰਿਤੀ_Host_Properties_EXT external_ਸਮ੍ਰਿਤੀ_host_props;

        props2.p_ਅਗਲਾ = &props3;
        props3.p_ਅਗਲਾ = &subgroup_props;
        subgroup_props.p_ਅਗਲਾ = &driver_props;
        driver_props.p_ਅਗਲਾ = &vk11_props;
        vk11_props.p_ਅਗਲਾ = &vk12_props;

        Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure * ਆਖਰੀ_struct = (Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure *)&vk12_props;

        ਜੇ (maintenance4_support) {
            ਆਖਰੀ_struct->p_ਅਗਲਾ = (Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure *)&props4;
            ਆਖਰੀ_struct = (Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure *)&props4;
        }
        ਜੇ (sm_builtins) {
            ਆਖਰੀ_struct->p_ਅਗਲਾ = (Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure *)&sm_props;
            ਆਖਰੀ_struct = (Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure *)&sm_props;
        }
        ਜੇ (amd_shader_core_properties2) {
            ਆਖਰੀ_struct->p_ਅਗਲਾ = (Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure *)&amd_shader_core_properties2_props;
            ਆਖਰੀ_struct = (Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure *)&amd_shader_core_properties2_props;
        }
        ਜੇ (device->subgroup_ਮਾਪ_control) {
            ਆਖਰੀ_struct->p_ਅਗਲਾ = (Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure *)&subgroup_ਮਾਪ_control_props;
            ਆਖਰੀ_struct = (Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure *)&subgroup_ਮਾਪ_control_props;
        }

#ਜੇ defined(VK_NV_cooperative_matrix2)
        vk::PhysicalDeviceCooperativeMatrix2PropertiesNV coopmat2_props;
        ਜੇ (coopmat2_support) {
            ਆਖਰੀ_struct->p_ਅਗਲਾ = (Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure *)&coopmat2_props;
            ਆਖਰੀ_struct = (Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure *)&coopmat2_props;
        }
#ਜੇ_ਅੰਤ

        ਜੇ (device->integer_dot_ਗੁਣਨਫਲ) {
            ਆਖਰੀ_struct->p_ਅਗਲਾ = (Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure *)&shader_integer_dot_ਗੁਣਨਫਲ_props;
            ਆਖਰੀ_struct = (Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure *)&shader_integer_dot_ਗੁਣਨਫਲ_props;
        }

        ਜੇ (device->external_ਸਮ੍ਰਿਤੀ_host) {
            ਆਖਰੀ_struct->p_ਅਗਲਾ = (Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure *)&external_ਸਮ੍ਰਿਤੀ_host_props;
            ਆਖਰੀ_struct = (Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure *)&external_ਸਮ੍ਰਿਤੀ_host_props;
        }

        device->physical_device.getProperties2(&props2);
        device->properties = props2.properties;
        device->vendor_id = device->properties.vendorID;
        device->driver_id = driver_props.driverID;

        device->external_ਸਮ੍ਰਿਤੀ_host = device->external_ਸਮ੍ਰਿਤੀ_host && ggml_vk_external_ਸਮ੍ਰਿਤੀ_host_usable(device->driver_id);

        // Implementing the async backend interfaces seems broken on older Intel HW,
        // see https://github.com/ggml-org/llama.cpp/issues/17302.
        device->support_async = (device->vendor_id != VK_VENDOR_ID_INTEL ||
                                 std::string(device->properties.deviceName.ਅੰਕੜੇ()).find("(DG1)") == std::string::npos) &&
                                getenv("GGML_VK_DISABLE_ASYNC") == ਖਾਲੀ_ਸੂਚਕ;

        ਜੇ (!device->support_async) {
            GGML_LOG_DEBUG("ggml_vulkan: WARNING: Async execution disabled on certain Intel devices.\n");
        }

        ਅਚਲ ਅੱਖਰ* GGML_VK_FORCE_MAX_ਵੰਡ_ਮਾਪ = getenv("GGML_VK_FORCE_MAX_ALLOCATION_SIZE");

        ਜੇ (GGML_VK_FORCE_MAX_ਵੰਡ_ਮਾਪ != ਖਾਲੀ_ਸੂਚਕ) {
            device->max_ਸਮ੍ਰਿਤੀ_ਵੰਡ_ਮਾਪ = std::stoull(GGML_VK_FORCE_MAX_ਵੰਡ_ਮਾਪ);
        } ਨਹੀਂਤਾਂ ਜੇ (maintenance4_support) {
            device->max_ਸਮ੍ਰਿਤੀ_ਵੰਡ_ਮਾਪ = std::min(props3.max_ਸਮ੍ਰਿਤੀ_ਵੰਡ_ਮਾਪ_2, props4.max_ਬਫਰ_ਮਾਪ_2);
        } ਨਹੀਂਤਾਂ {
            device->max_ਸਮ੍ਰਿਤੀ_ਵੰਡ_ਮਾਪ = props3.max_ਸਮ੍ਰਿਤੀ_ਵੰਡ_ਮਾਪ_2;
        }

        ਅਚਲ ਅੱਖਰ* GGML_VK_FORCE_MAX_ਬਫਰ_ਮਾਪ = getenv("GGML_VK_FORCE_MAX_BUFFER_SIZE");

        ਜੇ (GGML_VK_FORCE_MAX_ਬਫਰ_ਮਾਪ != ਖਾਲੀ_ਸੂਚਕ) {
            device->max_ਬਫਰ_ਮਾਪ = std::stoull(GGML_VK_FORCE_MAX_ਬਫਰ_ਮਾਪ);
        } ਨਹੀਂਤਾਂ ਜੇ (maintenance4_support) {
            device->max_ਬਫਰ_ਮਾਪ = props4.max_ਬਫਰ_ਮਾਪ_2;
        } ਨਹੀਂਤਾਂ {
            device->max_ਬਫਰ_ਮਾਪ = device->max_ਸਮ੍ਰਿਤੀ_ਵੰਡ_ਮਾਪ;
        }

        ਅਚਲ ਅੱਖਰ* GGML_VK_SUBALLOCATION_ਖੰਡ_ਮਾਪ = getenv("GGML_VK_SUBALLOCATION_BLOCK_SIZE");

        ਜੇ (GGML_VK_SUBALLOCATION_ਖੰਡ_ਮਾਪ != ਖਾਲੀ_ਸੂਚਕ) {
            device->suballocation_ਖੰਡ_ਮਾਪ = std::stoull(GGML_VK_SUBALLOCATION_ਖੰਡ_ਮਾਪ);
        } ਨਹੀਂਤਾਂ {
            // Limit batching of allocations to 1GB by default to avoid fragmentation issues
            device->suballocation_ਖੰਡ_ਮਾਪ = 1024*1024*1024;
        }
        device->suballocation_ਖੰਡ_ਮਾਪ = std::min(device->suballocation_ਖੰਡ_ਮਾਪ, device->max_ਸਮ੍ਰਿਤੀ_ਵੰਡ_ਮਾਪ);

        device->subgroup_ਮਾਪ = subgroup_props.subgroup_ਮਾਪ_2;
        device->subgroup_ਮਾਪ_log2 = uint32_t(log2f(ਦਸ਼ਮਲਵ(device->subgroup_ਮਾਪ)));
        device->uma = device->properties.deviceType == vk::PhysicalDeviceType::eIntegratedGpu;
        ਜੇ (sm_builtins) {
            device->shader_core_ਗਿਣਤੀ = sm_props.shaderSMCount;
        } ਨਹੀਂਤਾਂ ਜੇ (amd_shader_core_properties2) {
            device->shader_core_ਗਿਣਤੀ = amd_shader_core_properties2_props.ਸਰਗਰਮ_Compute_Unit_ਗਿਣਤੀ;
        } ਨਹੀਂਤਾਂ ਜੇ (device->vendor_id == VK_VENDOR_ID_INTEL) {
            device->shader_core_ਗਿਣਤੀ = ggml_vk_intel_shader_core_ਗਿਣਤੀ(device->physical_device);
        } ਨਹੀਂਤਾਂ {
            device->shader_core_ਗਿਣਤੀ = 0;
        }
        device->float_controls_rte_fp16 = vk12_props.shader_Rounding_ਬਹੁਲਕ_RTEFloat16;
        device->float_controls_denorm_preserve_fp16 = vk12_props.shaderDenormPreserveFloat16;

        device->subgroup_basic = (vk11_props.subgroupSupportedStages & vk::Shader_Stage_Flag_ਬਿੱਟ::eCompute) &&
                                 (vk11_props.subgroupSupportedOperations & vk::Subgroup_Feature_Flag_ਬਿੱਟ::eBasic);
        device->subgroup_arithmetic = (vk11_props.subgroupSupportedStages & vk::Shader_Stage_Flag_ਬਿੱਟ::eCompute) &&
                                      (vk11_props.subgroupSupportedOperations & vk::Subgroup_Feature_Flag_ਬਿੱਟ::eArithmetic);
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ __APPLE__
        // Workaround for subgroup arithmetic failing on MoltenVK with AMD GPUs (issue 15846)
        ਜੇ (device->vendor_id == VK_VENDOR_ID_AMD) {
            device->subgroup_arithmetic = ਝੂਠ;
        }
#ਜੇ_ਅੰਤ
        device->subgroup_shuffle = (vk11_props.subgroupSupportedStages & vk::Shader_Stage_Flag_ਬਿੱਟ::eCompute) &&
                                   (vk11_props.subgroupSupportedOperations & vk::Subgroup_Feature_Flag_ਬਿੱਟ::eShuffle);
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ __APPLE__
        ਜੇ (device->vendor_id == VK_VENDOR_ID_AMD) {
            device->subgroup_shuffle = ਝੂਠ;
        }
#ਜੇ_ਅੰਤ
        device->subgroup_clustered = (vk11_props.subgroupSupportedStages & vk::Shader_Stage_Flag_ਬਿੱਟ::eCompute) &&
                                     (vk11_props.subgroupSupportedOperations & vk::Subgroup_Feature_Flag_ਬਿੱਟ::eClustered);

        device->subgroup_ballot = (vk11_props.subgroupSupportedStages & vk::Shader_Stage_Flag_ਬਿੱਟ::eCompute) &&
                                  (vk11_props.subgroupSupportedOperations & vk::Subgroup_Feature_Flag_ਬਿੱਟ::eBallot);

        device->subgroup_vote = (vk11_props.subgroupSupportedStages & vk::Shader_Stage_Flag_ਬਿੱਟ::eCompute) &&
                                (vk11_props.subgroupSupportedOperations & vk::Subgroup_Feature_Flag_ਬਿੱਟ::eVote);

        // Submit at least every 100 nodes, in case there are workloads without as much matmul.
        device->max_nodes_ਪ੍ਰਤੀ_submit = 100;
        ਅਚਲ ਅੱਖਰ* GGML_VK_MAX_NODES_ਪ੍ਰਤੀ_SUBMIT = getenv("GGML_VK_MAX_NODES_PER_SUBMIT");
        ਜੇ (GGML_VK_MAX_NODES_ਪ੍ਰਤੀ_SUBMIT != ਖਾਲੀ_ਸੂਚਕ) {
            uint32_t max_nodes_ਪ੍ਰਤੀ_submit = std::stoul(GGML_VK_MAX_NODES_ਪ੍ਰਤੀ_SUBMIT);
            device->max_nodes_ਪ੍ਰਤੀ_submit = std::max(max_nodes_ਪ੍ਰਤੀ_submit, 1u);
        }

        ਅਚਲ ਤਾਰਕਿਕ force_disable_f16 = getenv("GGML_VK_DISABLE_F16") != ਖਾਲੀ_ਸੂਚਕ;

        device->fp16 = !force_disable_f16 && fp16_storage && fp16_compute;

        ਜੇ (!ggml_vk_khr_cooperative_ਮੈਟ੍ਰਿਕਸ_support(device->properties, driver_props, device->architecture)) {
            device->coopmat_support = ਝੂਠ;
        }

        device->integer_dot_ਗੁਣਨਫਲ = device->integer_dot_ਗੁਣਨਫਲ && shader_integer_dot_ਗੁਣਨਫਲ_props.integer_Dot_Product4x8_ਦੋਆਧਾਰੀ_ਅੰਕ_Packed_Signed_Accelerated;

        device->min_imported_host_ਸੂਚਕ_alignment = external_ਸਮ੍ਰਿਤੀ_host_props.min_Imported_Host_ਸੂਚਕ_Alignment;

        device->max_workgroup_ਮਾਪ_log2 = uint32_t(log2f(ਦਸ਼ਮਲਵ(device->properties.limits.max_Compute_Work_ਸਮੂਹ_Invocations)));

        std::vector<vk::ਕਤਾਰ_Family_Properties> ਕਤਾਰ_family_props = device->physical_device.get_ਕਤਾਰ_Family_Properties();

        // Try to find a non-graphics compute queue and transfer-focused queues
        // Allow overriding avoiding the graphics queue because it can increase performance on RADV
        ਅਚਲ ਤਾਰਕਿਕ allow_graphics_ਕਤਾਰ = (getenv("GGML_VK_ALLOW_GRAPHICS_QUEUE") != ਖਾਲੀ_ਸੂਚਕ);
        ਅਚਲ vk::ਕਤਾਰ_Flag_ਬਿੱਟ graphics_flag = allow_graphics_ਕਤਾਰ ? (vk::ਕਤਾਰ_Flag_ਬਿੱਟ)0 : vk::ਕਤਾਰ_Flag_ਬਿੱਟ::eGraphics;
        ਅਚਲ uint32_t compute_ਕਤਾਰ_family_ਸੂਚਕਾਂਕ = ggml_vk_ਲੱਭੋ_ਕਤਾਰ_family_ਸੂਚਕਾਂਕ(ਕਤਾਰ_family_props, vk::ਕਤਾਰ_Flag_ਬਿੱਟ::eCompute, graphics_flag, -1, 1);
        ਅਚਲ uint32_t transfer_ਕਤਾਰ_family_ਸੂਚਕਾਂਕ = ggml_vk_ਲੱਭੋ_ਕਤਾਰ_family_ਸੂਚਕਾਂਕ(ਕਤਾਰ_family_props, vk::ਕਤਾਰ_Flag_ਬਿੱਟ::eTransfer, vk::ਕਤਾਰ_Flag_ਬਿੱਟ::eCompute | graphics_flag, compute_ਕਤਾਰ_family_ਸੂਚਕਾਂਕ, 1);

        ਅਚਲ ਦਸ਼ਮਲਵ priorities[] = { 1.0f, 1.0f };
        device->single_ਕਤਾਰ = compute_ਕਤਾਰ_family_ਸੂਚਕਾਂਕ == transfer_ਕਤਾਰ_family_ਸੂਚਕਾਂਕ && ਕਤਾਰ_family_props[compute_ਕਤਾਰ_family_ਸੂਚਕਾਂਕ].ਕਤਾਰ_ਗਿਣਤੀ == 1;

        std::vector<vk::Device_ਕਤਾਰ_Create_Info> device_ਕਤਾਰ_create_infos;
        vk::DeviceCreateInfo device_create_info{};
        std::vector<ਅਚਲ ਅੱਖਰ *> device_extensions;
        vk::PhysicalDeviceFeatures device_features = device->physical_device.getFeatures();

        VkPhysicalDeviceFeatures2 device_features2;
        device_features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        device_features2.p_ਅਗਲਾ = ਖਾਲੀ_ਸੂਚਕ;
        device_features2.features = (VkPhysicalDeviceFeatures)device_features;

        VkPhysicalDeviceVulkan11Features vk11_features;
        vk11_features.p_ਅਗਲਾ = ਖਾਲੀ_ਸੂਚਕ;
        vk11_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
        device_features2.p_ਅਗਲਾ = &vk11_features;

        VkPhysicalDeviceVulkan12Features vk12_features;
        vk12_features.p_ਅਗਲਾ = ਖਾਲੀ_ਸੂਚਕ;
        vk12_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        vk11_features.p_ਅਗਲਾ = &vk12_features;

        ਆਖਰੀ_struct = (Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure *)&vk12_features;

        VkPhysicalDeviceInternallySynchronizedQueuesFeaturesKHR internally_synchronized_queues_features{};
        internally_synchronized_queues_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INTERNALLY_SYNCHRONIZED_QUEUES_FEATURES_KHR;
        internally_synchronized_queues_features.p_ਅਗਲਾ = ਖਾਲੀ_ਸੂਚਕ;
        internally_synchronized_queues_features.internallySynchronizedQueues = VK_FALSE;

        ਜੇ (internally_sync_support) {
            ਆਖਰੀ_struct->p_ਅਗਲਾ = (Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure *)&internally_synchronized_queues_features;
            ਆਖਰੀ_struct = (Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure *)&internally_synchronized_queues_features;
            device_extensions.push_back(VK_KHR_INTERNALLY_SYNCHRONIZED_QUEUES_EXTENSION_NAME);
        }

        VkPhysicalDevicePipelineRobustnessFeaturesEXT pl_robustness_features;
        pl_robustness_features.p_ਅਗਲਾ = ਖਾਲੀ_ਸੂਚਕ;
        pl_robustness_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_ROBUSTNESS_FEATURES_EXT;
        pl_robustness_features.pipelineRobustness = VK_FALSE;

        ਜੇ (pipeline_robustness) {
            ਆਖਰੀ_struct->p_ਅਗਲਾ = (Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure *)&pl_robustness_features;
            ਆਖਰੀ_struct = (Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure *)&pl_robustness_features;
            device_extensions.push_back("VK_EXT_pipeline_robustness");
        }

        Vk_Physical_Device_ਸਮ੍ਰਿਤੀ_ਤਰਜੀਹ_Features_EXT ਸਮ੍ਰਿਤੀ_ਤਰਜੀਹ_features;
        ਸਮ੍ਰਿਤੀ_ਤਰਜੀਹ_features.p_ਅਗਲਾ = ਖਾਲੀ_ਸੂਚਕ;
        ਸਮ੍ਰਿਤੀ_ਤਰਜੀਹ_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ਸਮ੍ਰਿਤੀ_ਤਰਜੀਹ_FEATURES_EXT;
        ਸਮ੍ਰਿਤੀ_ਤਰਜੀਹ_features.ਸਮ੍ਰਿਤੀ_ਤਰਜੀਹ_2 = VK_FALSE;
        ਜੇ (device->ਸਮ੍ਰਿਤੀ_ਤਰਜੀਹ) {
            ਆਖਰੀ_struct->p_ਅਗਲਾ = (Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure *)&ਸਮ੍ਰਿਤੀ_ਤਰਜੀਹ_features;
            ਆਖਰੀ_struct = (Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure *)&ਸਮ੍ਰਿਤੀ_ਤਰਜੀਹ_features;
            device_extensions.push_back("VK_EXT_memory_priority");
        }

        Vk_Physical_Device_Subgroup_ਮਾਪ_Control_Features_EXT subgroup_ਮਾਪ_control_features;
        subgroup_ਮਾਪ_control_features.p_ਅਗਲਾ = ਖਾਲੀ_ਸੂਚਕ;
        subgroup_ਮਾਪ_control_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_ਮਾਪ_CONTROL_FEATURES_EXT;
        subgroup_ਮਾਪ_control_features.compute_ਭਰਿਆ_Subgroups = ਝੂਠ;
        subgroup_ਮਾਪ_control_features.subgroup_ਮਾਪ_Control = ਝੂਠ;

        ਜੇ (device->subgroup_ਮਾਪ_control) {
            ਆਖਰੀ_struct->p_ਅਗਲਾ = (Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure *)&subgroup_ਮਾਪ_control_features;
            ਆਖਰੀ_struct = (Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure *)&subgroup_ਮਾਪ_control_features;
        }

#ਜੇ defined(VK_KHR_cooperative_ਮੈਟ੍ਰਿਕਸ)
        Vk_Physical_Device_Cooperative_ਮੈਟ੍ਰਿਕਸ_Features_KHR coopmat_features;
        coopmat_features.p_ਅਗਲਾ = ਖਾਲੀ_ਸੂਚਕ;
        coopmat_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_ਮੈਟ੍ਰਿਕਸ_FEATURES_KHR;
        coopmat_features.cooperative_ਮੈਟ੍ਰਿਕਸ_2 = VK_FALSE;

        ਜੇ (device->coopmat_support) {
            ਆਖਰੀ_struct->p_ਅਗਲਾ = (Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure *)&coopmat_features;
            ਆਖਰੀ_struct = (Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure *)&coopmat_features;
        }
#ਜੇ_ਅੰਤ

#ਜੇ defined(VK_NV_cooperative_matrix2)
        VkPhysicalDeviceCooperativeMatrix2FeaturesNV coopmat2_features {};
        coopmat2_features.p_ਅਗਲਾ = ਖਾਲੀ_ਸੂਚਕ;
        coopmat2_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_ਮੈਟ੍ਰਿਕਸ_2_FEATURES_NV;
        ਜੇ (coopmat2_support) {
            ਆਖਰੀ_struct->p_ਅਗਲਾ = (Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure *)&coopmat2_features;
            ਆਖਰੀ_struct = (Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure *)&coopmat2_features;
            device_extensions.push_back("VK_NV_cooperative_matrix2");
        }
#ਜੇ_ਅੰਤ

        Vk_Physical_Device_Cooperative_ਮੈਟ੍ਰਿਕਸ_Decode_ਸਦਿਸ਼_Features_NV coopmat2_decode_ਸਦਿਸ਼_features {};
        coopmat2_decode_ਸਦਿਸ਼_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_ਮੈਟ੍ਰਿਕਸ_DECODE_ਸਦਿਸ਼_FEATURES_NV;
        ਜੇ (coopmat2_decode_ਸਦਿਸ਼_support) {
            ਆਖਰੀ_struct->p_ਅਗਲਾ = (Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure *)&coopmat2_decode_ਸਦਿਸ਼_features;
            ਆਖਰੀ_struct = (Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure *)&coopmat2_decode_ਸਦਿਸ਼_features;
            device_extensions.push_back(VK_NV_COOPERATIVE_ਮੈਟ੍ਰਿਕਸ_DECODE_ਸਦਿਸ਼_EXTENSION_NAME);
        }

#ਜੇ defined(VK_KHR_shader_bfloat16)
        VkPhysicalDeviceShaderBfloat16FeaturesKHR bfloat16_features {};
        bfloat16_features.p_ਅਗਲਾ = ਖਾਲੀ_ਸੂਚਕ;
        bfloat16_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR;
        ਜੇ (bfloat16_support) {
            ਆਖਰੀ_struct->p_ਅਗਲਾ = (Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure *)&bfloat16_features;
            ਆਖਰੀ_struct = (Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure *)&bfloat16_features;
            device_extensions.push_back("VK_KHR_shader_bfloat16");
        }
#ਜੇ_ਅੰਤ

        VkPhysicalDeviceShaderOCPMicroscalingTypesFeaturesEXT ocp_microscaling_features {};
        ocp_microscaling_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OCP_MICROSCALING_TYPES_FEATURES_EXT;
        ਜੇ (ocp_microscaling_extension) {
            ਆਖਰੀ_struct->p_ਅਗਲਾ = (Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure *)&ocp_microscaling_features;
            ਆਖਰੀ_struct = (Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure *)&ocp_microscaling_features;
            device_extensions.push_back(VK_EXT_SHADER_OCP_MICROSCALING_TYPES_EXTENSION_NAME);
        }

        VkPhysicalDeviceShaderFloat8FeaturesEXT shader_float8_features {};
        shader_float8_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT;
        ਜੇ (shader_float8_extension) {
            ਆਖਰੀ_struct->p_ਅਗਲਾ = (Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure *)&shader_float8_features;
            ਆਖਰੀ_struct = (Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure *)&shader_float8_features;
            device_extensions.push_back(VK_EXT_SHADER_FLOAT8_EXTENSION_NAME);
        }

        VkPhysicalDeviceMaintenance4Features maint4_features {};
        maint4_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES;
        ਜੇ (maintenance4_support) {
            ਆਖਰੀ_struct->p_ਅਗਲਾ = (Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure *)&maint4_features;
            ਆਖਰੀ_struct = (Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure *)&maint4_features;
            device_extensions.push_back("VK_KHR_maintenance4");
        }

        Vk_Physical_Device_Shader_Integer_Dot_ਗੁਣਨਫਲ_Features_KHR shader_integer_dot_ਗੁਣਨਫਲ_features {};
        shader_integer_dot_ਗੁਣਨਫਲ_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_ਗੁਣਨਫਲ_FEATURES_KHR;
        ਜੇ (device->integer_dot_ਗੁਣਨਫਲ) {
            ਆਖਰੀ_struct->p_ਅਗਲਾ = (Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure *)&shader_integer_dot_ਗੁਣਨਫਲ_features;
            ਆਖਰੀ_struct = (Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure *)&shader_integer_dot_ਗੁਣਨਫਲ_features;
            device_extensions.push_back("VK_KHR_shader_integer_dot_product");
        }

        Vk_Physical_Device_Shader_Mixed_Float_Dot_ਗੁਣਨਫਲ_Features_VALVE dot2_features {};
        dot2_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_MIXED_FLOAT_DOT_ਗੁਣਨਫਲ_FEATURES_VALVE;
        ਜੇ (dot2_f16_support) {
            ਆਖਰੀ_struct->p_ਅਗਲਾ = (Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure *)&dot2_features;
            ਆਖਰੀ_struct = (Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure *)&dot2_features;
            device_extensions.push_back("VK_VALVE_shader_mixed_float_dot_product");
        }

        VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR pep_features {};
        pep_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR;
        ਜੇ (pipeline_executable_properties_support) {
            ਆਖਰੀ_struct->p_ਅਗਲਾ = (Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure *)&pep_features;
            ਆਖਰੀ_struct = (Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure *)&pep_features;
            device_extensions.push_back("VK_KHR_pipeline_executable_properties");
        }

        ਜੇ (device->external_ਸਮ੍ਰਿਤੀ_host) {
            device_extensions.push_back("VK_EXT_external_memory_host");
        }

#ਜੇ defined(VK_EXT_shader_64bit_indexing)
        Vk_Physical_Device_Shader64_ਦੋਆਧਾਰੀ_ਅੰਕ_Indexing_Features_EXT shader_64bit_indexing_features {};
        shader_64bit_indexing_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_64_ਦੋਆਧਾਰੀ_ਅੰਕ_INDEXING_FEATURES_EXT;
        ਜੇ (device->shader_64b_indexing) {
            ਆਖਰੀ_struct->p_ਅਗਲਾ = (Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure *)&shader_64bit_indexing_features;
            ਆਖਰੀ_struct = (Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure *)&shader_64bit_indexing_features;
            device_extensions.push_back("VK_EXT_shader_64bit_indexing");
        }
#ਜੇ_ਅੰਤ

        VkPhysicalDeviceFaultFeaturesEXT fault_features {};
        fault_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT;
        ਜੇ (device->device_fault) {
            ਆਖਰੀ_struct->p_ਅਗਲਾ = (Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure *)&fault_features;
            ਆਖਰੀ_struct = (Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure *)&fault_features;
            device_extensions.push_back("VK_EXT_device_fault");
        }

        vkGetPhysicalDeviceFeatures2(device->physical_device, &device_features2);

        device->device_fault = device->device_fault && fault_features.deviceFault;

        device->has_internally_synchronized_queues = internally_synchronized_queues_features.internallySynchronizedQueues;

        // Build queue create infos only after querying whether internally synchronized queues are enabled.
        // getQueue2() later uses the same flag, so creation/retrieval must stay consistent.
        vk::Device_ਕਤਾਰ_Create_Flags ਕਤਾਰ_flags = device->has_internally_synchronized_queues ?
                                                eInternallySynchronizedKHR :
                                                vk::Device_ਕਤਾਰ_Create_Flags();

        ਜੇ (compute_ਕਤਾਰ_family_ਸੂਚਕਾਂਕ != transfer_ਕਤਾਰ_family_ਸੂਚਕਾਂਕ) {
            device_ਕਤਾਰ_create_infos.push_back({ਕਤਾਰ_flags, compute_ਕਤਾਰ_family_ਸੂਚਕਾਂਕ, 1, priorities});
            device_ਕਤਾਰ_create_infos.push_back({ਕਤਾਰ_flags, transfer_ਕਤਾਰ_family_ਸੂਚਕਾਂਕ, 1, priorities + 1});
        } ਨਹੀਂਤਾਂ ਜੇ(!device->single_ਕਤਾਰ) {
            device_ਕਤਾਰ_create_infos.push_back({ਕਤਾਰ_flags, compute_ਕਤਾਰ_family_ਸੂਚਕਾਂਕ, 2, priorities});
        } ਨਹੀਂਤਾਂ {
            device_ਕਤਾਰ_create_infos.push_back({ਕਤਾਰ_flags, compute_ਕਤਾਰ_family_ਸੂਚਕਾਂਕ, 1, priorities});
        }

        device->pipeline_executable_properties_support = pipeline_executable_properties_support;

        device->fp16 = device->fp16 && vk12_features.shaderFloat16;

#ਜੇ defined(VK_KHR_shader_bfloat16)
        device->bf16 = bfloat16_support && bfloat16_features.shaderBFloat16Type;
#ਨਹੀਂਤਾਂ
        device->bf16 = ਝੂਠ;
#ਜੇ_ਅੰਤ

        device->dot2_f16 = dot2_f16_support && dot2_features.shader_Mixed_Float_Dot_ਗੁਣਨਫਲ_Float16_Acc_Float32;
        device->ocp_fp4 = ocp_microscaling_extension && ocp_microscaling_features.shaderFloat4 &&
                          shader_float8_extension && shader_float8_features.shaderFloat8 &&
                          !getenv("GGML_VK_DISABLE_OCP_FP4");

        device->pipeline_robustness = pl_robustness_features.pipelineRobustness;

        device->multi_ਜੋੜੋ = vk12_props.shader_Rounding_ਬਹੁਲਕ_RTEFloat16 &&
                            device->properties.limits.max_ਧੱਕੋ_Constants_ਮਾਪ >= ਆਕਾਰ(vk_op_multi_ਜੋੜੋ_ਧੱਕੋ_constants) &&
                            getenv("GGML_VK_DISABLE_MULTI_ADD") == ਖਾਲੀ_ਸੂਚਕ;

        device->shader_int64 = device_features2.features.shaderInt64;
        device->ਬਫਰ_device_ਪਤਾ = vk12_features.ਬਫਰ_Device_ਪਤਾ;
        device->vulkan_ਸਮ੍ਰਿਤੀ_model = vk12_features.vulkan_ਸਮ੍ਰਿਤੀ_Model;

        ਜੇ (device->subgroup_ਮਾਪ_control) {
            device->subgroup_min_ਮਾਪ = subgroup_ਮਾਪ_control_props.min_Subgroup_ਮਾਪ;
            device->subgroup_max_ਮਾਪ = subgroup_ਮਾਪ_control_props.max_Subgroup_ਮਾਪ;
            device_extensions.push_back("VK_EXT_subgroup_size_control");
        }

        device->subgroup_ਮਾਪ_control = device->subgroup_ਮਾਪ_control &&
                (subgroup_ਮਾਪ_control_props.required_Subgroup_ਮਾਪ_Stages & vk::Shader_Stage_Flag_ਬਿੱਟ::eCompute) &&
                subgroup_ਮਾਪ_control_features.subgroup_ਮਾਪ_Control;

        device->subgroup_require_ਭਰਿਆ_support = subgroup_ਮਾਪ_control_features.compute_ਭਰਿਆ_Subgroups;

#ਜੇ defined(VK_KHR_cooperative_ਮੈਟ੍ਰਿਕਸ)
        device->coopmat_support = device->coopmat_support && coopmat_features.cooperative_ਮੈਟ੍ਰਿਕਸ_2;
        device->coopmat1_fa_support = device->coopmat_support && device->subgroup_require_ਭਰਿਆ_support;
#ਜੇ_ਅੰਤ

        ਜੇ (coopmat2_support) {
#ਜੇ defined(VK_NV_cooperative_matrix2) && defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
            ਜੇ (coopmat2_features.cooperative_ਮੈਟ੍ਰਿਕਸ_Workgroup_Scope &&
                coopmat2_features.cooperative_ਮੈਟ੍ਰਿਕਸ_Flexible_Dimensions &&
                coopmat2_features.cooperative_ਮੈਟ੍ਰਿਕਸ_Reductions &&
                coopmat2_features.cooperative_ਮੈਟ੍ਰਿਕਸ_Conversions &&
                coopmat2_features.cooperative_ਮੈਟ੍ਰਿਕਸ_ਪ੍ਰਤੀ_Element_Operations &&
                coopmat2_features.cooperative_ਮੈਟ੍ਰਿਕਸ_Tensor_Addressing &&
                coopmat2_features.cooperative_ਮੈਟ੍ਰਿਕਸ_ਖੰਡ_Loads &&
                vk12_features.ਬਫਰ_Device_ਪਤਾ) {

                std::vector<Vk_Cooperative_ਮੈਟ੍ਰਿਕਸ_Flexible_Dimensions_Properties_NV> flexible_dimensions;
                uint32_t count = 0;

                PFN_vk_Get_Physical_Device_Cooperative_ਮੈਟ੍ਰਿਕਸ_Flexible_Dimensions_Properties_NV
                    vk_Get_Physical_Device_Cooperative_ਮੈਟ੍ਰਿਕਸ_Flexible_Dimensions_Properties_NV =
                        (PFN_vk_Get_Physical_Device_Cooperative_ਮੈਟ੍ਰਿਕਸ_Flexible_Dimensions_Properties_NV)
                        vk_instance.instance.getProcAddr("vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV");

                vk_Get_Physical_Device_Cooperative_ਮੈਟ੍ਰਿਕਸ_Flexible_Dimensions_Properties_NV(device->physical_device, &count, ਖਾਲੀ_ਸੂਚਕ);

                Vk_Cooperative_ਮੈਟ੍ਰਿਕਸ_Flexible_Dimensions_Properties_NV ਖਾਲੀ_ਥਾਵਾਂ_prop {};
                ਖਾਲੀ_ਥਾਵਾਂ_prop.sType = VK_STRUCTURE_TYPE_COOPERATIVE_ਮੈਟ੍ਰਿਕਸ_FLEXIBLE_DIMENSIONS_PROPERTIES_NV;
                flexible_dimensions.resize(count, ਖਾਲੀ_ਥਾਵਾਂ_prop);

                vk_Get_Physical_Device_Cooperative_ਮੈਟ੍ਰਿਕਸ_Flexible_Dimensions_Properties_NV(device->physical_device, &count, flexible_dimensions.ਅੰਕੜੇ());

                ਤਾਰਕਿਕ ਮਿਲਿਆ_fp16_128 = ਝੂਠ,
                     ਮਿਲਿਆ_fp16_256 = ਝੂਠ,
                     ਮਿਲਿਆ_fp32_128 = ਝੂਠ,
                     ਮਿਲਿਆ_fp32_256 = ਝੂਠ;
                ਤਾਰਕਿਕ ਮਿਲਿਆ_bf16_128 = ਝੂਠ,
                     ਮਿਲਿਆ_bf16_256 = ਝੂਠ;
                // need to support fp16*fp16 with fp16/fp32 accumulator, for workgroupsize 128
                // with 32x16x16 and 256 with 32x32x16.
                ਲਈ (ਸਵੈਚਾਲਿਤ &prop : flexible_dimensions) {
                    ਜੇ (prop.saturatingAccumulation == VK_FALSE &&
                        prop.scope == VK_SCOPE_WORKGROUP_KHR) {

                        ਜੇ (prop.AType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                            prop.BType == VK_COMPONENT_TYPE_FLOAT16_KHR) {

                            ਜੇ (prop.workgroupInvocations == 128 &&
                                prop.MGranularity <= 32 &&
                                prop.NGranularity <= 16 &&
                                prop.KGranularity <= 16) {
                                ਜੇ (prop.CType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                                    prop.ਨਤੀਜਾ_Type == VK_COMPONENT_TYPE_FLOAT16_KHR) {
                                    ਮਿਲਿਆ_fp16_128 = ਸੱਚ;
                                }
                                ਜੇ (prop.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                                    prop.ਨਤੀਜਾ_Type == VK_COMPONENT_TYPE_FLOAT32_KHR) {
                                    ਮਿਲਿਆ_fp32_128 = ਸੱਚ;
                                }
                            }
                            ਜੇ (prop.workgroupInvocations == 256 &&
                                prop.MGranularity <= 32 &&
                                prop.NGranularity <= 32 &&
                                prop.KGranularity <= 16) {
                                ਜੇ (prop.CType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                                    prop.ਨਤੀਜਾ_Type == VK_COMPONENT_TYPE_FLOAT16_KHR) {
                                    ਮਿਲਿਆ_fp16_256 = ਸੱਚ;
                                }
                                ਜੇ (prop.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                                    prop.ਨਤੀਜਾ_Type == VK_COMPONENT_TYPE_FLOAT32_KHR) {
                                    ਮਿਲਿਆ_fp32_256 = ਸੱਚ;
                                }
                            }
                        }

#ਜੇ defined(VK_KHR_shader_bfloat16) && defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
                        ਜੇ (bfloat16_support &&
                            prop.AType == VK_COMPONENT_TYPE_BFLOAT16_KHR &&
                            prop.BType == VK_COMPONENT_TYPE_BFLOAT16_KHR &&
                            prop.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                            prop.ਨਤੀਜਾ_Type == VK_COMPONENT_TYPE_FLOAT32_KHR) {

                            ਜੇ (prop.workgroupInvocations == 128 &&
                                prop.MGranularity <= 32 &&
                                prop.NGranularity <= 16 &&
                                prop.KGranularity <= 16) {
                                ਮਿਲਿਆ_bf16_128 = ਸੱਚ;
                            }
                            ਜੇ (prop.workgroupInvocations == 256 &&
                                prop.MGranularity <= 32 &&
                                prop.NGranularity <= 32 &&
                                prop.KGranularity <= 16) {
                                ਮਿਲਿਆ_bf16_256 = ਸੱਚ;
                            }
                        }
#ਜੇ_ਅੰਤ
                    }
                }
                ਜੇ (ਮਿਲਿਆ_fp16_128 && ਮਿਲਿਆ_fp16_256 &&
                    ਮਿਲਿਆ_fp32_128 && ਮਿਲਿਆ_fp32_256 &&
                    coopmat2_props.cooperative_ਮੈਟ੍ਰਿਕਸ_Flexible_Dimensions_Max_Dimension >= 512) {
                    device->coopmat2 = ਸੱਚ;
                    device->coopmat2_bf16_support = ਮਿਲਿਆ_bf16_128 && ਮਿਲਿਆ_bf16_256;
                    device->coopmat2_decode_ਸਦਿਸ਼ = coopmat2_decode_ਸਦਿਸ਼_support && coopmat2_decode_ਸਦਿਸ਼_features.cooperative_ਮੈਟ੍ਰਿਕਸ_Decode_ਸਦਿਸ਼;
                }
            }
#ਜੇ_ਅੰਤ
        }

        ਜੇ (!vk11_features.storage_Buffer16_ਦੋਆਧਾਰੀ_ਅੰਕ_Access) {
            std::cerr << "ggml_vulkan: device " << GGML_VK_NAME << idx << " does not support 16-bit storage." << std::ਸਤਰ_ਅੰਤ;
            ਸੁੱਟੋ std::runtime_error("Unsupported device");
        }

        device_extensions.push_back("VK_KHR_16bit_storage");

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_VULKAN_VALIDATE
        device_extensions.push_back("VK_KHR_shader_non_semantic_info");
#ਜੇ_ਅੰਤ

        ਜੇ (device->fp16) {
            device_extensions.push_back("VK_KHR_shader_float16_int8");
        }

#ਜੇ defined(VK_KHR_cooperative_ਮੈਟ੍ਰਿਕਸ)
        ਜੇ (device->coopmat_support) {
            // Query supported shapes
            std::vector<Vk_Cooperative_ਮੈਟ੍ਰਿਕਸ_Properties_KHR> cm_props;

            PFN_vk_Get_Physical_Device_Cooperative_ਮੈਟ੍ਰਿਕਸ_Properties_KHR pfn_vk_Get_Physical_Device_Cooperative_ਮੈਟ੍ਰਿਕਸ_Properties_KHR =
                (PFN_vk_Get_Physical_Device_Cooperative_ਮੈਟ੍ਰਿਕਸ_Properties_KHR)vkGetInstanceProcAddr(vk_instance.instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");

            uint32_t cm_props_ਅੰਕ;

            pfn_vk_Get_Physical_Device_Cooperative_ਮੈਟ੍ਰਿਕਸ_Properties_KHR(device->physical_device, &cm_props_ਅੰਕ, ਖਾਲੀ_ਸੂਚਕ);

            cm_props.resize(cm_props_ਅੰਕ);

            ਲਈ (ਸਵੈਚਾਲਿਤ& prop : cm_props) {
                prop.sType = VK_STRUCTURE_TYPE_COOPERATIVE_ਮੈਟ੍ਰਿਕਸ_PROPERTIES_KHR;
            }

            pfn_vk_Get_Physical_Device_Cooperative_ਮੈਟ੍ਰਿਕਸ_Properties_KHR(device->physical_device, &cm_props_ਅੰਕ, cm_props.ਅੰਕੜੇ());

            VK_LOG_DEBUG("ggml_vulkan: Cooperative Matrix Shapes: " << cm_props.size());

            ਲਈ (ਸਵੈਚਾਲਿਤ& prop : cm_props) {
                VK_LOG_DEBUG("ggml_vulkan: M: " << prop.MSize << " N: " << prop.NSize << " K: " << prop.KSize << " A: " << vk::to_string((vk::ComponentTypeKHR)prop.AType) << " B: " << vk::to_string((vk::ComponentTypeKHR)prop.BType) << " C: " << vk::to_string((vk::ComponentTypeKHR)prop.CType) << " Result: " << vk::to_string((vk::ComponentTypeKHR)prop.ਨਤੀਜਾ_Type) << " saturatingAccumulation: " << prop.saturatingAccumulation << " scope: " << vk::to_string((vk::ScopeKHR)prop.scope));

                ਜੇ ((vk::ComponentTypeKHR)prop.AType == vk::ComponentTypeKHR::eFloat16 &&
                    (vk::ComponentTypeKHR)prop.BType == vk::ComponentTypeKHR::eFloat16 &&
                    (vk::ScopeKHR)prop.scope == vk::ScopeKHR::eSubgroup
                ) {
                    ਜੇ ((vk::ComponentTypeKHR)prop.CType == vk::ComponentTypeKHR::eFloat32 &&
                        (vk::ComponentTypeKHR)prop.ਨਤੀਜਾ_Type == vk::ComponentTypeKHR::eFloat32) {
                        // coopmat sizes not set yet
                        ਜੇ (device->coopmat_m == 0) {
                            device->coopmat_acc_f32_support = ਸੱਚ;
                            device->coopmat_m = prop.MSize;
                            device->coopmat_n = prop.NSize;
                            device->coopmat_k = prop.KSize;
                        } ਨਹੀਂਤਾਂ ਜੇ (device->coopmat_m == prop.MSize && device->coopmat_n == prop.NSize && device->coopmat_k == prop.KSize) {
                            // Only enable if shape is identical
                            device->coopmat_acc_f32_support = ਸੱਚ;
                        }
                        ਜੇ (prop.MSize == 16 && prop.NSize == 16 && prop.KSize == 16) {
                            device->coopmat_support_16x16x16_f32acc = ਸੱਚ;
                        }
                    } ਨਹੀਂਤਾਂ ਜੇ ((vk::ComponentTypeKHR)prop.CType == vk::ComponentTypeKHR::eFloat16 &&
                               (vk::ComponentTypeKHR)prop.ਨਤੀਜਾ_Type == vk::ComponentTypeKHR::eFloat16) {
                        // coopmat sizes not set yet
                        ਜੇ (device->coopmat_m == 0) {
                            device->coopmat_acc_f16_support = ਸੱਚ;
                            device->coopmat_m = prop.MSize;
                            device->coopmat_n = prop.NSize;
                            device->coopmat_k = prop.KSize;
                        } ਨਹੀਂਤਾਂ ਜੇ (device->coopmat_m == prop.MSize && device->coopmat_n == prop.NSize && device->coopmat_k == prop.KSize) {
                            // Only enable if shape is identical
                            device->coopmat_acc_f16_support = ਸੱਚ;
                        }
                        ਜੇ (prop.MSize == 16 && prop.NSize == 16 && prop.KSize == 16) {
                            device->coopmat_support_16x16x16_f16acc = ਸੱਚ;
                        }
                    }
                } ਨਹੀਂਤਾਂ ਜੇ ((vk::ComponentTypeKHR)prop.AType      == vk::ComponentTypeKHR::eSint8 &&
                           (vk::ComponentTypeKHR)prop.BType      == vk::ComponentTypeKHR::eSint8 &&
                           (vk::ComponentTypeKHR)prop.CType      == vk::ComponentTypeKHR::eSint32 &&
                           (vk::ComponentTypeKHR)prop.ਨਤੀਜਾ_Type == vk::ComponentTypeKHR::eSint32 &&
                           (vk::ScopeKHR)prop.scope == vk::ScopeKHR::eSubgroup &&
                           device->coopmat_int_m == 0
                ) {
                    device->coopmat_int_support = ਸੱਚ;
                    device->coopmat_int_m = prop.MSize;
                    device->coopmat_int_n = prop.NSize;
                    device->coopmat_int_k = prop.KSize;
                }
#ਜੇ defined(VK_KHR_shader_bfloat16) && defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
                ਜੇ (bfloat16_support &&
                    prop.AType == VK_COMPONENT_TYPE_BFLOAT16_KHR &&
                    prop.BType == VK_COMPONENT_TYPE_BFLOAT16_KHR &&
                    prop.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                    prop.ਨਤੀਜਾ_Type == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                    (vk::ScopeKHR)prop.scope == vk::ScopeKHR::eSubgroup
                ) {
                    // coopmat sizes not set yet
                    ਜੇ (device->coopmat_m == 0) {
                        device->coopmat_bf16_support = ਸੱਚ;
                        device->coopmat_m = prop.MSize;
                        device->coopmat_n = prop.NSize;
                        device->coopmat_k = prop.KSize;
                    } ਨਹੀਂਤਾਂ ਜੇ (device->coopmat_m == prop.MSize && device->coopmat_n == prop.NSize && device->coopmat_k == prop.KSize) {
                        // Only enable if shape is identical
                        device->coopmat_bf16_support = ਸੱਚ;
                    }
                }
#ਜੇ_ਅੰਤ
            }

            ਜੇ (device->coopmat_m == 0 || !device->coopmat_acc_f32_support) {
                // No suitable matmul mode found
                GGML_LOG_DEBUG("ggml_vulkan: WARNING: No suitable matrix core mode found. Disabling matrix cores.\n");
                device->coopmat_support = ਝੂਠ;
            }
        }

        ਜੇ (device->coopmat_support) {
            device_extensions.push_back("VK_KHR_cooperative_matrix");
        }
#ਜੇ_ਅੰਤ
        device->name = GGML_VK_NAME + std::to_string(idx);

        device_create_info
            .ਸੈੱਟ_Flags(vk::DeviceCreateFlags())
            .ਸੈੱਟ_ਕਤਾਰ_Create_Infos(device_ਕਤਾਰ_create_infos)
            .ਸੈੱਟ_PEnabled_Extension_ਨਾਂ(device_extensions);
        device_create_info.ਸੈੱਟ_PNext(&device_features2);
        device->device = device->physical_device.createDevice(device_create_info);

        ਜੇ (device->device_fault) {
            device->pfn_vkGetDeviceFaultInfoEXT = (PFN_vkGetDeviceFaultInfoEXT)
                vkGetDeviceProcAddr(device->device, "vkGetDeviceFaultInfoEXT");
        }

        // Queues
        device->compute_ਕਤਾਰ = ggml_vk_create_ਕਤਾਰ(device, compute_ਕਤਾਰ_family_ਸੂਚਕਾਂਕ, 0, { vk::Pipeline_Stage_Flag_ਬਿੱਟ::eComputeShader | vk::Pipeline_Stage_Flag_ਬਿੱਟ::eTransfer }, ਝੂਠ);

        // Shaders
        // Disable matmul tile sizes early if performance low or not supported
        ਲਈ (uint32_t i = 0; i < GGML_TYPE_ਗਿਣਤੀ; ++i) {
            ਚੋਣ (device->vendor_id) {
#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ GGML_VULKAN_ਲੜੀਵਾਰ_TESTS
            ਹਾਲਤ VK_VENDOR_ID_AMD:
                device->mul_mat_l[i]    = device->coopmat_support && device->driver_id != vk::DriverId::eAmdProprietary;
                device->mul_mat_m[i]    = ਸੱਚ;
                device->mul_mat_s[i]    = ਸੱਚ;
                device->mul_mat_id_l[i] = ਝੂਠ;
                device->mul_mat_id_m[i] = ਸੱਚ;
                device->mul_mat_id_s[i] = ਸੱਚ;
                ਰੋਕੋ;
            ਹਾਲਤ VK_VENDOR_ID_INTEL: {
                // Current Windows driver does not expose BF16 support.
                // We only want to use l_warptile if coopmat is available
                ਅਚਲ ਤਾਰਕਿਕ use_l_warptile = (i == GGML_TYPE_BF16) ? (device->coopmat_bf16_support && device->coopmat_support) : device->coopmat_support;
                device->mul_mat_l[i] = use_l_warptile;
                device->mul_mat_id_l[i] = use_l_warptile;
                device->mul_mat_m[i] = ਸੱਚ;
                device->mul_mat_s[i] = ਸੱਚ;
                device->mul_mat_id_m[i] = ਸੱਚ;
                device->mul_mat_id_s[i] = ਸੱਚ;
                ਰੋਕੋ;
            }
            ਹਾਲਤ VK_VENDOR_ID_APPLE:
                device->mul_mat_l[i] = ਝੂਠ;
                device->mul_mat_m[i] = ਸੱਚ;
                device->mul_mat_s[i] = ਝੂਠ;
                device->mul_mat_id_l[i] = ਝੂਠ;
                device->mul_mat_id_m[i] = ਸੱਚ;
                device->mul_mat_id_s[i] = ਝੂਠ;
                ਰੋਕੋ;
            ਹਾਲਤ VK_VENDOR_ID_QUALCOMM:
                device->mul_mat_l[i] = ਝੂਠ;
                device->mul_mat_m[i] = ਸੱਚ;
                device->mul_mat_s[i] = !device->coopmat_support;
                device->mul_mat_id_l[i] = ਝੂਠ;
                device->mul_mat_id_m[i] = ਸੱਚ;
                device->mul_mat_id_s[i] = !device->coopmat_support;
                ਰੋਕੋ;
            ਹਾਲਤ VK_VENDOR_ID_SAMSUNG:
                device->mul_mat_l[i] = ਝੂਠ;
                device->mul_mat_m[i] = ਸੱਚ;
                device->mul_mat_s[i] = ਸੱਚ;
                device->mul_mat_id_l[i] = ਝੂਠ;
                device->mul_mat_id_m[i] = ਸੱਚ;
                device->mul_mat_id_s[i] = ਸੱਚ;
                ਰੋਕੋ;
#ਜੇ_ਅੰਤ
            ਮੂਲ:
                device->mul_mat_l[i] = ਸੱਚ;
                device->mul_mat_m[i] = ਸੱਚ;
                device->mul_mat_s[i] = ਸੱਚ;
                device->mul_mat_id_l[i] = ਸੱਚ;
                device->mul_mat_id_m[i] = ਸੱਚ;
                device->mul_mat_id_s[i] = ਸੱਚ;
                ਰੋਕੋ;
            }

#ਜੇ VK_ਸਿਰਲੇਖ_ਸੰਸਕਰਣ >= 287
            // Honeykrisp driver for Asahi Linux doesn't report VK_VENDOR_ID_APPLE.
            // Check for Honeykrisp driver and force same configuration as the VK_VENDOR_ID_APPLE case.
            ਜੇ (device->driver_id == vk::DriverId::eMesaHoneykrisp) {
                device->mul_mat_l[i] = ਝੂਠ;
                device->mul_mat_m[i] = ਸੱਚ;
                device->mul_mat_s[i] = ਝੂਠ;
                device->mul_mat_id_l[i] = ਝੂਠ;
                device->mul_mat_id_m[i] = ਸੱਚ;
                device->mul_mat_id_s[i] = ਝੂਠ;
            }
#ਜੇ_ਅੰਤ

            device->mul_mat_l_int[i]    = device->mul_mat_l[i];
            device->mul_mat_m_int[i]    = device->mul_mat_m[i];
            device->mul_mat_s_int[i]    = device->mul_mat_s[i];
            device->mul_mat_id_l_int[i] = device->mul_mat_id_l[i];
            device->mul_mat_id_m_int[i] = device->mul_mat_id_m[i];
            device->mul_mat_id_s_int[i] = device->mul_mat_id_s[i];
        }


        std::vector<vk::Descriptor_ਸੈੱਟ_Layout_Binding> dsl_binding;
        std::vector<vk::DescriptorBindingFlags> dsl_binding_flags;
        ਲਈ (uint32_t i = 0; i < MAX_PARAMETER_ਗਿਣਤੀ; i++) {
            dsl_binding.push_back({i, vk::DescriptorType::e_Storage_ਬਫਰ, 1, vk::Shader_Stage_Flag_ਬਿੱਟ::eCompute});
            dsl_binding_flags.push_back({});
        }

        vk::Descriptor_ਸੈੱਟ_Layout_Binding_Flags_Create_Info dslbfci = { dsl_binding_flags };

        vk::Descriptor_ਸੈੱਟ_Layout_Create_Info descriptor_ਸੈੱਟ_layout_create_info(
            {},
            dsl_binding);
        descriptor_ਸੈੱਟ_layout_create_info.ਸੈੱਟ_PNext(&dslbfci);
        device->dsl = device->device.create_Descriptor_ਸੈੱਟ_Layout(descriptor_ਸੈੱਟ_layout_create_info);

        ggml_vk_ਲੱਦੋ_shaders(device);

        // Prefer a dedicated transfer queue on AMD dGPUs (non-GCN) when graphics queue use is disabled.
        ਅਚਲ ਤਾਰਕਿਕ prefers_transfer_ਕਤਾਰ =
            device->vendor_id == VK_VENDOR_ID_AMD &&
            device->architecture != AMD_GCN &&
            !device->uma &&
            !allow_graphics_ਕਤਾਰ;

        ਜੇ (!device->single_ਕਤਾਰ) {
            ਅਚਲ uint32_t transfer_ਕਤਾਰ_ਸੂਚਕਾਂਕ = compute_ਕਤਾਰ_family_ਸੂਚਕਾਂਕ == transfer_ਕਤਾਰ_family_ਸੂਚਕਾਂਕ ? 1 : 0;
            device->transfer_ਕਤਾਰ = ggml_vk_create_ਕਤਾਰ(device, transfer_ਕਤਾਰ_family_ਸੂਚਕਾਂਕ, transfer_ਕਤਾਰ_ਸੂਚਕਾਂਕ, { vk::Pipeline_Stage_Flag_ਬਿੱਟ::eTransfer }, ਸੱਚ);

            device->async_use_transfer_ਕਤਾਰ = prefers_transfer_ਕਤਾਰ || (getenv("GGML_VK_ASYNC_USE_TRANSFER_QUEUE") != ਖਾਲੀ_ਸੂਚਕ);
        } ਨਹੀਂਤਾਂ {
            device->transfer_ਕਤਾਰ = ggml_vk_create_aliased_ਕਤਾਰ(device, device->compute_ਕਤਾਰ);

            device->async_use_transfer_ਕਤਾਰ = ਝੂਠ;
        }

        device->fence = device->device.createFence({});

        device->ਬਫਰ_type = {
            /* .iface    = */ ggml_backend_vk_ਬਫਰ_type_interface,
            /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_vk_reg(), idx),
            /* .context  = */ ਨਵਾਂ ggml_backend_vk_ਬਫਰ_type_context{ device->name, device },
        };

        device->idx = idx;

        device->serialize_submissions = getenv("GGML_VK_SERIALIZE_SUBMISSIONS") != ਖਾਲੀ_ਸੂਚਕ;

        device->disable_fusion = getenv("GGML_VK_DISABLE_FUSION") != ਖਾਲੀ_ਸੂਚਕ;

        device->disable_descriptor_reuse = getenv("GGML_VK_DISABLE_DESCRIPTOR_REUSE") != ਖਾਲੀ_ਸੂਚਕ;

        device->ਜੋੜੋ_rms_fusion = !device->disable_fusion &&
                                 device->subgroup_arithmetic &&
                                 device->vendor_id != VK_VENDOR_ID_INTEL;
        device->partials_binding_alignment =
            std::max(4u, (uint32_t)device->properties.limits.min_Storage_ਬਫਰ_ਵਿੱਥ_Alignment);

        device->mmvq_ਬਹੁਲਕ = 0;
        ਜੇ (getenv("GGML_VK_DISABLE_MMVQ")) {
            device->mmvq_ਬਹੁਲਕ = -1;
        } ਨਹੀਂਤਾਂ ਜੇ (getenv("GGML_VK_FORCE_MMVQ")) {
            device->mmvq_ਬਹੁਲਕ = 1;
        }

        // publish the device only when it is fully built
        vk_instance.devices[idx] = device;

        ਵਾਪਸ device;
    }

    ਵਾਪਸ vk_instance.devices[idx];
}

ਸਥਿਰ ਖਾਲੀ ggml_vk_print_gpu_info(size_t idx) {
    GGML_ASSERT(idx < vk_instance.device_indices.size());
    size_t dev_ਅੰਕ = vk_instance.device_indices[idx];
    VK_LOG_DEBUG("ggml_vk_print_gpu_info(" << dev_ਅੰਕ << ")");
    GGML_ASSERT(vk_instance_initialized);

    std::vector<vk::PhysicalDevice> devices = vk_instance.instance.enumeratePhysicalDevices();

    ਜੇ (dev_ਅੰਕ >= devices.size()) {
        std::cerr << "ggml_vulkan: Device with index " << dev_ਅੰਕ << " does not exist." << std::ਸਤਰ_ਅੰਤ;
        ਸੁੱਟੋ std::runtime_error("Device not found");
    }

    vk::PhysicalDevice physical_device = devices[dev_ਅੰਕ];
    std::vector<vk::ExtensionProperties> ext_props = physical_device.enumerateDeviceExtensionProperties();

    ਤਾਰਕਿਕ fp16_storage = ਝੂਠ;
    ਤਾਰਕਿਕ fp16_compute = ਝੂਠ;
    ਤਾਰਕਿਕ coopmat_support = ਝੂਠ;
    ਤਾਰਕਿਕ coopmat2_support = ਝੂਠ;
    ਤਾਰਕਿਕ coopmat2_decode_ਸਦਿਸ਼_support = ਝੂਠ;
    ਤਾਰਕਿਕ integer_dot_ਗੁਣਨਫਲ = ਝੂਠ;
    ਤਾਰਕਿਕ bfloat16_support = ਝੂਠ;
    ਤਾਰਕਿਕ dot2_f16_support = ਝੂਠ;
    ਤਾਰਕਿਕ ocp_microscaling_extension = ਝੂਠ;
    ਤਾਰਕਿਕ shader_float8_extension = ਝੂਠ;

    ਲਈ (ਸਵੈਚਾਲਿਤ properties : ext_props) {
        ਜੇ (strcmp("VK_KHR_16bit_storage", properties.extensionName) == 0) {
            fp16_storage = ਸੱਚ;
        } ਨਹੀਂਤਾਂ ਜੇ (strcmp("VK_KHR_shader_float16_int8", properties.extensionName) == 0) {
            fp16_compute = ਸੱਚ;
#ਜੇ defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
       } ਨਹੀਂਤਾਂ ਜੇ (strcmp("VK_KHR_cooperative_matrix", properties.extensionName) == 0 &&
                   !getenv("GGML_VK_DISABLE_COOPMAT")) {
            coopmat_support = ਸੱਚ;
#ਜੇ_ਅੰਤ
#ਜੇ defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
        } ਨਹੀਂਤਾਂ ਜੇ (strcmp("VK_NV_cooperative_matrix2", properties.extensionName) == 0 &&
                   !getenv("GGML_VK_DISABLE_COOPMAT2")) {
            coopmat2_support = ਸੱਚ;
#ਜੇ_ਅੰਤ
        } ਨਹੀਂਤਾਂ ਜੇ (strcmp(VK_NV_COOPERATIVE_ਮੈਟ੍ਰਿਕਸ_DECODE_ਸਦਿਸ਼_EXTENSION_NAME, properties.extensionName) == 0 &&
                   !getenv("GGML_VK_DISABLE_COOPMAT2_DECODE_VECTOR")) {
            coopmat2_decode_ਸਦਿਸ਼_support = ਸੱਚ;
#ਜੇ defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
        } ਨਹੀਂਤਾਂ ਜੇ (strcmp("VK_KHR_shader_integer_dot_product", properties.extensionName) == 0 &&
                    !getenv("GGML_VK_DISABLE_INTEGER_DOT_PRODUCT")) {
            integer_dot_ਗੁਣਨਫਲ = ਸੱਚ;
#ਜੇ_ਅੰਤ
#ਜੇ defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        } ਨਹੀਂਤਾਂ ਜੇ (strcmp("VK_KHR_shader_bfloat16", properties.extensionName) == 0 &&
                    !getenv("GGML_VK_DISABLE_BFLOAT16")) {
            bfloat16_support = ਸੱਚ;
#ਜੇ_ਅੰਤ
#ਜੇ defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT)
        } ਨਹੀਂਤਾਂ ਜੇ (strcmp(VK_EXT_SHADER_OCP_MICROSCALING_TYPES_EXTENSION_NAME, properties.extensionName) == 0) {
            ocp_microscaling_extension = ਸੱਚ;
#ਜੇ_ਅੰਤ
#ਜੇ defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
        } ਨਹੀਂਤਾਂ ਜੇ (strcmp(VK_EXT_SHADER_FLOAT8_EXTENSION_NAME, properties.extensionName) == 0) {
            shader_float8_extension = ਸੱਚ;
#ਜੇ_ਅੰਤ
        } ਨਹੀਂਤਾਂ ਜੇ (strcmp("VK_VALVE_shader_mixed_float_dot_product", properties.extensionName) == 0 &&
                    !getenv("GGML_VK_DISABLE_DOT2")) {
            dot2_f16_support = ਸੱਚ;
        }
    }

    ਅਚਲ vk_device_architecture device_architecture = get_device_architecture(physical_device);

    ਅਚਲ ਅੱਖਰ* GGML_VK_DISABLE_F16 = getenv("GGML_VK_DISABLE_F16");
    ਤਾਰਕਿਕ force_disable_f16 = GGML_VK_DISABLE_F16 != ਖਾਲੀ_ਸੂਚਕ;

    ਤਾਰਕਿਕ fp16 = !force_disable_f16 && fp16_storage && fp16_compute;

    vk::PhysicalDeviceProperties2 props2;
    vk::PhysicalDeviceMaintenance3Properties props3;
    vk::PhysicalDeviceSubgroupProperties subgroup_props;
    vk::PhysicalDeviceDriverProperties driver_props;
    vk::Physical_Device_Shader_Integer_Dot_ਗੁਣਨਫਲ_Properties_KHR shader_integer_dot_ਗੁਣਨਫਲ_props;
    props2.p_ਅਗਲਾ = &props3;
    props3.p_ਅਗਲਾ = &subgroup_props;
    subgroup_props.p_ਅਗਲਾ = &driver_props;

    // Pointer to the last chain element
    Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure * ਆਖਰੀ_struct = (Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure *)&driver_props;

    ਜੇ (integer_dot_ਗੁਣਨਫਲ) {
        ਆਖਰੀ_struct->p_ਅਗਲਾ = (Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure *)&shader_integer_dot_ਗੁਣਨਫਲ_props;
        ਆਖਰੀ_struct = (Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure *)&shader_integer_dot_ਗੁਣਨਫਲ_props;
    }

    physical_device.getProperties2(&props2);

    VkPhysicalDeviceFeatures2 device_features2;
    device_features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    device_features2.p_ਅਗਲਾ = ਖਾਲੀ_ਸੂਚਕ;

    VkPhysicalDeviceVulkan11Features vk11_features;
    vk11_features.p_ਅਗਲਾ = ਖਾਲੀ_ਸੂਚਕ;
    vk11_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    device_features2.p_ਅਗਲਾ = &vk11_features;

    VkPhysicalDeviceVulkan12Features vk12_features;
    vk12_features.p_ਅਗਲਾ = ਖਾਲੀ_ਸੂਚਕ;
    vk12_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vk11_features.p_ਅਗਲਾ = &vk12_features;

    // Pointer to the last chain element
    ਆਖਰੀ_struct = (Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure *)&vk12_features;

#ਜੇ defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    Vk_Physical_Device_Cooperative_ਮੈਟ੍ਰਿਕਸ_Features_KHR coopmat_features;
    coopmat_features.p_ਅਗਲਾ = ਖਾਲੀ_ਸੂਚਕ;
    coopmat_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_ਮੈਟ੍ਰਿਕਸ_FEATURES_KHR;
    coopmat_features.cooperative_ਮੈਟ੍ਰਿਕਸ_2 = VK_FALSE;

    ਜੇ (coopmat_support) {
        ਆਖਰੀ_struct->p_ਅਗਲਾ = (Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure *)&coopmat_features;
        ਆਖਰੀ_struct = (Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure *)&coopmat_features;
    }
#ਜੇ_ਅੰਤ

    Vk_Physical_Device_Shader_Integer_Dot_ਗੁਣਨਫਲ_Features_KHR shader_integer_dot_ਗੁਣਨਫਲ_features {};
    shader_integer_dot_ਗੁਣਨਫਲ_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_ਗੁਣਨਫਲ_FEATURES_KHR;
    ਜੇ (integer_dot_ਗੁਣਨਫਲ) {
        ਆਖਰੀ_struct->p_ਅਗਲਾ = (Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure *)&shader_integer_dot_ਗੁਣਨਫਲ_features;
        ਆਖਰੀ_struct = (Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure *)&shader_integer_dot_ਗੁਣਨਫਲ_features;
    }

#ਜੇ defined(VK_KHR_shader_bfloat16)
    VkPhysicalDeviceShaderBfloat16FeaturesKHR bfloat16_features {};
    bfloat16_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR;
    ਜੇ (bfloat16_support) {
        ਆਖਰੀ_struct->p_ਅਗਲਾ = (Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure *)&bfloat16_features;
        ਆਖਰੀ_struct = (Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure *)&bfloat16_features;
    }
#ਜੇ_ਅੰਤ

#ਜੇ defined(VK_NV_cooperative_matrix2)
    VkPhysicalDeviceCooperativeMatrix2FeaturesNV coopmat2_features {};
    coopmat2_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_ਮੈਟ੍ਰਿਕਸ_2_FEATURES_NV;
    ਜੇ (coopmat2_support) {
        ਆਖਰੀ_struct->p_ਅਗਲਾ = (Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure *)&coopmat2_features;
        ਆਖਰੀ_struct = (Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure *)&coopmat2_features;
    }
#ਜੇ_ਅੰਤ

    Vk_Physical_Device_Cooperative_ਮੈਟ੍ਰਿਕਸ_Decode_ਸਦਿਸ਼_Features_NV coopmat2_decode_ਸਦਿਸ਼_features {};
    coopmat2_decode_ਸਦਿਸ਼_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_ਮੈਟ੍ਰਿਕਸ_DECODE_ਸਦਿਸ਼_FEATURES_NV;
    ਜੇ (coopmat2_decode_ਸਦਿਸ਼_support) {
        ਆਖਰੀ_struct->p_ਅਗਲਾ = (Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure *)&coopmat2_decode_ਸਦਿਸ਼_features;
        ਆਖਰੀ_struct = (Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure *)&coopmat2_decode_ਸਦਿਸ਼_features;
    }

    Vk_Physical_Device_Shader_Mixed_Float_Dot_ਗੁਣਨਫਲ_Features_VALVE dot2_features {};
    dot2_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_MIXED_FLOAT_DOT_ਗੁਣਨਫਲ_FEATURES_VALVE;
    ਜੇ (dot2_f16_support) {
        ਆਖਰੀ_struct->p_ਅਗਲਾ = (Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure *)&dot2_features;
        ਆਖਰੀ_struct = (Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure *)&dot2_features;
    }

#ਜੇ defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
    VkPhysicalDeviceShaderOCPMicroscalingTypesFeaturesEXT ocp_microscaling_features {};
    ocp_microscaling_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OCP_MICROSCALING_TYPES_FEATURES_EXT;
    VkPhysicalDeviceShaderFloat8FeaturesEXT shader_float8_features {};
    shader_float8_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT;
    ਜੇ (ocp_microscaling_extension) {
        ਆਖਰੀ_struct->p_ਅਗਲਾ = (Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure *)&ocp_microscaling_features;
        ਆਖਰੀ_struct = (Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure *)&ocp_microscaling_features;
    }
    ਜੇ (shader_float8_extension) {
        ਆਖਰੀ_struct->p_ਅਗਲਾ = (Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure *)&shader_float8_features;
        ਆਖਰੀ_struct = (Vk_ਅਧਾਰ_ਬਾਹਰਲਾ_Structure *)&shader_float8_features;
    }
#ਜੇ_ਅੰਤ

    vkGetPhysicalDeviceFeatures2(physical_device, &device_features2);

    fp16 = fp16 && vk12_features.shaderFloat16;

#ਜੇ defined(VK_KHR_shader_bfloat16)
    ਤਾਰਕਿਕ bf16 = bfloat16_support && bfloat16_features.shaderBFloat16Type;
#ਨਹੀਂਤਾਂ
    ਤਾਰਕਿਕ bf16 = ਝੂਠ;
#ਜੇ_ਅੰਤ

    uint32_t default_subgroup_ਮਾਪ = get_subgroup_ਮਾਪ("", device_architecture);
    ਅਚਲ size_t subgroup_ਮਾਪ = (default_subgroup_ਮਾਪ != 0) ? default_subgroup_ਮਾਪ : subgroup_props.subgroup_ਮਾਪ_2;
    ਅਚਲ ਤਾਰਕਿਕ uma = props2.properties.deviceType == vk::PhysicalDeviceType::eIntegratedGpu;

    integer_dot_ਗੁਣਨਫਲ = integer_dot_ਗੁਣਨਫਲ
                       && shader_integer_dot_ਗੁਣਨਫਲ_props.integer_Dot_Product4x8_ਦੋਆਧਾਰੀ_ਅੰਕ_Packed_Signed_Accelerated
                       && shader_integer_dot_ਗੁਣਨਫਲ_features.shader_Integer_Dot_ਗੁਣਨਫਲ;

    coopmat_support = coopmat_support
#ਜੇ defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
                   && coopmat_features.cooperative_ਮੈਟ੍ਰਿਕਸ_2
#ਜੇ_ਅੰਤ
                   && ggml_vk_khr_cooperative_ਮੈਟ੍ਰਿਕਸ_support(props2.properties, driver_props, device_architecture);

#ਜੇ defined(VK_NV_cooperative_matrix2) && defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
    coopmat2_support = coopmat2_support &&
                       coopmat2_features.cooperative_ਮੈਟ੍ਰਿਕਸ_Workgroup_Scope &&
                       coopmat2_features.cooperative_ਮੈਟ੍ਰਿਕਸ_Flexible_Dimensions &&
                       coopmat2_features.cooperative_ਮੈਟ੍ਰਿਕਸ_Reductions &&
                       coopmat2_features.cooperative_ਮੈਟ੍ਰਿਕਸ_Conversions &&
                       coopmat2_features.cooperative_ਮੈਟ੍ਰਿਕਸ_ਪ੍ਰਤੀ_Element_Operations &&
                       coopmat2_features.cooperative_ਮੈਟ੍ਰਿਕਸ_Tensor_Addressing &&
                       coopmat2_features.cooperative_ਮੈਟ੍ਰਿਕਸ_ਖੰਡ_Loads;
#ਨਹੀਂਤਾਂ
    coopmat2_support = ਝੂਠ;
#ਜੇ_ਅੰਤ

    coopmat2_decode_ਸਦਿਸ਼_support = coopmat2_decode_ਸਦਿਸ਼_support && coopmat2_decode_ਸਦਿਸ਼_features.cooperative_ਮੈਟ੍ਰਿਕਸ_Decode_ਸਦਿਸ਼;
#ਜੇ !defined(GGML_VULKAN_COOPMAT2_DECODE_ਸਦਿਸ਼_GLSLC_SUPPORT)
    coopmat2_decode_ਸਦਿਸ਼_support = ਝੂਠ;
#ਜੇ_ਅੰਤ

    std::string ਮੈਟ੍ਰਿਕਸ_cores = coopmat2_support ? (coopmat2_decode_ਸਦਿਸ਼_support ? "NV_coopmat2v" : "NV_coopmat2")
                             : coopmat_support  ? "KHR_coopmat"
                             : "none";

    ਤਾਰਕਿਕ dot2_f16 = dot2_f16_support && dot2_features.shader_Mixed_Float_Dot_ਗੁਣਨਫਲ_Float16_Acc_Float32;
    ਅਚਲ ਅੱਖਰ *fp16_str = fp16 ? (dot2_f16 ? "dot2" : "1") : "0";
#ਜੇ defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
    ਅਚਲ ਤਾਰਕਿਕ fp4 = ocp_microscaling_extension && ocp_microscaling_features.shaderFloat4 &&
                     shader_float8_extension && shader_float8_features.shaderFloat8 &&
                     !getenv("GGML_VK_DISABLE_OCP_FP4");
#ਨਹੀਂਤਾਂ
    GGML_ਅਣਵਰਤਿਆ(ocp_microscaling_extension);
    GGML_ਅਣਵਰਤਿਆ(shader_float8_extension);
    ਅਚਲ ਤਾਰਕਿਕ fp4 = ਝੂਠ;
#ਜੇ_ਅੰਤ

    std::string device_name = props2.properties.deviceName.ਅੰਕੜੇ();
    GGML_LOG_DEBUG("ggml_vulkan: %zu = %s (%s) | uma: %d | fp16: %s | bf16: %d | fp4: %d | warp size: %zu | shared memory: %d | int dot: %d | matrix cores: %s\n",
              idx, device_name.c_str(), driver_props.driverName.ਅੰਕੜੇ(), uma, fp16_str, bf16, fp4, subgroup_ਮਾਪ,
              props2.properties.limits.max_Compute_Shared_ਸਮ੍ਰਿਤੀ_ਮਾਪ, integer_dot_ਗੁਣਨਫਲ, ਮੈਟ੍ਰਿਕਸ_cores.c_str());

    ਜੇ (props2.properties.deviceType == vk::PhysicalDeviceType::eCpu) {
        GGML_LOG_DEBUG("ggml_vulkan: Warning: Device type is CPU. This is probably not the device you want.\n");
    }
}

ਸਥਿਰ DispatchLoaderDynamic ggml_vk_default_dispatcher_instance;

DispatchLoaderDynamic & ggml_vk_default_dispatcher() {
    ਵਾਪਸ ggml_vk_default_dispatcher_instance;
}

ਖਾਲੀ ggml_vk_instance_init() {
    ਜੇ (vk_instance_initialized) {
        ਵਾਪਸ;
    }
    VK_LOG_DEBUG("ggml_vk_instance_init()");

    // See https://github.com/KhronosGroup/Vulkan-Hpp?tab=readme-ov-file#extensions--per-device-function-pointers-
    ggml_vk_default_dispatcher_instance.init(vkGetInstanceProcAddr);

    // vkEnumerateInstanceVersion is Vulkan 1.1. A null value indicated Vulkan 1.0.
    ਜੇ (ggml_vk_default_dispatcher_instance.vk_Enumerate_Instance_ਸੰਸਕਰਣ == ਖਾਲੀ_ਸੂਚਕ) {
        GGML_LOG_ERROR("ggml_vulkan: Error: Vulkan 1.2 required.");
        ਸੁੱਟੋ vk::SystemError(vk::ਨਤੀਜਾ_2::eErrorFeatureNotPresent, "Vulkan 1.2 required");
    }

    uint32_t api_ਸੰਸਕਰਣ = vk::enumerate_Instance_ਸੰਸਕਰਣ();

    ਜੇ (api_ਸੰਸਕਰਣ < VK_API_ਸੰਸਕਰਣ_1_2) {
        GGML_LOG_ERROR("ggml_vulkan: Error: Vulkan 1.2 required.");
        ਸੁੱਟੋ vk::SystemError(vk::ਨਤੀਜਾ_2::eErrorFeatureNotPresent, "Vulkan 1.2 required");
    }

    vk::ApplicationInfo app_info{ "ggml-vulkan", 1, ਖਾਲੀ_ਸੂਚਕ, 0, api_ਸੰਸਕਰਣ };

    ਅਚਲ std::vector<vk::ExtensionProperties> instance_extensions = vk::enumerateInstanceExtensionProperties();
    ਅਚਲ ਤਾਰਕਿਕ layer_settings = ggml_vk_instance_layer_settings_ਉਪਲਬਧ();
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ __APPLE__
    ਅਚਲ ਤਾਰਕਿਕ portability_enumeration_ext = ggml_vk_instance_portability_enumeration_ext_ਉਪਲਬਧ(instance_extensions);
#ਜੇ_ਅੰਤ
    ਅਚਲ ਤਾਰਕਿਕ debug_utils_ext = ggml_vk_instance_debug_utils_ext_ਉਪਲਬਧ(instance_extensions) && getenv("GGML_VK_DEBUG_MARKERS") != ਖਾਲੀ_ਸੂਚਕ;
    std::vector<ਅਚਲ ਅੱਖਰ*> layers;

    ਜੇ (layer_settings) {
        layers.push_back("VK_LAYER_KHRONOS_validation");
    }
    std::vector<ਅਚਲ ਅੱਖਰ*> extensions;
    ਜੇ (layer_settings) {
        extensions.push_back("VK_EXT_layer_settings");
    }
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ __APPLE__
    ਜੇ (portability_enumeration_ext) {
        extensions.push_back("VK_KHR_portability_enumeration");
    }
#ਜੇ_ਅੰਤ
    ਜੇ (debug_utils_ext) {
        extensions.push_back("VK_EXT_debug_utils");
    }
    VkBool32 enable_ਉੱਤਮ_practice = layer_settings;
    std::vector<vk::LayerSettingEXT> settings = {
        {
            "VK_LAYER_KHRONOS_validation",
            "validate_best_practices",
            vk::LayerSettingTypeEXT::eBool32,
            1,
            &enable_ਉੱਤਮ_practice
        },
    };
    vk::LayerSettingsCreateInfoEXT layer_setting_info(settings);
    vk::InstanceCreateInfo instance_create_info(vk::InstanceCreateFlags{}, &app_info, layers, extensions, &layer_setting_info);
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ __APPLE__
    ਜੇ (portability_enumeration_ext) {
        instance_create_info.flags |= vk::Instance_Create_Flag_ਬਿੱਟ::eEnumeratePortabilityKHR;
    }
#ਜੇ_ਅੰਤ

    vk_instance.instance = vk::createInstance(instance_create_info);
    vk_instance_initialized = ਸੱਚ;

    ਜੇ (debug_utils_ext) {
        vk_instance.debug_utils_support              = ਸੱਚ;
        vk_instance.pfn_vk_ਸੈੱਟ_Debug_Utils_Object_Name_EXT = (PFN_vk_ਸੈੱਟ_Debug_Utils_Object_Name_EXT) vkGetInstanceProcAddr(vk_instance.instance, "vkSetDebugUtilsObjectNameEXT");
        vk_instance.pfn_vk_ਕਤਾਰ_Begin_Debug_Utils_Label_EXT = (PFN_vk_ਕਤਾਰ_Begin_Debug_Utils_Label_EXT) vkGetInstanceProcAddr(vk_instance.instance, "vkQueueBeginDebugUtilsLabelEXT");
        vk_instance.pfn_vk_ਕਤਾਰ_ਸਿਰਾ_Debug_Utils_Label_EXT = (PFN_vk_ਕਤਾਰ_ਸਿਰਾ_Debug_Utils_Label_EXT) vkGetInstanceProcAddr(vk_instance.instance, "vkQueueEndDebugUtilsLabelEXT");
        vk_instance.pfn_vkCmdBeginDebugUtilsLabelEXT = (PFN_vkCmdBeginDebugUtilsLabelEXT) vkGetInstanceProcAddr(vk_instance.instance, "vkCmdBeginDebugUtilsLabelEXT");
        vk_instance.pfn_vk_Cmd_ਸਿਰਾ_Debug_Utils_Label_EXT =   (PFN_vk_Cmd_ਸਿਰਾ_Debug_Utils_Label_EXT) vkGetInstanceProcAddr(vk_instance.instance, "vkCmdEndDebugUtilsLabelEXT");
        vk_instance.pfn_vk_Cmd_ਪਾਓ_Debug_Utils_Label_EXT = (PFN_vk_Cmd_ਪਾਓ_Debug_Utils_Label_EXT) vkGetInstanceProcAddr(vk_instance.instance, "vkCmdInsertDebugUtilsLabelEXT");
    }

    vk_perf_logger_enabled = getenv("GGML_VK_PERF_LOGGER") != ਖਾਲੀ_ਸੂਚਕ;
    vk_perf_logger_concurrent = getenv("GGML_VK_PERF_LOGGER_CONCURRENT") != ਖਾਲੀ_ਸੂਚਕ;
    vk_enable_sync_logger = getenv("GGML_VK_SYNC_LOGGER") != ਖਾਲੀ_ਸੂਚਕ;
    vk_ਸਮ੍ਰਿਤੀ_logger_enabled = getenv("GGML_VK_MEMORY_LOGGER") != ਖਾਲੀ_ਸੂਚਕ;
    ਅਚਲ ਅੱਖਰ* GGML_VK_PIPELINE_STATS = getenv("GGML_VK_PIPELINE_STATS");
    ਜੇ (GGML_VK_PIPELINE_STATS != ਖਾਲੀ_ਸੂਚਕ) {
        vk_pipeline_stats_filter = GGML_VK_PIPELINE_STATS;
    }
    ਅਚਲ ਅੱਖਰ* GGML_VK_PERF_LOGGER_FREQUENCY = getenv("GGML_VK_PERF_LOGGER_FREQUENCY");

    ਜੇ (GGML_VK_PERF_LOGGER_FREQUENCY != ਖਾਲੀ_ਸੂਚਕ) {
        vk_perf_logger_frequency = std::stoul(GGML_VK_PERF_LOGGER_FREQUENCY);
    }

    // See https://github.com/KhronosGroup/Vulkan-Hpp?tab=readme-ov-file#extensions--per-device-function-pointers-
    VULKAN_HPP_DEFAULT_DISPATCHER.init(vk_instance.instance);

    std::vector<vk::PhysicalDevice> devices = vk_instance.instance.enumeratePhysicalDevices();

    // Emulate behavior of CUDA_VISIBLE_DEVICES for Vulkan
    ਅੱਖਰ * devices_env = getenv("GGML_VK_VISIBLE_DEVICES");
    ਜੇ (devices_env != ਖਾਲੀ_ਸੂਚਕ) {
        size_t ਅੰਕ_ਉਪਲਬਧ_devices = devices.size();

        std::string devices(devices_env);
        std::replace(devices.begin(), devices.end(), ',', ' ');

        std::stringstream ss(devices);
        size_t tmp;
        ਜਦੋਂ (ss >> tmp) {
            ਜੇ(tmp >= ਅੰਕ_ਉਪਲਬਧ_devices) {
                std::cerr << "ggml_vulkan: Invalid device index " << tmp << " in GGML_VK_VISIBLE_DEVICES." << std::ਸਤਰ_ਅੰਤ;
                ਸੁੱਟੋ std::runtime_error("Invalid Vulkan device index");
            }
            vk_instance.device_indices.push_back(tmp);
        }
    } ਨਹੀਂਤਾਂ {
        // If no vulkan devices are found, return early
        ਜੇ (devices.ਖਾਲੀ_ਥਾਵਾਂ()) {
            GGML_LOG_INFO("ggml_vulkan: No devices found.\n");
            ਵਾਪਸ;
        }

        // Default to using all dedicated GPUs
        ਲਈ (size_t i = 0; i < devices.size(); i++) {
            vk::PhysicalDeviceProperties2 new_props;
            vk::PhysicalDeviceDriverProperties new_driver;
            vk::PhysicalDeviceIDProperties new_id;
            new_props.p_ਅਗਲਾ = &new_driver;
            new_driver.p_ਅਗਲਾ = &new_id;
            devices[i].getProperties2(&new_props);

            ਜੇ ((new_props.properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu || new_props.properties.deviceType == vk::PhysicalDeviceType::eIntegratedGpu) && ggml_vk_device_is_supported(devices[i])) {
                // Check if there are two physical devices corresponding to the same GPU
                // This handles the case where the same GPU appears with different drivers (e.g., RADV + AMDVLK on Linux),
                // see https://github.com/ggml-org/llama.cpp/pull/7582 for original deduplication.
                // MoltenVK on macOS may report the same UUID for distinct GPUs on multi-GPU cards,
                // see https://github.com/KhronosGroup/MoltenVK/issues/2683. Skip when both old/new
                // driver is MoltenVK
                ਸਵੈਚਾਲਿਤ old_device = std::find_if(
                    vk_instance.device_indices.begin(),
                    vk_instance.device_indices.end(),
                    [&devices, &new_id, &new_driver](ਅਚਲ size_t k){
                        vk::PhysicalDeviceProperties2 old_props;
                        vk::PhysicalDeviceDriverProperties old_driver;
                        vk::PhysicalDeviceIDProperties old_id;
                        old_props.p_ਅਗਲਾ = &old_driver;
                        old_driver.p_ਅਗਲਾ = &old_id;
                        devices[k].getProperties2(&old_props);

                        ਤਾਰਕਿਕ same_uuid = std::equal(std::begin(old_id.deviceUUID), std::end(old_id.deviceUUID), std::begin(new_id.deviceUUID));
                        same_uuid = same_uuid || (
                            old_id.deviceLUIDValid && new_id.deviceLUIDValid &&
                            std::equal(std::begin(old_id.deviceLUID), std::end(old_id.deviceLUID), std::begin(new_id.deviceLUID))
                        );
                        ਤਾਰਕਿਕ both_molten_vk = (new_driver.driverID == vk::DriverId::eMoltenvk && old_driver.driverID == vk::DriverId::eMoltenvk);

                        ਵਾਪਸ same_uuid && !both_molten_vk;
                    }
                );
                ਜੇ (old_device == vk_instance.device_indices.end()) {
                    vk_instance.device_indices.push_back(i);
                } ਨਹੀਂਤਾਂ {
                    // There can be two physical devices corresponding to the same GPU if there are 2 different drivers
                    // This can cause error when splitting layers aross the devices, need to keep only 1
                    VK_LOG_DEBUG("Device " << i << " and device " << *old_device << " have the same deviceUUID");

                    vk::PhysicalDeviceProperties2 old_props;
                    vk::PhysicalDeviceDriverProperties old_driver;
                    old_props.p_ਅਗਲਾ = &old_driver;
                    devices[*old_device].getProperties2(&old_props);

                    std::map<vk::DriverId, ਪੂਰਨ> driver_priorities {};
                    ਪੂਰਨ old_ਤਰਜੀਹ = std::numeric_limits<ਪੂਰਨ>::max();
                    ਪੂਰਨ new_ਤਰਜੀਹ = std::numeric_limits<ਪੂਰਨ>::max();

                    // Check https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkDriverId.html for the list of driver id
                    // Smaller number -> higher priority
                    ਚੋਣ (old_props.properties.vendorID) {
                        ਹਾਲਤ VK_VENDOR_ID_AMD:
                            driver_priorities[vk::DriverId::eMesaRadv] = 1;
                            driver_priorities[vk::DriverId::e_Amd_Open_ਸਰੋਤ] = 2;
                            driver_priorities[vk::DriverId::eAmdProprietary] = 3;
                            ਰੋਕੋ;
                        ਹਾਲਤ VK_VENDOR_ID_INTEL:
                            driver_priorities[vk::DriverId::e_Intel_Open_ਸਰੋਤ_MESA] = 1;
                            driver_priorities[vk::DriverId::eIntelProprietaryWindows] = 2;
                            ਰੋਕੋ;
                        ਹਾਲਤ VK_VENDOR_ID_NVIDIA:
                            driver_priorities[vk::DriverId::eNvidiaProprietary] = 1;
#ਜੇ defined(VK_API_ਸੰਸਕਰਣ_1_3) && VK_ਸਿਰਲੇਖ_ਸੰਸਕਰਣ >= 235
                            driver_priorities[vk::DriverId::eMesaNvk] = 2;
#ਜੇ_ਅੰਤ
                            ਰੋਕੋ;
                        ਹਾਲਤ VK_VENDOR_ID_QUALCOMM:
                            driver_priorities[vk::DriverId::eQualcommProprietary] = 1;
                            driver_priorities[vk::DriverId::eMesaTurnip] = 2;
                            ਰੋਕੋ;
                    }
                    driver_priorities[vk::DriverId::eMesaDozen] = 100;

                    ਜੇ (driver_priorities.count(old_driver.driverID)) {
                        old_ਤਰਜੀਹ = driver_priorities[old_driver.driverID];
                    }
                    ਜੇ (driver_priorities.count(new_driver.driverID)) {
                        new_ਤਰਜੀਹ = driver_priorities[new_driver.driverID];
                    }

                    ਜੇ (new_ਤਰਜੀਹ < old_ਤਰਜੀਹ) {
                        ਸਵੈਚਾਲਿਤ r = std::remove(vk_instance.device_indices.begin(), vk_instance.device_indices.end(), *old_device);
                        vk_instance.device_indices.erase(r, vk_instance.device_indices.end());
                        vk_instance.device_indices.push_back(i);

                        VK_LOG_DEBUG("Prioritize device " << i << " driver " << new_driver.driverName << " over device " << *old_device << " driver " << old_driver.driverName);
                    }
                    ਨਹੀਂਤਾਂ {
                        VK_LOG_DEBUG("Prioritize device " << *old_device << " driver " << old_driver.driverName << " over device " << i << " driver " << new_driver.driverName << std::ਸਤਰ_ਅੰਤ);
                    }
                }
            }
        }

        // If no GPUs found, fall back to the first non-CPU device.
        // If only CPU devices are available, return without devices.
        ਜੇ (vk_instance.device_indices.ਖਾਲੀ_ਥਾਵਾਂ()) {
            ਲਈ (size_t i = 0; i < devices.size(); i++) {
                ਜੇ (devices[i].getProperties().deviceType != vk::PhysicalDeviceType::eCpu) {
                    vk_instance.device_indices.push_back(i);
                    ਰੋਕੋ;
                }
            }
        }

        ਜੇ (vk_instance.device_indices.ਖਾਲੀ_ਥਾਵਾਂ()) {
            GGML_LOG_INFO("ggml_vulkan: No devices found.\n");
            ਵਾਪਸ;
        }
    }
    GGML_LOG_DEBUG("ggml_vulkan: Found %zu Vulkan devices:\n", vk_instance.device_indices.size());

    size_t host_ਬਫਰ_import_ਗਿਣਤੀ = 0;
    ਲਈ (size_t i = 0; i < vk_instance.device_indices.size(); i++) {
        vk::PhysicalDevice vkdev = devices[vk_instance.device_indices[i]];
        std::vector<vk::ExtensionProperties> extensionprops = vkdev.enumerateDeviceExtensionProperties();

        ਤਾਰਕਿਕ membudget_supported = ਝੂਠ;
        ਤਾਰਕਿਕ external_ਸਮ੍ਰਿਤੀ_host_supported = ਝੂਠ;
        ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & ext : extensionprops) {
            ਜੇ (strcmp(VK_EXT_ਸਮ੍ਰਿਤੀ_BUDGET_EXTENSION_NAME, ext.extensionName) == 0) {
                membudget_supported = ਸੱਚ;
            } ਨਹੀਂਤਾਂ ਜੇ (strcmp("VK_EXT_external_memory_host", ext.extensionName) == 0) {
                external_ਸਮ੍ਰਿਤੀ_host_supported = ਸੱਚ;
            }
        }

        vk_instance.device_supports_membudget.push_back(membudget_supported);

        ਤਾਰਕਿਕ host_ਬਫਰ_import = ਝੂਠ;
        ਜੇ (external_ਸਮ੍ਰਿਤੀ_host_supported) {
            vk::PhysicalDeviceProperties2 props2;
            vk::PhysicalDeviceDriverProperties driver_props;
            vk::Physical_Device_External_ਸਮ੍ਰਿਤੀ_Host_Properties_EXT ext_props;
            props2.p_ਅਗਲਾ = &driver_props;
            driver_props.p_ਅਗਲਾ = &ext_props;
            vkdev.getProperties2(&props2);
            host_ਬਫਰ_import = ggml_vk_external_ਸਮ੍ਰਿਤੀ_host_usable(driver_props.driverID) &&
                                 ggml_vk_host_ਬਫਰ_import_usable(props2.properties.vendorID);
            ਜੇ (host_ਬਫਰ_import) {
                vk_instance.host_import_alignment = std::max<size_t>(vk_instance.host_import_alignment, ext_props.min_Imported_Host_ਸੂਚਕ_Alignment);
                host_ਬਫਰ_import_ਗਿਣਤੀ++;
            }
        }
        vk_instance.device_host_ਬਫਰ_import.push_back(host_ਬਫਰ_import);

        ggml_vk_print_gpu_info(i);
    }

    vk_instance.multi_device = host_ਬਫਰ_import_ਗਿਣਤੀ > 1 && vk_instance.device_host_ਬਫਰ_import[0];
}

ਖਾਲੀ ggml_vk_init(ggml_backend_vk_context * ctx, size_t idx) {
    VK_LOG_DEBUG("ggml_vk_init(" << ctx->name << ", " << idx << ")");
    ggml_vk_instance_init();
    GGML_ASSERT(idx < vk_instance.device_indices.size());

    ctx->name = GGML_VK_NAME + std::to_string(idx);

    ctx->device = ggml_vk_get_device(idx);

    ctx->semaphore_idx = 0;
    ctx->event_idx = 0;

    ctx->prealloc_ਮਾਪ_x = 0;
    ctx->prealloc_ਮਾਪ_y = 0;
    ctx->prealloc_ਮਾਪ_ਵੰਡੋ_k = 0;
    // Fixed size of 1KB, for deterministic behavior
    ctx->prealloc_ਮਾਪ_ਜੋੜੋ_rms_partials = 1024;

    ctx->fence = ctx->device->device.createFence({});
    ctx->almost_ready_fence = ctx->device->device.createFence({});

    ctx->compute_cmd_pool.init(ctx->device, ctx->device->compute_ਕਤਾਰ.get());
    ਜੇ (ctx->device->async_use_transfer_ਕਤਾਰ) {
        vk::SemaphoreTypeCreateInfo tci{ vk::SemaphoreType::eTimeline, 0 };
        vk::SemaphoreCreateInfo ci{};
        ci.ਸੈੱਟ_PNext(&tci);
        ctx->transfer_semaphore.s = ctx->device->device.createSemaphore(ci);
        ctx->transfer_semaphore.value = 0;

        ctx->transfer_cmd_pool.init(ctx->device, ctx->device->transfer_ਕਤਾਰ.get());
    }

    ਜੇ (vk_perf_logger_enabled) {
        ctx->perf_logger = std::unique_ptr<vk_perf_logger>(ਨਵਾਂ vk_perf_logger());
    }

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_VULKAN_CHECK_RESULTS
    ਅਚਲ ਅੱਖਰ* skip_checks = getenv("GGML_VULKAN_SKIP_CHECKS");
    vk_skip_checks = (skip_checks == ਨਿਰਾ ? 0 : atoi(skip_checks));
    ਅਚਲ ਅੱਖਰ* output_tensor = getenv("GGML_VULKAN_OUTPUT_TENSOR");
    vk_output_tensor = (output_tensor == ਨਿਰਾ ? 0 : atoi(output_tensor));
#ਜੇ_ਅੰਤ
}

vk_pipeline ggml_vk_get_ਮੰਜ਼ਿਲ_fp16(ggml_backend_vk_context * ctx, ggml_type type) {
    VK_LOG_DEBUG("ggml_vk_get_to_fp16()");
    ਚੋਣ (type) {
        ਹਾਲਤ GGML_TYPE_F32:
        ਹਾਲਤ GGML_TYPE_Q1_0:
        ਹਾਲਤ GGML_TYPE_Q2_0:
        ਹਾਲਤ GGML_TYPE_Q4_0:
        ਹਾਲਤ GGML_TYPE_Q4_1:
        ਹਾਲਤ GGML_TYPE_Q5_0:
        ਹਾਲਤ GGML_TYPE_Q5_1:
        ਹਾਲਤ GGML_TYPE_Q8_0:
        ਹਾਲਤ GGML_TYPE_Q2_K:
        ਹਾਲਤ GGML_TYPE_Q3_K:
        ਹਾਲਤ GGML_TYPE_Q4_K:
        ਹਾਲਤ GGML_TYPE_Q5_K:
        ਹਾਲਤ GGML_TYPE_Q6_K:
        ਹਾਲਤ GGML_TYPE_IQ1_S:
        ਹਾਲਤ GGML_TYPE_IQ1_M:
        ਹਾਲਤ GGML_TYPE_IQ2_XXS:
        ਹਾਲਤ GGML_TYPE_IQ2_XS:
        ਹਾਲਤ GGML_TYPE_IQ2_S:
        ਹਾਲਤ GGML_TYPE_IQ3_XXS:
        ਹਾਲਤ GGML_TYPE_IQ3_S:
        ਹਾਲਤ GGML_TYPE_IQ4_XS:
        ਹਾਲਤ GGML_TYPE_IQ4_NL:
        ਹਾਲਤ GGML_TYPE_MXFP4:
        ਹਾਲਤ GGML_TYPE_NVFP4:
        ਹਾਲਤ GGML_TYPE_TQ1_0:
        ਹਾਲਤ GGML_TYPE_TQ2_0:
            ਰੋਕੋ;
        ਮੂਲ:
            ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }

    ਵਾਪਸ ctx->device->pipeline_dequant[type];
}

ਸਥਿਰ vk_pipeline ggml_vk_get_dequantize_mul_mat_vec(ggml_backend_vk_context * ctx, ggml_type a_type, ggml_type b_type, uint32_t ਅੰਕ_ਖਾਨੇ_ਗਿਣਤੀ, uint32_t m, uint32_t k) {
    VK_LOG_DEBUG("ggml_vk_get_dequantize_mul_mat_vec()");
    GGML_ASSERT(b_type == GGML_TYPE_F32 || b_type == GGML_TYPE_F16 || b_type == GGML_TYPE_Q8_1);
    GGML_ASSERT(ਅੰਕ_ਖਾਨੇ_ਗਿਣਤੀ >= 1 && ਅੰਕ_ਖਾਨੇ_ਗਿਣਤੀ <= mul_mat_vec_max_ਖਾਨੇ_ਗਿਣਤੀ);

    ਜੇ (b_type == GGML_TYPE_Q8_1) {
        ਚੋਣ (a_type) {
            ਹਾਲਤ GGML_TYPE_Q2_0:
            ਹਾਲਤ GGML_TYPE_Q4_0:
            ਹਾਲਤ GGML_TYPE_Q4_1:
            ਹਾਲਤ GGML_TYPE_Q5_0:
            ਹਾਲਤ GGML_TYPE_Q5_1:
            ਹਾਲਤ GGML_TYPE_Q8_0:
            ਹਾਲਤ GGML_TYPE_MXFP4:
            ਹਾਲਤ GGML_TYPE_Q2_K:
            ਹਾਲਤ GGML_TYPE_Q3_K:
            ਹਾਲਤ GGML_TYPE_Q4_K:
            ਹਾਲਤ GGML_TYPE_Q5_K:
            ਹਾਲਤ GGML_TYPE_Q6_K:
            ਹਾਲਤ GGML_TYPE_IQ1_S:
            ਹਾਲਤ GGML_TYPE_IQ1_M:
            ਹਾਲਤ GGML_TYPE_IQ4_XS:
                ਰੋਕੋ;
            ਮੂਲ:
                ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
        }
    }

    ਚੋਣ (a_type) {
        ਹਾਲਤ GGML_TYPE_F32:
        ਹਾਲਤ GGML_TYPE_F16:
        ਹਾਲਤ GGML_TYPE_BF16:
        ਹਾਲਤ GGML_TYPE_Q1_0:
        ਹਾਲਤ GGML_TYPE_Q2_0:
        ਹਾਲਤ GGML_TYPE_Q4_0:
        ਹਾਲਤ GGML_TYPE_Q4_1:
        ਹਾਲਤ GGML_TYPE_Q5_0:
        ਹਾਲਤ GGML_TYPE_Q5_1:
        ਹਾਲਤ GGML_TYPE_Q8_0:
        ਹਾਲਤ GGML_TYPE_Q2_K:
        ਹਾਲਤ GGML_TYPE_Q3_K:
        ਹਾਲਤ GGML_TYPE_Q4_K:
        ਹਾਲਤ GGML_TYPE_Q5_K:
        ਹਾਲਤ GGML_TYPE_Q6_K:
        ਹਾਲਤ GGML_TYPE_IQ1_S:
        ਹਾਲਤ GGML_TYPE_IQ1_M:
        ਹਾਲਤ GGML_TYPE_IQ2_XXS:
        ਹਾਲਤ GGML_TYPE_IQ2_XS:
        ਹਾਲਤ GGML_TYPE_IQ2_S:
        ਹਾਲਤ GGML_TYPE_IQ3_XXS:
        ਹਾਲਤ GGML_TYPE_IQ3_S:
        ਹਾਲਤ GGML_TYPE_IQ4_XS:
        ਹਾਲਤ GGML_TYPE_IQ4_NL:
        ਹਾਲਤ GGML_TYPE_MXFP4:
        ਹਾਲਤ GGML_TYPE_NVFP4:
        ਹਾਲਤ GGML_TYPE_TQ1_0:
        ਹਾਲਤ GGML_TYPE_TQ2_0:
            ਰੋਕੋ;
        ਮੂਲ:
            ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }

    // heuristic to choose workgroup size
    uint32_t dmmv_wg = DMMV_WG_ਮਾਪ_SUBGROUP;
    ਜੇ ((ctx->device->vendor_id == VK_VENDOR_ID_NVIDIA && ctx->device->architecture != vk_device_architecture::NVIDIA_PRE_TURING) || ctx->device->vendor_id == VK_VENDOR_ID_INTEL) {
        // Prefer larger workgroups when M is small, to spread the work out more
        // and keep more SMs busy.
        // q6_k seems to prefer small workgroup size even for "medium" values of M.
        ਜੇ (a_type == GGML_TYPE_Q6_K) {
            ਜੇ (m < 4096 && k >= 1024) {
                dmmv_wg = DMMV_WG_ਮਾਪ_LARGE;
            }
        } ਨਹੀਂਤਾਂ {
            ਜੇ (m <= 8192 && k >= 1024) {
                dmmv_wg = DMMV_WG_ਮਾਪ_LARGE;
            }
        }
    }

    ਜੇ (b_type == GGML_TYPE_Q8_1) {
        ਜੇ (ctx->device->vendor_id == VK_VENDOR_ID_INTEL) {
            dmmv_wg = DMMV_WG_ਮਾਪ_SUBGROUP;
        }
        ਵਾਪਸ ctx->device->pipeline_dequant_mul_mat_vec_q8_1_f32[dmmv_wg][a_type][ਅੰਕ_ਖਾਨੇ_ਗਿਣਤੀ-1];
    }

    ਵਾਪਸ b_type == GGML_TYPE_F32 ? ctx->device->pipeline_dequant_mul_mat_vec_f32_f32[dmmv_wg][a_type][ਅੰਕ_ਖਾਨੇ_ਗਿਣਤੀ-1] : ctx->device->pipeline_dequant_mul_mat_vec_f16_f32[dmmv_wg][a_type][ਅੰਕ_ਖਾਨੇ_ਗਿਣਤੀ-1];
}

ਸਥਿਰ vk_pipeline ggml_vk_get_dequantize_mul_mat_vec_id(ggml_backend_vk_context * ctx, ggml_type a_type, ggml_type b_type, uint32_t m, uint32_t k) {
    VK_LOG_DEBUG("ggml_vk_get_dequantize_mul_mat_vec_id()");
    GGML_ASSERT(b_type == GGML_TYPE_F32 || b_type == GGML_TYPE_Q8_1);

    ਜੇ (b_type == GGML_TYPE_Q8_1) {
        ਚੋਣ (a_type) {
            ਹਾਲਤ GGML_TYPE_Q2_0:
            ਹਾਲਤ GGML_TYPE_Q4_0:
            ਹਾਲਤ GGML_TYPE_Q4_1:
            ਹਾਲਤ GGML_TYPE_Q5_0:
            ਹਾਲਤ GGML_TYPE_Q5_1:
            ਹਾਲਤ GGML_TYPE_Q8_0:
            ਹਾਲਤ GGML_TYPE_MXFP4:
            ਹਾਲਤ GGML_TYPE_Q2_K:
            ਹਾਲਤ GGML_TYPE_Q3_K:
            ਹਾਲਤ GGML_TYPE_Q4_K:
            ਹਾਲਤ GGML_TYPE_Q5_K:
            ਹਾਲਤ GGML_TYPE_Q6_K:
            ਹਾਲਤ GGML_TYPE_IQ1_S:
            ਹਾਲਤ GGML_TYPE_IQ1_M:
            ਹਾਲਤ GGML_TYPE_IQ4_XS:
                ਰੋਕੋ;
            ਮੂਲ:
                ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
        }
    }

    ਚੋਣ (a_type) {
        ਹਾਲਤ GGML_TYPE_F32:
        ਹਾਲਤ GGML_TYPE_F16:
        ਹਾਲਤ GGML_TYPE_BF16:
        ਹਾਲਤ GGML_TYPE_Q1_0:
        ਹਾਲਤ GGML_TYPE_Q2_0:
        ਹਾਲਤ GGML_TYPE_Q4_0:
        ਹਾਲਤ GGML_TYPE_Q4_1:
        ਹਾਲਤ GGML_TYPE_Q5_0:
        ਹਾਲਤ GGML_TYPE_Q5_1:
        ਹਾਲਤ GGML_TYPE_Q8_0:
        ਹਾਲਤ GGML_TYPE_Q2_K:
        ਹਾਲਤ GGML_TYPE_Q3_K:
        ਹਾਲਤ GGML_TYPE_Q4_K:
        ਹਾਲਤ GGML_TYPE_Q5_K:
        ਹਾਲਤ GGML_TYPE_Q6_K:
        ਹਾਲਤ GGML_TYPE_IQ1_S:
        ਹਾਲਤ GGML_TYPE_IQ1_M:
        ਹਾਲਤ GGML_TYPE_IQ2_XXS:
        ਹਾਲਤ GGML_TYPE_IQ2_XS:
        ਹਾਲਤ GGML_TYPE_IQ2_S:
        ਹਾਲਤ GGML_TYPE_IQ3_XXS:
        ਹਾਲਤ GGML_TYPE_IQ3_S:
        ਹਾਲਤ GGML_TYPE_IQ4_XS:
        ਹਾਲਤ GGML_TYPE_IQ4_NL:
        ਹਾਲਤ GGML_TYPE_MXFP4:
        ਹਾਲਤ GGML_TYPE_NVFP4:
        ਹਾਲਤ GGML_TYPE_TQ1_0:
        ਹਾਲਤ GGML_TYPE_TQ2_0:
            ਰੋਕੋ;
        ਮੂਲ:
            ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }

    // heuristic to choose workgroup size
    uint32_t dmmv_wg = DMMV_WG_ਮਾਪ_SUBGROUP;
    ਜੇ ((ctx->device->vendor_id == VK_VENDOR_ID_NVIDIA && ctx->device->architecture != vk_device_architecture::NVIDIA_PRE_TURING) || ctx->device->vendor_id == VK_VENDOR_ID_INTEL) {
        // Prefer larger workgroups when M is small, to spread the work out more
        // and keep more SMs busy.
        // q6_k seems to prefer small workgroup size even for "medium" values of M.
        ਜੇ (a_type == GGML_TYPE_Q6_K) {
            ਜੇ (m < 4096 && k >= 1024) {
                dmmv_wg = DMMV_WG_ਮਾਪ_LARGE;
            }
        } ਨਹੀਂਤਾਂ {
            ਜੇ (m <= 8192 && k >= 1024) {
                dmmv_wg = DMMV_WG_ਮਾਪ_LARGE;
            }
        }
    }

    ਜੇ (b_type == GGML_TYPE_Q8_1) {
        ਜੇ (ctx->device->vendor_id == VK_VENDOR_ID_INTEL) {
            dmmv_wg = DMMV_WG_ਮਾਪ_SUBGROUP;
        }
        ਵਾਪਸ ctx->device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[dmmv_wg][a_type];
    }

    ਵਾਪਸ ctx->device->pipeline_dequant_mul_mat_vec_id_f32[dmmv_wg][a_type];
}

vk_subbuffer ggml_vk_tensor_subbuffer(
    ਅਚਲ ggml_backend_vk_context * ctx, ਅਚਲ ggml_tensor * tensor, ਤਾਰਕਿਕ allow_misalign) {

    vk_ਬਫਰ ਬਫਰ = ਖਾਲੀ_ਸੂਚਕ;
    size_t ਵਿੱਥ = 0;
    ਜੇ (ctx->device->uma) {
        ggml_vk_host_get(ctx->device, tensor->ਅੰਕੜੇ, ਬਫਰ, ਵਿੱਥ);
    }
    ਜੇ (!ਬਫਰ) {
        ਸਵੈਚਾਲਿਤ buf_ctx = (ggml_backend_vk_ਬਫਰ_context *)tensor->ਬਫਰ->context;
        ਬਫਰ = buf_ctx->dev_ਬਫਰ;
        ਵਿੱਥ = vk_tensor_ਵਿੱਥ(tensor) + tensor->view_offs;
    }
    GGML_ASSERT(ਬਫਰ != ਖਾਲੀ_ਸੂਚਕ);

    size_t size = ggml_nbytes(tensor);

    ਅਚਲ size_t descriptor_ਵਿੱਥ = ggml_vk_descriptor_ਵਿੱਥ(
        ਵਿੱਥ, ctx->device->properties.limits.min_Storage_ਬਫਰ_ਵਿੱਥ_Alignment, ggml_type_ਮਾਪ(tensor->type));
    ਅਚਲ size_t misalign_ਬਾਈਟਾਂ = ਵਿੱਥ - descriptor_ਵਿੱਥ;
    // The shader must support misaligned offsets when indexing into the buffer
    GGML_ASSERT(allow_misalign || misalign_ਬਾਈਟਾਂ == 0);
    ਵਿੱਥ = descriptor_ਵਿੱਥ;
    size += misalign_ਬਾਈਟਾਂ;

    ਵਾਪਸ vk_subbuffer{ਬਫਰ, ਵਿੱਥ, size};
}

ਸਥਿਰ vk_command_ਬਫਰ* ggml_vk_get_or_create_cmd_ਬਫਰ(vk_device& device, vk_command_pool& pool) {
    ਲਈ (ਸਵੈਚਾਲਿਤ& cmd_ਬਫਰ : pool.cmd_buffers) {
        ਜੇ (!cmd_ਬਫਰ.ਵਿੱਚ_use) {
            cmd_ਬਫਰ.use_counter++;
            cmd_ਬਫਰ.ਵਿੱਚ_use = ਸੱਚ;
            ਵਾਪਸ &cmd_ਬਫਰ;
        }
    }
    ਵਾਪਸ ggml_vk_create_cmd_ਬਫਰ(device, pool);
}

ਸਥਿਰ vk_submission ggml_vk_begin_submission(vk_device& device, vk_command_pool& p, ਤਾਰਕਿਕ one_ਸਮਾਂ = ਸੱਚ) {
    vk_submission s;
    s.ਬਫਰ = ggml_vk_get_or_create_cmd_ਬਫਰ(device, p);
    ਜੇ (one_ਸਮਾਂ) {
        s.ਬਫਰ->buf.begin({ vk::Command_ਬਫਰ_Usage_Flag_ਬਿੱਟ::e_One_ਸਮਾਂ_Submit });
    } ਨਹੀਂਤਾਂ {
        s.ਬਫਰ->buf.begin({ vk::Command_ਬਫਰ_Usage_Flags{} });
    }

    ਵਾਪਸ s;
}

ਖਾਲੀ ggml_vk_cmd_label_begin(vk::Command_ਬਫਰ buf, ਅਚਲ ਅੱਖਰ * name) {
    vk::DebugUtilsLabelEXT label = {};
    label.pLabelName = name;
    label.color = std::array<ਦਸ਼ਮਲਵ, 4>{1.0f, 1.0f, 1.0f, 1.0f};
    vk_instance.pfn_vkCmdBeginDebugUtilsLabelEXT(buf, reinterpret_cast<VkDebugUtilsLabelEXT *>(&label));
}

ਖਾਲੀ ggml_vk_ctx_ਸਿਰਾ(vk_context& ctx) {
    VK_LOG_DEBUG("ggml_vk_ctx_end(" << ctx << ", " << ctx->seqs.size() << ")");
    ਜੇ (ctx->s == ਖਾਲੀ_ਸੂਚਕ) {
        ਵਾਪਸ;
    }

    // close open labels so this buffer is balanced; reopened in ggml_vk_ctx_begin
    ਜੇ (vk_instance.debug_utils_support) {
        ਲਈ (size_t i = 0; i < ctx->debug_labels.size(); i++) {
            vk_instance.pfn_vk_Cmd_ਸਿਰਾ_Debug_Utils_Label_EXT(ctx->s->ਬਫਰ->buf);
        }
        // the enclosing per-command-buffer region
        vk_instance.pfn_vk_Cmd_ਸਿਰਾ_Debug_Utils_Label_EXT(ctx->s->ਬਫਰ->buf);
    }

    ctx->s->ਬਫਰ->buf.end();
    ctx->s = ਖਾਲੀ_ਸੂਚਕ;
}

ਖਾਲੀ ggml_vk_ctx_begin(vk_device& device, vk_context& subctx) {
    VK_LOG_DEBUG("ggml_vk_ctx_begin(" << device->name << ")");
    ਜੇ (subctx->s != ਖਾਲੀ_ਸੂਚਕ) {
        ggml_vk_ctx_ਸਿਰਾ(subctx);
    }

    subctx->seqs.push_back({ ggml_vk_begin_submission(device, *subctx->p) });
    subctx->s = subctx->seqs[subctx->seqs.size() - 1].ਅੰਕੜੇ();

    ਜੇ (vk_instance.debug_utils_support) {
        // outermost region, one per command buffer, so the gaps between submits stand out
        ਅਚਲ std::string name = "submit " + std::to_string(device->debug_cmdbuf_idx++);
        ggml_vk_cmd_label_begin(subctx->s->ਬਫਰ->buf, name.c_str());

        // reopen labels left open when the previous command buffer was submitted
        ਲਈ (ਅਚਲ std::string & label : subctx->debug_labels) {
            ggml_vk_cmd_label_begin(subctx->s->ਬਫਰ->buf, label.c_str());
        }
    }
}

vk_context ggml_vk_get_compute_ctx(ggml_backend_vk_context * ctx) {
    vk_context ਨਤੀਜਾ;
    ਜੇ (!ctx->compute_ctx.expired()) {
        ਨਤੀਜਾ = ctx->compute_ctx.ਤਾਲਾ();
    } ਨਹੀਂਤਾਂ {
        ਨਤੀਜਾ = ggml_vk_create_context(ctx, ctx->compute_cmd_pool);

        ctx->compute_ctx = ਨਤੀਜਾ;
        ggml_vk_ctx_begin(ctx->device, ਨਤੀਜਾ);
    }

    ਜੇ (ctx->device->async_use_transfer_ਕਤਾਰ && ctx->transfer_semaphore_ਆਖਰੀ_submitted < ctx->transfer_semaphore.value) {
        ਨਤੀਜਾ->s->wait_semaphores.push_back(ctx->transfer_semaphore);
        ctx->transfer_semaphore_ਆਖਰੀ_submitted = ctx->transfer_semaphore.value;
    }

    ਵਾਪਸ ਨਤੀਜਾ;
}

vk_context ggml_vk_get_transfer_ctx(ggml_backend_vk_context * ctx) {
    vk_context ਨਤੀਜਾ;
    ਜੇ (!ctx->transfer_ctx.expired()) {
        ਨਤੀਜਾ = ctx->transfer_ctx.ਤਾਲਾ();
    } ਨਹੀਂਤਾਂ {
        ਨਤੀਜਾ = ggml_vk_create_context(ctx, ctx->transfer_cmd_pool);

        ctx->transfer_ctx = ਨਤੀਜਾ;
        ggml_vk_ctx_begin(ctx->device, ਨਤੀਜਾ);
    }

    ਵਾਪਸ ਨਤੀਜਾ;
}

ਤਾਰਕਿਕ ggml_vk_submit_transfer_ctx(ggml_backend_vk_context * ctx) {
    ਜੇ (!ctx->device->async_use_transfer_ਕਤਾਰ || ctx->transfer_ctx.expired()) {
        ਵਾਪਸ ਝੂਠ;
    }

    vk_context cpy_ctx = ctx->transfer_ctx.ਤਾਲਾ();
    ggml_vk_ctx_ਸਿਰਾ(cpy_ctx);

    ਲਈ (ਸਵੈਚਾਲਿਤ& cpy : cpy_ctx->ਵਿੱਚ_memcpys) {
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(cpy.dst, cpy.src, cpy.n);
    }

    ctx->transfer_semaphore.value++;
    cpy_ctx->seqs.back().back().signal_semaphores.push_back(ctx->transfer_semaphore);

    ggml_vk_submit(cpy_ctx, {});
    ctx->transfer_ctx.reset();
    ਵਾਪਸ ਸੱਚ;
}

size_t ggml_vk_align_ਮਾਪ(size_t ਚੌੜਾਈ, size_t align) {
    VK_LOG_DEBUG("ggml_vk_align_size(" << ਚੌੜਾਈ << ", " << align << ")");
    ਵਾਪਸ CEIL_DIV(ਚੌੜਾਈ, align) * align;
}

ਖਾਲੀ deferred_memcpy(ਖਾਲੀ * dst, ਅਚਲ ਖਾਲੀ * src, size_t size, std::vector<vk_staging_memcpy>* memcpys) {
    ਜੇ (memcpys == ਖਾਲੀ_ਸੂਚਕ) {
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(dst, src, size);
    } ਨਹੀਂਤਾਂ {
        memcpys->emplace_back(dst, src, size);
    }
}

ਖਾਲੀ deferred_memset(ਖਾਲੀ * dst, uint32_t val, size_t size, std::vector<vk_staging_memset>* memsets) {
    ਜੇ (memsets == ਖਾਲੀ_ਸੂਚਕ) {
        memset(dst, val, size);
    } ਨਹੀਂਤਾਂ {
        memsets->emplace_back(dst, val, size);
    }
}

ਸਥਿਰ uint32_t ggml_vk_guess_ਵੰਡੋ_k(ggml_backend_vk_context * ctx, uint32_t m, uint32_t n, uint32_t k, ਤਾਰਕਿਕ disable_ਵੰਡੋ_k, ਅਚਲ vk_pipeline& pipeline) {
    VK_LOG_DEBUG("ggml_vk_guess_split_k(" << m << ", " << n << ", " << k << ", " << disable_ਵੰਡੋ_k << ")");

    ਜੇ (disable_ਵੰਡੋ_k) {
        ਵਾਪਸ 1;
    }

    uint32_t ਵੰਡੋ_k = 1;
    ਜੇ (ctx->device->shader_core_ਗਿਣਤੀ != 0 && n >= pipeline->wg_denoms[1]) {
        // If k is 'large' and the SMs will fill less than halfway, use split_k.
        uint32_t m_tiles = CEIL_DIV(m, pipeline->wg_denoms[0]);
        uint32_t n_tiles = CEIL_DIV(n, pipeline->wg_denoms[1]);

        ਜੇ (k >= 2048) {
            ਜੇ (m_tiles * n_tiles <= ctx->device->shader_core_ਗਿਣਤੀ / 2) {
                ਵੰਡੋ_k = ctx->device->shader_core_ਗਿਣਤੀ / (m_tiles * n_tiles);
            } ਨਹੀਂਤਾਂ ਜੇ (m_tiles * n_tiles <= ctx->device->shader_core_ਗਿਣਤੀ * 2 / 3) {
                ਵੰਡੋ_k = 3;
            }
            // Cap the split at 8x. Unless k is huge this is a lot of overhead.
            ਵੰਡੋ_k = std::min(ਵੰਡੋ_k, 8u);

            // ggml_vk_matmul will align the splits to be a multiple of 256.
            // If this rounded up size would cause the last split to be empty,
            // then reduce the split count.
            ਜਦੋਂ (ਸੱਚ) {
                ਜੇ (ਵੰਡੋ_k == 1) {
                    ਰੋਕੋ;
                }
                uint32_t k_ਵੰਡੋ = CEIL_DIV(k, ਵੰਡੋ_k);
                k_ਵੰਡੋ = ROUNDUP_POW2(k_ਵੰਡੋ, 256);
                ਜੇ (k_ਵੰਡੋ * (ਵੰਡੋ_k - 1) < k) {
                    ਰੋਕੋ;
                }
                ਵੰਡੋ_k--;
            }
        }
    }

    ਵਾਪਸ ਵੰਡੋ_k;
}

ਖਾਲੀ ggml_vk_matmul(
        ggml_backend_vk_context * ctx, vk_context& subctx, vk_pipeline& pipeline,
        vk_subbuffer&& a, vk_subbuffer&& b, vk_subbuffer&& d, vk_subbuffer&& ਵੰਡੋ_k_ਬਫਰ,
        uint32_t m, uint32_t n, uint32_t k, uint32_t stride_a, uint32_t stride_b, uint32_t stride_d,
        uint32_t batch_stride_a, uint32_t batch_stride_b, uint32_t batch_stride_d,
        uint32_t ਵੰਡੋ_k, uint32_t batch, uint32_t ne02, uint32_t ne12, uint32_t broadcast2, uint32_t broadcast3,
        uint32_t padded_n, uint64_t b_ਖਾਨਾ_ਬਾਈਟਾਂ, uint64_t d_ਖਾਨਾ_ਬਾਈਟਾਂ) {
        VK_LOG_DEBUG("ggml_vk_matmul(a: (" << a.ਬਫਰ->ਬਫਰ << ", " << a.ਵਿੱਥ << ", " << a.size << "), b: (" << b.ਬਫਰ->ਬਫਰ << ", " << b.ਵਿੱਥ << ", " << b.size << "), d: (" << d.ਬਫਰ->ਬਫਰ << ", " << d.ਵਿੱਥ << ", " << d.size << "), split_k: (" << (ਵੰਡੋ_k_ਬਫਰ.ਬਫਰ != ਖਾਲੀ_ਸੂਚਕ ? ਵੰਡੋ_k_ਬਫਰ.ਬਫਰ->ਬਫਰ : VK_NULL_HANDLE) << ", " << ਵੰਡੋ_k_ਬਫਰ.ਵਿੱਥ << ", " << ਵੰਡੋ_k_ਬਫਰ.size << "), m: " << m << ", n: " << n << ", k: " << k << ", stride_a: " << stride_a << ", stride_b: " << stride_b << ", stride_d: " << stride_d << ", batch_stride_a: " << batch_stride_a << ", batch_stride_b: " << batch_stride_b << ", batch_stride_d: " << batch_stride_d << ", split_k: " << ਵੰਡੋ_k << ", batch: " << batch << ", ne02: " << ne02 << ", ne12: " << ne12 << ", broadcast2: " << broadcast2 << ", broadcast3: " << broadcast3 << ", padded_n: " << padded_n << ")");
    ਜੇ (ਵੰਡੋ_k == 1) {
        // a small tile can still produce more workgroups than the device allows on the n axis:
        // split the dispatch into n-chunks, shifting the B and D column windows
        ਅਚਲ uint64_t max_n = (uint64_t)ctx->device->properties.limits.max_Compute_Work_ਸਮੂਹ_ਗਿਣਤੀ[1] * pipeline->wg_denoms[1];
        ਅਚਲ uint64_t n_chunks = CEIL_DIV(n, max_n);
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, n_chunks * CEIL_DIV(batch, ctx->device->properties.limits.max_Compute_Work_ਸਮੂਹ_ਗਿਣਤੀ[2]));

        ਲਈ (uint64_t n_ਅਧਾਰ = 0; n_ਅਧਾਰ < n; n_ਅਧਾਰ += max_n) {
            ਅਚਲ uint32_t n_chunk = (uint32_t) std::min(n - n_ਅਧਾਰ, max_n);
            GGML_ASSERT(n_ਅਧਾਰ * b_ਖਾਨਾ_ਬਾਈਟਾਂ < b.size && n_ਅਧਾਰ * d_ਖਾਨਾ_ਬਾਈਟਾਂ < d.size);
            GGML_ASSERT(b.size != VK_ਸਾਰਾ_ਮਾਪ && d.size != VK_ਸਾਰਾ_ਮਾਪ);
            vk_subbuffer b_chunk = b;
            vk_subbuffer d_chunk = d;
            b_chunk.ਵਿੱਥ += n_ਅਧਾਰ * b_ਖਾਨਾ_ਬਾਈਟਾਂ;
            b_chunk.size   -= n_ਅਧਾਰ * b_ਖਾਨਾ_ਬਾਈਟਾਂ;
            d_chunk.ਵਿੱਥ += n_ਅਧਾਰ * d_ਖਾਨਾ_ਬਾਈਟਾਂ;
            d_chunk.size   -= n_ਅਧਾਰ * d_ਖਾਨਾ_ਬਾਈਟਾਂ;

            uint32_t ਅਧਾਰ_work_ਸਮੂਹ_z = 0;
            ਜਦੋਂ (ਅਧਾਰ_work_ਸਮੂਹ_z < batch) {
                uint32_t groups_z = std::min(batch - ਅਧਾਰ_work_ਸਮੂਹ_z, ctx->device->properties.limits.max_Compute_Work_ਸਮੂਹ_ਗਿਣਤੀ[2]);

                ਅਚਲ vk_mat_mat_ਧੱਕੋ_constants pc = { m, n_chunk, k, stride_a, stride_b, stride_d, batch_stride_a, batch_stride_b, batch_stride_d, ਅਧਾਰ_work_ਸਮੂਹ_z, batch, k, ne02, ne12, broadcast2, broadcast3, uint32_t(padded_n - n_ਅਧਾਰ) };
                ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { a, b_chunk, d_chunk }, pc, { m, n_chunk, groups_z });
                ਅਧਾਰ_work_ਸਮੂਹ_z += groups_z;
            }
        }
        ਵਾਪਸ;
    }

    ਜੇ (ctx->prealloc_ਵੰਡੋ_k_need_sync) {
        ggml_vk_sync_buffers(ctx, subctx);
    }

    GGML_ASSERT(batch_stride_d == m * n);

    // Round the split size up to a multiple of 256 (k-quant alignment)
    uint32_t k_ਵੰਡੋ = CEIL_DIV(k, ਵੰਡੋ_k);
    k_ਵੰਡੋ = ROUNDUP_POW2(k_ਵੰਡੋ, 256);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, CEIL_DIV(batch, ctx->device->properties.limits.max_Compute_Work_ਸਮੂਹ_ਗਿਣਤੀ[2]));

    uint32_t ਅਧਾਰ_work_ਸਮੂਹ_z = 0;
    ਜਦੋਂ (ਅਧਾਰ_work_ਸਮੂਹ_z < batch) {
        uint32_t groups_z = std::min(batch - ਅਧਾਰ_work_ਸਮੂਹ_z, ctx->device->properties.limits.max_Compute_Work_ਸਮੂਹ_ਗਿਣਤੀ[2]);

        ਅਚਲ vk_mat_mat_ਧੱਕੋ_constants pc1 = { m, n, k, stride_a, stride_b, stride_d, batch_stride_a, batch_stride_b, batch_stride_d, ਅਧਾਰ_work_ਸਮੂਹ_z, batch, k_ਵੰਡੋ, ne02, ne12, broadcast2, broadcast3, padded_n };
        // Make sure enough workgroups get assigned for split k to work
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { a, b, ਵੰਡੋ_k_ਬਫਰ }, pc1, { (CEIL_DIV(m, pipeline->wg_denoms[0]) * pipeline->wg_denoms[0]) * ਵੰਡੋ_k, n, groups_z });
        ਅਧਾਰ_work_ਸਮੂਹ_z += groups_z;
    }
    ggml_vk_sync_buffers(ctx, subctx);
    ਅਚਲ std::array<uint32_t, 2> pc2 = { (uint32_t)(m * n * batch), ਵੰਡੋ_k };
    ggml_vk_dispatch_pipeline(ctx, subctx, ctx->device->pipeline_matmul_ਵੰਡੋ_k_reduce, { ਵੰਡੋ_k_ਬਫਰ, d }, pc2, { m * n * batch, 1, 1 });
    ctx->prealloc_ਵੰਡੋ_k_need_sync = ਸੱਚ;
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_vk_get_mul_mat_mat_f16acc(ggml_backend_vk_context * ctx, ggml_type src0_type, ggml_type src1_type, ggml_prec prec) {
    ਜੇ (src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_BF16) ਵਾਪਸ ਝੂਠ;
    ਜੇ (src1_type == GGML_TYPE_Q8_1) ਵਾਪਸ ਝੂਠ;
    ਜੇ (src0_type == GGML_TYPE_F16) {
        ਵਾਪਸ prec == GGML_PREC_DEFAULT && ctx->device->fp16 && !(ctx->device->coopmat_support && !ctx->device->coopmat_acc_f16_support);
    }
    // quant types
    ਜੇ (ctx->device->coopmat2) {
        ਵਾਪਸ prec == GGML_PREC_DEFAULT;
    }
    ਜੇ (ctx->device->coopmat_support) {
        ਵਾਪਸ ctx->device->fp16 && ctx->device->coopmat_acc_f16_support && prec == GGML_PREC_DEFAULT;
    }
    ਵਾਪਸ ctx->device->fp16 && prec == GGML_PREC_DEFAULT;
}

ਸਥਿਰ ਅਚਲ std::vector<vk_matmul_pipeline_pair>* ggml_vk_get_mul_mat_mat_pipeline_map(
        ggml_backend_vk_context * ctx, ggml_type src0_type, ggml_type src1_type, ggml_prec prec, ਤਾਰਕਿਕ mul_mat_id = ਝੂਠ) {
    ਤਾਰਕਿਕ f16acc = ggml_vk_get_mul_mat_mat_f16acc(ctx, src0_type, src1_type, prec);
    vk_matmul_pipeline_ਕੁੰਜੀ ਕੁੰਜੀ{src0_type, src1_type, mul_mat_id, f16acc};
    ਸਵੈਚਾਲਿਤ it = ctx->device->pipeline_matmul.find(ਕੁੰਜੀ);
    ਜੇ (it == ctx->device->pipeline_matmul.end() || it->second.ਖਾਲੀ_ਥਾਵਾਂ()) {
        // Try without f16acc
        ਜੇ (f16acc) {
            ਕੁੰਜੀ.f16acc = ਝੂਠ;
            it = ctx->device->pipeline_matmul.find(ਕੁੰਜੀ);
            ਜੇ (it != ctx->device->pipeline_matmul.end() && !it->second.ਖਾਲੀ_ਥਾਵਾਂ()) ਵਾਪਸ &it->second;
        }
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }
    ਵਾਪਸ &it->second;
}

ਸਥਿਰ vk_pipeline ggml_vk_guess_matmul_pipeline_map(ggml_backend_vk_context * ctx,
        ਅਚਲ std::vector<vk_matmul_pipeline_pair>& configs,
        uint32_t m, uint32_t n, ਤਾਰਕਿਕ aligned, ਤਾਰਕਿਕ mul_mat_id) {
    ਸਵੈਚਾਲਿਤ& selector = mul_mat_id ? ctx->device->matmul_id_tile_selector : ctx->device->matmul_tile_selector;
    uint32_t idx = selector(m, n, 0, ctx->device->shader_core_ਗਿਣਤੀ, configs);
    ਜੇ (idx >= configs.size()) idx = (uint32_t)configs.size() - 1;
    ਵਾਪਸ (aligned && configs[idx].aligned) ? configs[idx].aligned : configs[idx].unaligned;
}

ਸਥਿਰ uint32_t ggml_vk_guess_matmul_pipeline_align_map(ggml_backend_vk_context * ctx,
        ਅਚਲ std::vector<vk_matmul_pipeline_pair>& configs,
        uint32_t m, uint32_t n, ਤਾਰਕਿਕ mul_mat_id) {
    ਸਵੈਚਾਲਿਤ& selector = mul_mat_id ? ctx->device->matmul_id_tile_selector : ctx->device->matmul_tile_selector;
    uint32_t idx = selector(m, n, 0, ctx->device->shader_core_ਗਿਣਤੀ, configs);
    ਜੇ (idx >= configs.size()) idx = (uint32_t)configs.size() - 1;
    ਵਾਪਸ configs[idx].align;
}

ਸਥਿਰ uint64_t ggml_vk_mul_mat_id_max_tiles(uint64_t n_experts, uint64_t n_ਪੰਕਤੀਆਂ, uint32_t bn) {
    // Each nonempty expert can add at most one partial tile to the combined grid.
    ਵਾਪਸ std::min(n_ਪੰਕਤੀਆਂ, CEIL_DIV(n_ਪੰਕਤੀਆਂ, bn) + std::min(n_experts, n_ਪੰਕਤੀਆਂ) - 1);
}

ਸਥਿਰ ਖਾਲੀ ggml_vk_matmul_id(
        ggml_backend_vk_context * ctx, vk_context& subctx, vk_pipeline& pipeline,
        vk_subbuffer&& a, vk_subbuffer&& b, vk_subbuffer&& d, vk_subbuffer&& ids, ਅਚਲ vk_subbuffer & expert_ਗਿਣਤੀ_buf,
        uint32_t m, uint32_t n, uint32_t k, uint32_t stride_a, uint32_t stride_b, uint32_t stride_d,
        uint32_t batch_stride_a, uint32_t batch_stride_b, uint32_t batch_stride_d,
        uint32_t n_as, uint32_t nei0, uint32_t nei1, uint32_t nbi1, uint32_t ne11,
        uint32_t ਪੰਕਤੀ_ids_ਵਿੱਥ, uint32_t max_tiles) {
    VK_LOG_DEBUG("ggml_vk_matmul_id(a: (" << a.ਬਫਰ->ਬਫਰ << ", " << a.ਵਿੱਥ << ", " << a.size << "), b: (" << b.ਬਫਰ->ਬਫਰ << ", " << b.ਵਿੱਥ << ", " << b.size << "), d: (" << d.ਬਫਰ->ਬਫਰ << ", " << d.ਵਿੱਥ << ", " << d.size << "), ids: (" << ids.ਬਫਰ->ਬਫਰ << ", " << ids.ਵਿੱਥ << ", " << ids.size << "), expert_count: (" << expert_ਗਿਣਤੀ_buf.ਬਫਰ->ਬਫਰ << ", " << expert_ਗਿਣਤੀ_buf.ਵਿੱਥ << ", " << expert_ਗਿਣਤੀ_buf.size << "), " <<
        "m: " << m << ", n: " << n << ", k: " << k << ", stride_a: " << stride_a << ", stride_b: " << stride_b << ", stride_d: " << stride_d << ", " <<
        "batch_stride_a: " << batch_stride_a << ", batch_stride_b: " << batch_stride_b << ", batch_stride_d: " << batch_stride_d << ", " <<
        "n_as: " << n_as << ", nei0: " << nei0 << ", nei1: " << nei1 << ", nbi1: " << nbi1 << ", ne11: " << ne11 << ")");
    ਅਚਲ vk_mat_mat_id_ਧੱਕੋ_constants pc = { m, n, k, stride_a, stride_b, stride_d, batch_stride_a, batch_stride_b, batch_stride_d,
                                              nei0, nei1, nbi1, ne11, n_as, ਪੰਕਤੀ_ids_ਵਿੱਥ };
    ਅਚਲ uint32_t tiles_y = std::min(max_tiles, ctx->device->properties.limits.max_Compute_Work_ਸਮੂਹ_ਗਿਣਤੀ[1]);
    ਅਚਲ std::array<uint32_t, 3> elements = { m, tiles_y * pipeline->wg_denoms[1], uint32_t(CEIL_DIV(max_tiles, tiles_y)) };
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { a, b, d, ids, expert_ਗਿਣਤੀ_buf }, pc, elements);
}

ਤਾਰਕਿਕ ggml_vk_dim01_contiguous(ਅਚਲ ggml_tensor * tensor) {
    ਵਾਪਸ
        tensor->nb[0] == ggml_type_ਮਾਪ(tensor->type) &&
        tensor->nb[1] == (tensor->nb[0]*tensor->ne[0])/ggml_blck_ਮਾਪ(tensor->type) &&
        (tensor->ne[3] == 1 || tensor->nb[3] == tensor->nb[2]*tensor->ne[2]);
}

// Batch stride in elements of a tensor read in place.
ਸਥਿਰ uint32_t ggml_vk_batch_stride(ਅਚਲ ggml_tensor * tensor) {
    ਵਾਪਸ (uint32_t)(tensor->nb[2] / ggml_type_ਮਾਪ(tensor->type) * ggml_blck_ਮਾਪ(tensor->type));
}

vk_pipeline ggml_vk_get_cpy_pipeline(ggml_backend_vk_context * ctx, ਅਚਲ ggml_tensor * src, ਅਚਲ ggml_tensor * dst, ggml_type ਮੰਜ਼ਿਲ) {

    // Choose "contiguous copy" shader if src/dst are contiguous
    ਤਾਰਕਿਕ contig = ggml_is_contiguous(src) && (!dst || ggml_is_contiguous(dst));

    // Use optimized "transpose" shader if src dim1 is the innermost dimension.
    ਤਾਰਕਿਕ transpose = dst && src->nb[1] == ggml_type_ਮਾਪ(ਮੰਜ਼ਿਲ) && ggml_are_same_shape(dst, src);

    ਜੇ (transpose && src->type == ਮੰਜ਼ਿਲ) {
        ਜੇ (ggml_type_ਮਾਪ(ਮੰਜ਼ਿਲ) == 4) {
            ਵਾਪਸ ctx->device->pipeline_cpy_transpose_32;
        } ਨਹੀਂਤਾਂ ਜੇ (ggml_type_ਮਾਪ(ਮੰਜ਼ਿਲ) == 2) {
            ਵਾਪਸ ctx->device->pipeline_cpy_transpose_16;
        }
    }

    // Same, for a 0<->2 swap: src dim2 is the innermost dimension.
    ਤਾਰਕਿਕ transpose02 = dst && !contig && src->nb[2] == ggml_type_ਮਾਪ(ਮੰਜ਼ਿਲ) &&
                       ggml_is_contiguous(dst) && ggml_are_same_shape(dst, src);

    ਜੇ (transpose02 && src->type == ਮੰਜ਼ਿਲ) {
        ਜੇ (ggml_type_ਮਾਪ(ਮੰਜ਼ਿਲ) == 4) {
            ਵਾਪਸ ctx->device->pipeline_cpy_transpose_02_32;
        } ਨਹੀਂਤਾਂ ਜੇ (ggml_type_ਮਾਪ(ਮੰਜ਼ਿਲ) == 2) {
            ਵਾਪਸ ctx->device->pipeline_cpy_transpose_02_16;
        }
    }

    ਜੇ (src->type == GGML_TYPE_F32 && ਮੰਜ਼ਿਲ == GGML_TYPE_F16) {
        ਜੇ (contig) {
            ਵਾਪਸ ctx->device->pipeline_contig_cpy_f32_f16;
        } ਨਹੀਂਤਾਂ {
            ਵਾਪਸ ctx->device->pipeline_cpy_f32_f16;
        }
    }
    ਜੇ (src->type == GGML_TYPE_F16 && ਮੰਜ਼ਿਲ == GGML_TYPE_F32) {
        ਜੇ (contig) {
            ਵਾਪਸ ctx->device->pipeline_contig_cpy_f16_f32;
        } ਨਹੀਂਤਾਂ {
            ਵਾਪਸ ctx->device->pipeline_cpy_f16_f32;
        }
    }
    ਜੇ (src->type == GGML_TYPE_F32 && ਮੰਜ਼ਿਲ == GGML_TYPE_BF16) {
        ਜੇ (contig) {
            ਵਾਪਸ ctx->device->pipeline_contig_cpy_f32_bf16;
        } ਨਹੀਂਤਾਂ {
            ਵਾਪਸ ctx->device->pipeline_cpy_f32_bf16;
        }
    }
    ਜੇ (src->type == GGML_TYPE_BF16 && ਮੰਜ਼ਿਲ == GGML_TYPE_F32) {
        ਜੇ (contig) {
            ਵਾਪਸ ctx->device->pipeline_contig_cpy_bf16_f32;
        } ਨਹੀਂਤਾਂ {
            ਵਾਪਸ ctx->device->pipeline_cpy_bf16_f32;
        }
    }
    ਜੇ (src->type == GGML_TYPE_BF16 && ਮੰਜ਼ਿਲ == GGML_TYPE_F16) {
        ਜੇ (contig) {
            ਵਾਪਸ ctx->device->pipeline_contig_cpy_bf16_f16;
        } ਨਹੀਂਤਾਂ {
            ਵਾਪਸ ctx->device->pipeline_cpy_bf16_f16;
        }
    }
    ਜੇ (src->type == GGML_TYPE_F32 && ਮੰਜ਼ਿਲ == GGML_TYPE_I32) {
        ਜੇ (contig) {
            ਵਾਪਸ ctx->device->pipeline_contig_cpy_f32_i32;
        } ਨਹੀਂਤਾਂ {
            ਵਾਪਸ ctx->device->pipeline_cpy_f32_i32;
        }
    }
    ਜੇ (src->type == GGML_TYPE_I32 && ਮੰਜ਼ਿਲ == GGML_TYPE_F32) {
        ਜੇ (contig) {
            ਵਾਪਸ ctx->device->pipeline_contig_cpy_i32_f32;
        } ਨਹੀਂਤਾਂ {
            ਵਾਪਸ ctx->device->pipeline_cpy_i32_f32;
        }
    }
    ਜੇ (src->type == GGML_TYPE_F32) {
        ਚੋਣ (ਮੰਜ਼ਿਲ) {
        ਹਾਲਤ GGML_TYPE_Q1_0:
        ਹਾਲਤ GGML_TYPE_Q2_0:
        ਹਾਲਤ GGML_TYPE_Q4_0:
        ਹਾਲਤ GGML_TYPE_Q4_1:
        ਹਾਲਤ GGML_TYPE_Q5_0:
        ਹਾਲਤ GGML_TYPE_Q5_1:
        ਹਾਲਤ GGML_TYPE_Q8_0:
        ਹਾਲਤ GGML_TYPE_IQ4_NL:
            ਵਾਪਸ ctx->device->pipeline_cpy_f32_quant[ਮੰਜ਼ਿਲ];
        ਮੂਲ:
            ਰੋਕੋ;
        }
    }

    ਜੇ (ਮੰਜ਼ਿਲ == GGML_TYPE_F32) {
        ਚੋਣ (src->type) {
        ਹਾਲਤ GGML_TYPE_Q1_0:
        ਹਾਲਤ GGML_TYPE_Q2_0:
        ਹਾਲਤ GGML_TYPE_Q4_0:
        ਹਾਲਤ GGML_TYPE_Q4_1:
        ਹਾਲਤ GGML_TYPE_Q5_0:
        ਹਾਲਤ GGML_TYPE_Q5_1:
        ਹਾਲਤ GGML_TYPE_Q8_0:
        ਹਾਲਤ GGML_TYPE_IQ4_NL:
            ਵਾਪਸ ctx->device->pipeline_cpy_quant_f32[src->type];
        ਮੂਲ:
            ਰੋਕੋ;
        }
    }

    ਜੇ (src->type == ਮੰਜ਼ਿਲ) {
        // Copy two or four bytes at a time, depending on block size.
        // For quantized types, we scale by block size/type size. But
        // this path is also used for bf16->bf16 for example, where the
        // type size must be exactly 2 or 4.
        GGML_ASSERT(ggml_is_quantized(ਮੰਜ਼ਿਲ) || ggml_type_ਮਾਪ(src->type) == 2 || ggml_type_ਮਾਪ(src->type) == 4);
        ਜੇ ((ggml_type_ਮਾਪ(src->type) % 4) == 0) {
            ਜੇ (contig) {
                ਵਾਪਸ ctx->device->pipeline_contig_cpy_32;
            } ਨਹੀਂਤਾਂ {
                ਵਾਪਸ ctx->device->pipeline_cpy_32;
            }
        } ਨਹੀਂਤਾਂ {
            ਜੇ (contig) {
                ਵਾਪਸ ctx->device->pipeline_contig_cpy_16;
            } ਨਹੀਂਤਾਂ {
                ਵਾਪਸ ctx->device->pipeline_cpy_16;
            }
        }
    }

    std::cerr << "Missing CPY op for types: " << ggml_type_name(src->type) << " " << ggml_type_name(ਮੰਜ਼ਿਲ) << std::ਸਤਰ_ਅੰਤ;
    GGML_ABORT("fatal error");
}

ਖਾਲੀ ggml_vk_cpy_ਮੰਜ਼ਿਲ_contiguous(ggml_backend_vk_context * ctx, vk_context& subctx, vk_pipeline pipeline, ਅਚਲ ggml_tensor * tensor, ਅਚਲ vk_subbuffer & ਵਿੱਚ, ਅਚਲ vk_subbuffer & ਬਾਹਰਲਾ) {
    VK_LOG_DEBUG("ggml_vk_cpy_to_contiguous((" << tensor << ", type=" << tensor->type << ", ne0=" << tensor->ne[0] << ", ne1=" << tensor->ne[1] << ", ne2=" << tensor->ne[2] << ", ne3=" << tensor->ne[3] << ", nb0=" << tensor->nb[0] << ", nb1=" << tensor->nb[1] << ", nb2=" << tensor->nb[2] << ", nb3=" << tensor->nb[3] << "), ";
    std::cerr << "buffer in size=" << ਵਿੱਚ.ਬਫਰ->size << ", buffer out size=" << ਬਾਹਰਲਾ.ਬਫਰ->size << ")");

    ਅਚਲ uint32_t ne = ggml_nelements(tensor);
    std::array<uint32_t, 3> elements;

    ਜੇ (ne > 262144) {
        elements = { 512, 512, CEIL_DIV(ne, 262144) };
    } ਨਹੀਂਤਾਂ ਜੇ (ne > 512) {
        elements = { 512, CEIL_DIV(ne, 512), 1 };
    } ਨਹੀਂਤਾਂ {
        elements = { ne, 1, 1 };
    }

    vk_op_unary_ਧੱਕੋ_constants pc = vk_op_unary_ਧੱਕੋ_constants_init(tensor, tensor, ne);
    pc.nb10 = 1;
    pc.nb11 = (uint32_t)tensor->ne[0];
    pc.nb12 = (uint32_t)(tensor->ne[0] * tensor->ne[1]);
    pc.nb13 = (uint32_t)(tensor->ne[0] * tensor->ne[1] * tensor->ne[2]);
    init_pushconst_fastdiv(pc);
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { ਵਿੱਚ, ਬਾਹਰਲਾ }, pc, elements);
    ggml_vk_sync_buffers(ctx, subctx);
}

ਸਥਿਰ ਖਾਲੀ ggml_vk_cpy_ਮੰਜ਼ਿਲ_strided(
        ggml_backend_vk_context * ctx, vk_context& subctx, vk_pipeline pipeline, ਅਚਲ ggml_tensor * tensor,
        ਅਚਲ vk_subbuffer & ਵਿੱਚ, ਅਚਲ vk_subbuffer & ਬਾਹਰਲਾ,
        uint32_t nb10, uint32_t nb11, uint32_t nb12, uint32_t nb13) {
    VK_LOG_DEBUG("ggml_vk_cpy_to_strided((" << tensor << ", type=" << tensor->type << ", ne0=" << tensor->ne[0] << ", ne1=" << tensor->ne[1] << ", ne2=" << tensor->ne[2] << ", ne3=" << tensor->ne[3] << ", nb0=" << tensor->nb[0] << ", nb1=" << tensor->nb[1] << ", nb2=" << tensor->nb[2] << ", nb3=" << tensor->nb[3] << "), ";
    std::cerr << "dst_nb=(" << nb10 << ", " << nb11 << ", " << nb12 << ", " << nb13 << "), buffer in size=" << ਵਿੱਚ.ਬਫਰ->size << ", buffer out size=" << ਬਾਹਰਲਾ.ਬਫਰ->size << ")");

    ਅਚਲ uint32_t ne = ggml_nelements(tensor);
    std::array<uint32_t, 3> elements;

    ਜੇ (ne > 262144) {
        elements = { 512, 512, CEIL_DIV(ne, 262144) };
    } ਨਹੀਂਤਾਂ ਜੇ (ne > 512) {
        elements = { 512, CEIL_DIV(ne, 512), 1 };
    } ਨਹੀਂਤਾਂ {
        elements = { ne, 1, 1 };
    }

    vk_op_unary_ਧੱਕੋ_constants pc = vk_op_unary_ਧੱਕੋ_constants_init(tensor, tensor, ne);
    pc.nb10 = nb10;
    pc.nb11 = nb11;
    pc.nb12 = nb12;
    pc.nb13 = nb13;
    init_pushconst_fastdiv(pc);
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { ਵਿੱਚ, ਬਾਹਰਲਾ }, pc, elements);
    ggml_vk_sync_buffers(ctx, subctx);
}

vk_pipeline ggml_vk_get_quantize_pipeline(ggml_backend_vk_context * ctx, ggml_type type) {
    ਚੋਣ(type) {
        ਹਾਲਤ GGML_TYPE_Q8_1:
            ਵਾਪਸ ctx->device->pipeline_quantize_q8_1_x4;
        ਮੂਲ:
            std::cerr << "Missing quantize pipeline for type: " << ggml_type_name(type) << std::ਸਤਰ_ਅੰਤ;
            GGML_ABORT("fatal error");
    }
}

ਖਾਲੀ ggml_vk_quantize_q8_1(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ vk_subbuffer & ਵਿੱਚ, ਅਚਲ vk_subbuffer & ਬਾਹਰਲਾ, uint32_t ne) {
    VK_LOG_DEBUG("ggml_vk_quantize_q8_1(" << "buffer in size=" << ਵਿੱਚ.ਬਫਰ->size << ", buffer out size=" << ਬਾਹਰਲਾ.ਬਫਰ->size << ", " << ne << ")");

    vk_pipeline pipeline = ggml_vk_get_quantize_pipeline(ctx, GGML_TYPE_Q8_1);

    ਅਚਲ uint32_t ਅੰਕ_ਖੰਡਾਂ = CEIL_DIV(ne, pipeline->wg_denoms[0]);
    // clamp the number of elements to the max workgroup count. The shader will iterate over the total number of blocks.
    ਅਚਲ uint64_t max_elements = std::min<uint64_t>(uint64_t{ctx->device->properties.limits.max_Compute_Work_ਸਮੂਹ_ਗਿਣਤੀ[0]} * pipeline->wg_denoms[0], std::numeric_limits<uint32_t>::max());
    ਅਚਲ uint32_t elements = std::min(ne, static_cast<uint32_t>(max_elements));

    ਅਚਲ vk_quantize_q8_1_ਧੱਕੋ_constants pc = {
        ne,
        ਅੰਕ_ਖੰਡਾਂ,
    };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { ਵਿੱਚ, ਬਾਹਰਲਾ }, pc, { elements, 1, 1 });
    ggml_vk_sync_buffers(ctx, subctx);
}

ਸਥਿਰ vk_pipeline ggml_vk_get_64b_indexing_pipeline(ggml_backend_vk_context * ctx, vk_pipeline &pipeline) {
    GGML_ਅਣਵਰਤਿਆ(ctx);
#ਜੇ defined(VK_EXT_shader_64bit_indexing)
    vk_pipeline *ptr = &pipeline;
    ਜਦੋਂ (*ptr) {
        ਜੇ ((*ptr)->is_64b_indexing) {
            ਵਾਪਸ *ptr;
        }
        ptr = &(*ptr)->ਅਗਲਾ;
    }
#ਜੇ_ਅੰਤ
    ਵਾਪਸ pipeline;
}

// perf-heuristic for choosing mul_mat b type, based on shader availability
ਸਥਿਰ ggml_type ggml_vk_mul_mat_b_type(
        ggml_backend_vk_context * ctx, ggml_type src0_type, ਅਚਲ ggml_tensor * src1,
        ggml_type f16_type, ggml_prec prec, ਤਾਰਕਿਕ mul_mat_id, ਤਾਰਕਿਕ can_quantize_y, ਤਾਰਕਿਕ prefer_f16_b) {
    ਸਵੈਚਾਲਿਤ ਉਪਲਬਧ = [&](ggml_type b) {
        ਵਾਪਸ ggml_vk_get_mul_mat_mat_pipeline_map(ctx, src0_type, b, prec, mul_mat_id) != ਖਾਲੀ_ਸੂਚਕ;
    };
    ਜੇ (can_quantize_y && ਉਪਲਬਧ(GGML_TYPE_Q8_1))             ਵਾਪਸ GGML_TYPE_Q8_1;
    ਜੇ (prefer_f16_b   && ਉਪਲਬਧ(f16_type))                   ਵਾਪਸ f16_type;
    ਜੇ (ggml_vk_dim01_contiguous(src1) && ਉਪਲਬਧ(src1->type)) ਵਾਪਸ src1->type;
    ਜੇ (src1->type == GGML_TYPE_BF16 && ਉਪਲਬਧ(GGML_TYPE_F32)) ਵਾਪਸ GGML_TYPE_F32;
    ਵਾਪਸ f16_type;
}

ਸਥਿਰ ਖਾਲੀ ggml_vk_mul_mat_q_f16(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ggml_tensor * dst, ਤਾਰਕਿਕ disable_ਵੰਡੋ_k) {
    VK_LOG_DEBUG("ggml_vk_mul_mat_q_f16((" << src0 << ", name=" << src0->name << ", type=" << ggml_type_name(src0->type) << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << ggml_type_name(src1->type) << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << ggml_type_name(dst->type) << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::cerr << "))");
    GGML_ASSERT(ggml_vk_dim01_contiguous(src0) || src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16);  // NOLINT
    GGML_ASSERT(ggml_vk_dim01_contiguous(src1) || src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);  // NOLINT

    ਅਚਲ uint64_t ne00 = src0->ne[0];
    ਅਚਲ uint64_t ne01 = src0->ne[1];
    ਅਚਲ uint64_t ne02 = src0->ne[2];
    ਅਚਲ uint64_t ne03 = src0->ne[3];

    ਅਚਲ uint64_t ne10 = src1->ne[0];
    ਅਚਲ uint64_t ne11 = src1->ne[1];
    ਅਚਲ uint64_t ne12 = src1->ne[2];
    ਅਚਲ uint64_t ne13 = src1->ne[3];

    ਅਚਲ uint64_t ne21 = dst->ne[1];
    ਅਚਲ uint32_t stride_d = dst->nb[1] / ggml_type_ਮਾਪ(dst->type);
    ਅਚਲ uint32_t stride_batch_d = stride_d*ne21;

    ਅਚਲ uint64_t r2 = ne12 / ne02;
    ਅਚਲ uint64_t r3 = ne13 / ne03;

    ggml_backend_vk_ਬਫਰ_context * dst_buf_ctx = (ggml_backend_vk_ਬਫਰ_context *)dst->ਬਫਰ->context;
    ggml_backend_vk_ਬਫਰ_context * src0_buf_ctx = (ggml_backend_vk_ਬਫਰ_context *)src0->ਬਫਰ->context;
    ggml_backend_vk_ਬਫਰ_context * src1_buf_ctx = (ggml_backend_vk_ਬਫਰ_context *)src1->ਬਫਰ->context;

    vk_ਬਫਰ d_Qx = ਖਾਲੀ_ਸੂਚਕ;
    size_t qx_buf_ਵਿੱਥ = 0;
    vk_ਬਫਰ d_Qy = ਖਾਲੀ_ਸੂਚਕ;
    size_t qy_buf_ਵਿੱਥ = 0;

    ਤਾਰਕਿਕ src0_uma = ਝੂਠ;
    ਤਾਰਕਿਕ src1_uma = ਝੂਠ;

    ਜੇ (ctx->device->uma) {
        ggml_vk_host_get(ctx->device, src0->ਅੰਕੜੇ, d_Qx, qx_buf_ਵਿੱਥ);
        ggml_vk_host_get(ctx->device, src1->ਅੰਕੜੇ, d_Qy, qy_buf_ਵਿੱਥ);
        src0_uma = d_Qx != ਖਾਲੀ_ਸੂਚਕ;
        src1_uma = d_Qy != ਖਾਲੀ_ਸੂਚਕ;
    }

    ਅਚਲ ਤਾਰਕਿਕ x_non_contig = (ctx->device->coopmat2 && src0->type == GGML_TYPE_F32) ||
                              !ggml_vk_dim01_contiguous(src0);
    // If src0 is BF16, try to use a BF16 x BF16 multiply
    ggml_type f16_type = src0->type == GGML_TYPE_BF16 ? GGML_TYPE_BF16 : GGML_TYPE_F16;

    ਅਚਲ ਤਾਰਕਿਕ is_coopmat = ctx->device->coopmat_support || ctx->device->coopmat2;
    ਅਚਲ ਤਾਰਕਿਕ can_quantize_y = (ctx->device->integer_dot_ਗੁਣਨਫਲ || ctx->device->coopmat_int_support) &&
                                src1->type == GGML_TYPE_F32 && ggml_is_contiguous(src1) && (ne11 * ne10) % 4 == 0;
    // coopmat shaders are fast enough that memory bandwidth reduction from f16 activations helps
    ਅਚਲ ਤਾਰਕਿਕ prefer_f16_b = is_coopmat && ggml_is_quantized(src0->type) && src1->type == GGML_TYPE_F32;

    ਅਚਲ ggml_type y_kernel_type = ggml_vk_mul_mat_b_type(ctx, src0->type, src1, f16_type,
                                        (ggml_prec)dst->op_params[0], ਝੂਠ, can_quantize_y, prefer_f16_b);

    ਅਚਲ ਤਾਰਕਿਕ quantize_y       = y_kernel_type == GGML_TYPE_Q8_1;
    ਅਚਲ ਤਾਰਕਿਕ y_f32_kernel     = y_kernel_type == GGML_TYPE_F32;
    ਅਚਲ ਤਾਰਕਿਕ y_needs_reformat = !quantize_y && (!ggml_vk_dim01_contiguous(src1) || y_kernel_type != src1->type);

    ਅਚਲ std::vector<vk_matmul_pipeline_pair>* mmp_map =
        ggml_vk_get_mul_mat_mat_pipeline_map(ctx, src0->type, y_kernel_type, (ggml_prec)dst->op_params[0]);

    ਅਚਲ ਤਾਰਕਿਕ qx_needs_dequant = mmp_map == ਖਾਲੀ_ਸੂਚਕ || x_non_contig;
    ਅਚਲ ਤਾਰਕਿਕ qy_needs_dequant = y_needs_reformat;

    ਜੇ (qx_needs_dequant) {
        // dequant src0 to f16
        mmp_map = ggml_vk_get_mul_mat_mat_pipeline_map(ctx, f16_type, y_kernel_type, (ggml_prec)dst->op_params[0]);
    }

    GGML_ASSERT(mmp_map != ਖਾਲੀ_ਸੂਚਕ);

    ਅਚਲ uint32_t kpad = quantize_y ? 0 : ggml_vk_align_ਮਾਪ(ne10, ggml_vk_guess_matmul_pipeline_align_map(ctx, *mmp_map, ne01, ne11, ਝੂਠ));
    ਅਚਲ ਤਾਰਕਿਕ aligned = !quantize_y && ne10 == kpad && ne01 > 8 && ne11 > 8;

    vk_pipeline pipeline = ggml_vk_guess_matmul_pipeline_map(ctx, *mmp_map, ne01, ne11, aligned, ਝੂਠ);

    ਜੇ (ggml_nbytes(src0) > ctx->device->properties.limits.max_Storage_ਬਫਰ_Range) {
        pipeline = ggml_vk_get_64b_indexing_pipeline(ctx, pipeline);
    }

    // Reserve extra storage in the N dimension for the Y matrix, so we can avoid bounds-checking
    uint32_t padded_n = qy_needs_dequant ? ROUNDUP_POW2(ne11, pipeline->wg_denoms[1]) : ne11;
    ਅਚਲ uint64_t x_ne = ggml_nelements(src0);
    // 128 elements per Q8_1 x4 block
    ਅਚਲ uint64_t y_ne = padded_n * ne10 * ne12 * ne13;
    ਅਚਲ uint64_t d_ne = ggml_nelements(dst);

    ਅਚਲ uint32_t ਵੰਡੋ_k = ggml_vk_guess_ਵੰਡੋ_k(ctx, ne01, ne11, ne10, disable_ਵੰਡੋ_k, pipeline);

    ਅਚਲ uint64_t qx_sz = ggml_type_ਮਾਪ(src0->type) * x_ne / ggml_blck_ਮਾਪ(src0->type);
    ਅਚਲ uint64_t qy_sz = ggml_type_ਮਾਪ(src1->type) * y_ne / ggml_blck_ਮਾਪ(src1->type);
    ਅਚਲ uint64_t x_sz = !qx_needs_dequant ? qx_sz : ਆਕਾਰ(ggml_fp16_t) * x_ne;
    ਅਚਲ uint64_t y_sz = quantize_y ? (ggml_vk_align_ਮਾਪ(y_ne, 128) * ggml_type_ਮਾਪ(GGML_TYPE_Q8_1) / ggml_blck_ਮਾਪ(GGML_TYPE_Q8_1)) : (y_f32_kernel ? ਆਕਾਰ(ਦਸ਼ਮਲਵ) * y_ne : ਆਕਾਰ(ggml_fp16_t) * y_ne);
    ਅਚਲ uint64_t d_sz = ਆਕਾਰ(ਦਸ਼ਮਲਵ) * d_ne;

    vk_pipeline ਮੰਜ਼ਿਲ_fp16_vk_0 = ਖਾਲੀ_ਸੂਚਕ;
    vk_pipeline ਮੰਜ਼ਿਲ_fp16_vk_1 = ਖਾਲੀ_ਸੂਚਕ;
    vk_pipeline ਮੰਜ਼ਿਲ_q8_1 = ਖਾਲੀ_ਸੂਚਕ;

    ਜੇ (x_non_contig) {
        ਮੰਜ਼ਿਲ_fp16_vk_0 = ggml_vk_get_cpy_pipeline(ctx, src0, ਖਾਲੀ_ਸੂਚਕ, f16_type);
    } ਨਹੀਂਤਾਂ {
        ਮੰਜ਼ਿਲ_fp16_vk_0 = ggml_vk_get_ਮੰਜ਼ਿਲ_fp16(ctx, src0->type);
    }
    ਜੇ (y_needs_reformat) {
        ਮੰਜ਼ਿਲ_fp16_vk_1 = ggml_vk_get_cpy_pipeline(ctx, src1, ਖਾਲੀ_ਸੂਚਕ, y_kernel_type);
    } ਨਹੀਂਤਾਂ {
        ਮੰਜ਼ਿਲ_fp16_vk_1 = ggml_vk_get_ਮੰਜ਼ਿਲ_fp16(ctx, src1->type);
    }
    GGML_ASSERT(!qx_needs_dequant || ਮੰਜ਼ਿਲ_fp16_vk_0 != ਖਾਲੀ_ਸੂਚਕ);  // NOLINT
    GGML_ASSERT(!qy_needs_dequant || ਮੰਜ਼ਿਲ_fp16_vk_1 != ਖਾਲੀ_ਸੂਚਕ);  // NOLINT

    ਜੇ (quantize_y) {
        ਮੰਜ਼ਿਲ_q8_1 = ggml_vk_get_quantize_pipeline(ctx, GGML_TYPE_Q8_1);
    }

    {
        ਅਚਲ uint64_t ਵੰਡੋ_k_ਮਾਪ = ਵੰਡੋ_k > 1 ? d_sz * ਵੰਡੋ_k : 0;
        ਜੇ (
                (qx_needs_dequant && x_sz > ctx->device->properties.limits.max_Storage_ਬਫਰ_Range) ||
                (qy_needs_dequant && y_sz > ctx->device->properties.limits.max_Storage_ਬਫਰ_Range) ||
                (ਵੰਡੋ_k > 1 && ਵੰਡੋ_k_ਮਾਪ > ctx->device->properties.limits.max_Storage_ਬਫਰ_Range)) {
            GGML_ABORT("Requested preallocation size is too large");
        }
        ਜੇ (qx_needs_dequant && ctx->prealloc_ਮਾਪ_x < x_sz) {
            ctx->prealloc_ਮਾਪ_x = x_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        ਜੇ ((qy_needs_dequant || quantize_y) && ctx->prealloc_ਮਾਪ_y < y_sz) {
            ctx->prealloc_ਮਾਪ_y = y_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        ਜੇ (ਵੰਡੋ_k > 1 && ctx->prealloc_ਮਾਪ_ਵੰਡੋ_k < ਵੰਡੋ_k_ਮਾਪ) {
            ctx->prealloc_ਮਾਪ_ਵੰਡੋ_k = ਵੰਡੋ_k_ਮਾਪ;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }

        // Request descriptor sets
        ਜੇ (qx_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, ਮੰਜ਼ਿਲ_fp16_vk_0, 1);
        }
        ਜੇ (qy_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, ਮੰਜ਼ਿਲ_fp16_vk_1, 1);
        }
        ਜੇ (quantize_y) {
            ggml_pipeline_request_descriptor_sets(ctx, ਮੰਜ਼ਿਲ_q8_1, 1);
        }
        ਜੇ (ਵੰਡੋ_k > 1) {
            ggml_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_matmul_ਵੰਡੋ_k_reduce, 1);
        }
    }

    vk_ਬਫਰ d_D = dst_buf_ctx->dev_ਬਫਰ;
    ਅਚਲ uint64_t d_buf_ਵਿੱਥ = vk_tensor_ਵਿੱਥ(dst) + dst->view_offs;
    GGML_ASSERT(d_D != ਖਾਲੀ_ਸੂਚਕ);
    GGML_ASSERT(d_D->size >= d_buf_ਵਿੱਥ + d_sz);
    vk_ਬਫਰ d_X;
    uint64_t x_buf_ਵਿੱਥ = 0;
    vk_ਬਫਰ d_Y;
    uint64_t y_buf_ਵਿੱਥ = 0;
    ਜੇ (!src0_uma) {
        d_Qx = src0_buf_ctx->dev_ਬਫਰ;
        qx_buf_ਵਿੱਥ = vk_tensor_ਵਿੱਥ(src0) + src0->view_offs;
        GGML_ASSERT(d_Qx != ਖਾਲੀ_ਸੂਚਕ);
    }
    ਜੇ (!src1_uma) {
        d_Qy = src1_buf_ctx->dev_ਬਫਰ;
        qy_buf_ਵਿੱਥ = vk_tensor_ਵਿੱਥ(src1) + src1->view_offs;
        GGML_ASSERT(d_Qy != ਖਾਲੀ_ਸੂਚਕ);
    }
    ਜੇ (qx_needs_dequant) {
        d_X = ctx->prealloc_x;
        GGML_ASSERT(d_X->size >= x_sz);
    } ਨਹੀਂਤਾਂ {
        d_X = d_Qx;
        x_buf_ਵਿੱਥ = qx_buf_ਵਿੱਥ;
        GGML_ASSERT(qx_sz == x_sz);
    }
    ਜੇ (qy_needs_dequant) {
        d_Y = ctx->prealloc_y;
        GGML_ASSERT(d_Y->size >= y_sz);
    } ਨਹੀਂਤਾਂ ਜੇ (quantize_y) {
        d_Y = ctx->prealloc_y;
        GGML_ASSERT(d_Y->size >= CEIL_DIV(y_sz, 144) * 144);
    } ਨਹੀਂਤਾਂ {
        d_Y = d_Qy;
        y_buf_ਵਿੱਥ = qy_buf_ਵਿੱਥ;
        GGML_ASSERT(qy_sz == y_sz);
    }

    ਜੇ (x_non_contig || qx_needs_dequant) {
        ਜੇ (ctx->prealloc_x_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }
    }

    ਜੇ (x_non_contig) {
        ggml_vk_cpy_ਮੰਜ਼ਿਲ_contiguous(ctx, subctx, ਮੰਜ਼ਿਲ_fp16_vk_0, src0, ggml_vk_subbuffer(ctx, d_Qx, qx_buf_ਵਿੱਥ), ggml_vk_subbuffer(ctx, d_X, 0));
    } ਨਹੀਂਤਾਂ ਜੇ (qx_needs_dequant) {
        ਅਚਲ std::vector<uint32_t> pc = { (uint32_t)ne01, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)(ggml_nelements(src0)) };
        ggml_vk_dispatch_pipeline(ctx, subctx, ਮੰਜ਼ਿਲ_fp16_vk_0, { vk_subbuffer{ d_Qx, qx_buf_ਵਿੱਥ, qx_sz }, vk_subbuffer{ d_X, 0, x_sz } }, pc, { (uint32_t)(x_ne), 1, 1});
        ggml_vk_sync_buffers(ctx, subctx);
    }
    ਜੇ (y_needs_reformat) {
        ਜੇ (ctx->prealloc_y_ਆਖਰੀ_pipeline_ਵਰਤਿਆ != ਮੰਜ਼ਿਲ_fp16_vk_1.get() ||
            ctx->prealloc_y_ਆਖਰੀ_tensor_ਵਰਤਿਆ != src1 ||
            ctx->prealloc_y_ਆਖਰੀ_k_padded) {
            ਜੇ (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ggml_vk_cpy_ਮੰਜ਼ਿਲ_contiguous(ctx, subctx, ਮੰਜ਼ਿਲ_fp16_vk_1, src1, ggml_vk_subbuffer(ctx, d_Qy, qy_buf_ਵਿੱਥ), ggml_vk_subbuffer(ctx, d_Y, 0));
            ctx->prealloc_y_ਆਖਰੀ_pipeline_ਵਰਤਿਆ = ਮੰਜ਼ਿਲ_fp16_vk_1.get();
            ctx->prealloc_y_ਆਖਰੀ_tensor_ਵਰਤਿਆ = src1;
            ctx->prealloc_y_ਆਖਰੀ_k_padded = ਝੂਠ;
        }
    }
    ਜੇ (quantize_y) {
        ਜੇ (ctx->prealloc_y_ਆਖਰੀ_pipeline_ਵਰਤਿਆ != ਮੰਜ਼ਿਲ_q8_1.get() ||
            ctx->prealloc_y_ਆਖਰੀ_tensor_ਵਰਤਿਆ != src1 ||
            ctx->prealloc_y_ਆਖਰੀ_k_padded) {
            ਜੇ (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ggml_vk_quantize_q8_1(ctx, subctx, ggml_vk_subbuffer(ctx, d_Qy, qy_buf_ਵਿੱਥ), ggml_vk_subbuffer(ctx, d_Y, 0), y_ne);
            ctx->prealloc_y_ਆਖਰੀ_pipeline_ਵਰਤਿਆ = ਮੰਜ਼ਿਲ_q8_1.get();
            ctx->prealloc_y_ਆਖਰੀ_tensor_ਵਰਤਿਆ = src1;
            ctx->prealloc_y_ਆਖਰੀ_k_padded = ਝੂਠ;
        }
    }

    // The partial tiles of a quantized A rely on the bound range to read zeros past the last row,
    // so the range stays exact: the strided extent of a tensor read in place, the staged size otherwise.
    ਅਚਲ uint64_t x_range = qx_needs_dequant ? x_sz : ggml_nbytes(src0);
    ਅਚਲ uint64_t y_range = (qy_needs_dequant || quantize_y) ? y_sz : ggml_nbytes(src1);

    uint32_t stride_batch_x = qx_needs_dequant ? ne00*ne01 : ggml_vk_batch_stride(src0);
    uint32_t stride_batch_y = (qy_needs_dequant || quantize_y) ? ne10*ne11 : ggml_vk_batch_stride(src1);

    ਜੇ (!ggml_vk_dim01_contiguous(src0) && !qx_needs_dequant) {
        stride_batch_x = src0->nb[0] / ggml_type_ਮਾਪ(src0->type);
    }

    ਜੇ (!ggml_vk_dim01_contiguous(src1) && !qy_needs_dequant && !quantize_y) {
        stride_batch_y = src1->nb[0] / ggml_type_ਮਾਪ(src1->type);
    }

    // column stride in bytes of the B buffer: raw src1 keeps its own stride, staging buffers are tightly packed
    ਅਚਲ uint64_t b_ਖਾਨਾ_ਬਾਈਟਾਂ = (y_needs_reformat || qy_needs_dequant || quantize_y)
        ? (quantize_y ? (uint64_t)(ne10 / ggml_blck_ਮਾਪ(GGML_TYPE_Q8_1)) * ggml_type_ਮਾਪ(GGML_TYPE_Q8_1) : (uint64_t)ne10 * ਆਕਾਰ(ggml_fp16_t))
        : (uint64_t)src1->nb[1];

    // compute
    ggml_vk_matmul(
        ctx, subctx, pipeline,
        { d_X, x_buf_ਵਿੱਥ, x_range }, { d_Y, y_buf_ਵਿੱਥ, y_range },
        ggml_vk_subbuffer(ctx, d_D, d_buf_ਵਿੱਥ), { ctx->prealloc_ਵੰਡੋ_k, 0, d_sz * ਵੰਡੋ_k },
        ne01, ne11, ne10,
        ne10, ne10, stride_d, stride_batch_x, stride_batch_y, stride_batch_d,
        ਵੰਡੋ_k, ne12*ne13, ne02, ne12, r2, r3, padded_n,
        b_ਖਾਨਾ_ਬਾਈਟਾਂ, (uint64_t)dst->nb[1]
    );  // NOLINT

    ਜੇ (x_non_contig || qx_needs_dequant) {
        ctx->prealloc_x_need_sync = ਸੱਚ;
    }
    ਜੇ (y_needs_reformat || quantize_y) {
        ctx->prealloc_y_need_sync = ਸੱਚ;
    }
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_vk_should_use_mmvq(ਅਚਲ vk_device& device, uint32_t m, uint32_t n, uint32_t k, ggml_type src0_type) {
    ਜੇ (device->mmvq_ਬਹੁਲਕ == 1) {
        ਵਾਪਸ ਸੱਚ;
    } ਨਹੀਂਤਾਂ ਜੇ (device->mmvq_ਬਹੁਲਕ == -1) {
        ਵਾਪਸ ਝੂਠ;
    }

    // q6_k only has 2-byte alignment which makes it somewhat problematic,
    // using MMVQ is only a win on Intel.
    ਤਾਰਕਿਕ mmvq_q6 = device->vendor_id == VK_VENDOR_ID_INTEL;
    ਜੇ (src0_type == GGML_TYPE_Q6_K && !mmvq_q6) {
        ਵਾਪਸ ਝੂਠ;
    }

    // MMVQ is generally good for batches
    ਜੇ (n > 1) {
        ਵਾਪਸ ਸੱਚ;
    }

    // Quantization overhead is not worth it for small k
    ਚੋਣ (device->vendor_id) {
    ਹਾਲਤ VK_VENDOR_ID_NVIDIA:
        ਜੇ (src0_type == GGML_TYPE_Q2_0 || src0_type == GGML_TYPE_Q2_K || src0_type == GGML_TYPE_Q3_K || src0_type == GGML_TYPE_IQ1_S || src0_type == GGML_TYPE_IQ1_M) {
            ਵਾਪਸ ਸੱਚ;
        }

        ਜੇ (k <= 4096) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਚੋਣ (src0_type) {
        ਹਾਲਤ GGML_TYPE_MXFP4:
        ਹਾਲਤ GGML_TYPE_Q8_0:
            ਵਾਪਸ device->architecture == vk_device_architecture::NVIDIA_PRE_TURING;
        ਮੂਲ:
            ਵਾਪਸ ਸੱਚ;
        }
    ਹਾਲਤ VK_VENDOR_ID_AMD:
        ਜੇ (k < 2048) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਚੋਣ (src0_type) {
        ਹਾਲਤ GGML_TYPE_Q8_0:
            ਵਾਪਸ device->architecture == vk_device_architecture::AMD_GCN;
        ਮੂਲ:
            ਵਾਪਸ ਸੱਚ;
        }
    ਹਾਲਤ VK_VENDOR_ID_INTEL:
        ਜੇ (device->architecture == vk_device_architecture::INTEL_XE2) {
            ਜੇ (src0_type == GGML_TYPE_Q2_0 || src0_type == GGML_TYPE_Q2_K || src0_type == GGML_TYPE_Q3_K || src0_type == GGML_TYPE_Q6_K) {
                ਵਾਪਸ ਸੱਚ;
            }
        }

        ਜੇ (device->driver_id == vk::DriverId::eIntelProprietaryWindows) {
            // Intel Windows proprietary driver MMVQ performance for !Q2/Q3/Q6 is worse than fp16,
            // see https://github.com/ggml-org/llama.cpp/issues/17628 and
            // https://github.com/ggml-org/llama.cpp/pull/23056
            ਵਾਪਸ ਝੂਠ;
        }

        ਜੇ (k < 2048) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਚੋਣ (src0_type) {
        // From tests on A770 Linux, may need more tuning
        ਹਾਲਤ GGML_TYPE_Q4_0:
        ਹਾਲਤ GGML_TYPE_Q5_1:
        ਹਾਲਤ GGML_TYPE_IQ4_XS:
            ਵਾਪਸ ਝੂਠ;
        ਮੂਲ:
            ਵਾਪਸ ਸੱਚ;
        }
    ਹਾਲਤ VK_VENDOR_ID_QUALCOMM:
        ਵਾਪਸ ਝੂਠ;
    ਮੂਲ:
        ਵਾਪਸ ਸੱਚ;
    }

    GGML_ਅਣਵਰਤਿਆ(m);
}

ਸਥਿਰ ਖਾਲੀ ggml_vk_mul_mat_vec_q_f16(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ਬਣਤਰ ggml_cgraph * cgraph, ਪੂਰਨ ਗੰਢ_idx, ਤਾਰਕਿਕ ਅਦਲਾ_ਬਦਲੀ_inputs = ਝੂਠ) {
    ggml_tensor * dst = cgraph->nodes[ਗੰਢ_idx];
    ਅਚਲ ggml_tensor * src0 = dst->src[ਅਦਲਾ_ਬਦਲੀ_inputs ? 1 : 0];
    ਅਚਲ ggml_tensor * src1 = dst->src[ਅਦਲਾ_ਬਦਲੀ_inputs ? 0 : 1];

    VK_LOG_DEBUG("ggml_vk_mul_mat_vec_q_f16((" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::cerr << ")),)");
    GGML_ASSERT(ggml_vk_dim01_contiguous(src0) || src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16);  // NOLINT
    GGML_ASSERT(ggml_vk_dim01_contiguous(src1) || src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);  // NOLINT

    ਅਚਲ uint64_t ne00 = src0->ne[0];
    ਅਚਲ uint64_t ne01 = src0->ne[1];
    ਅਚਲ uint64_t ne02 = src0->ne[2];
    ਅਚਲ uint64_t ne03 = src0->ne[3];

    ਅਚਲ uint64_t ne10 = src1->ne[0];
    ਅਚਲ uint64_t ne11 = src1->ne[1];
    ਅਚਲ uint64_t ne12 = src1->ne[2];
    ਅਚਲ uint64_t ne13 = src1->ne[3];

    ਅਚਲ uint64_t ne20 = dst->ne[ਅਦਲਾ_ਬਦਲੀ_inputs ? 1 : 0];
    ਅਚਲ uint64_t ne21 = dst->ne[ਅਦਲਾ_ਬਦਲੀ_inputs ? 0 : 1];
    // const uint64_t ne22 = dst->ne[2];
    // const uint64_t ne23 = dst->ne[3];

    ਅਚਲ uint64_t r2 = ne12 / ne02;
    ਅਚਲ uint64_t r3 = ne13 / ne03;

    // batch_n indicates that we need to compute a few vector results, and this assumes
    // ne12 and ne13 are 1. It overloads the batch_strides to hold the row strides.
    GGML_ASSERT(ne11 == 1 || ne12 * ne13 == 1);
    ਤਾਰਕਿਕ batch_n = ne11 > 1;

    ਅਚਲ ਤਾਰਕਿਕ x_non_contig = !ggml_vk_dim01_contiguous(src0);

    // BF16 src1 has no mul_mat_vec B-type; widen to F32
    ਅਚਲ ਤਾਰਕਿਕ widen_bf16 = src1->type == GGML_TYPE_BF16;
    ਅਚਲ ਤਾਰਕਿਕ y_non_contig = !ggml_vk_dim01_contiguous(src1) || widen_bf16;

    ਅਚਲ ggml_type y_kernel_type = widen_bf16 ? GGML_TYPE_F32 : src1->type;
    ਅਚਲ ਤਾਰਕਿਕ f16_f32_kernel = y_kernel_type == GGML_TYPE_F32;
    ਤਾਰਕਿਕ quantize_y = ctx->device->integer_dot_ਗੁਣਨਫਲ && src1->type == GGML_TYPE_F32 && ggml_is_contiguous(src1) && !y_non_contig && (ne11 * ne10) % 4 == 0 && ggml_vk_should_use_mmvq(ctx->device, ne01, ne11, ne10, src0->type);

    vk_pipeline ਮੰਜ਼ਿਲ_fp16_vk_0 = ਖਾਲੀ_ਸੂਚਕ;
    vk_pipeline ਮੰਜ਼ਿਲ_fp16_vk_1 = ਖਾਲੀ_ਸੂਚਕ;
    ਜੇ (x_non_contig) {
        ਮੰਜ਼ਿਲ_fp16_vk_0 = ggml_vk_get_cpy_pipeline(ctx, src0, ਖਾਲੀ_ਸੂਚਕ, src0->type);
    }
    ਜੇ (y_non_contig) {
        ਮੰਜ਼ਿਲ_fp16_vk_1 = ggml_vk_get_cpy_pipeline(ctx, src1, ਖਾਲੀ_ਸੂਚਕ, y_kernel_type);
    } ਨਹੀਂਤਾਂ {
        ਮੰਜ਼ਿਲ_fp16_vk_1 = ggml_vk_get_ਮੰਜ਼ਿਲ_fp16(ctx, src1->type);
    }

    // Check for mmq first
    vk_pipeline dmmv = quantize_y ? ggml_vk_get_dequantize_mul_mat_vec(ctx, src0->type, GGML_TYPE_Q8_1, ne11, ne20, ne00) : ਖਾਲੀ_ਸੂਚਕ;
    vk_pipeline ਮੰਜ਼ਿਲ_q8_1 = ਖਾਲੀ_ਸੂਚਕ;

    ਜੇ (dmmv == ਖਾਲੀ_ਸੂਚਕ) {
        // Fall back to f16 dequant mul mat
        dmmv = ggml_vk_get_dequantize_mul_mat_vec(ctx, src0->type, y_kernel_type, ne11, ne20, ne00);
        quantize_y = ਝੂਠ;
    }

    ਜੇ (quantize_y) {
        ਮੰਜ਼ਿਲ_q8_1 = ggml_vk_get_quantize_pipeline(ctx, GGML_TYPE_Q8_1);
    }

    ਜੇ (ggml_nbytes(src0) > ctx->device->properties.limits.max_Storage_ਬਫਰ_Range) {
        dmmv = ggml_vk_get_64b_indexing_pipeline(ctx, dmmv);
    }

    ਅਚਲ ਤਾਰਕਿਕ qx_needs_dequant = x_non_contig;
    ਅਚਲ ਤਾਰਕਿਕ qy_needs_dequant = !quantize_y && ((src1->type != GGML_TYPE_F16 && !f16_f32_kernel) || y_non_contig);

    // Not implemented
    GGML_ASSERT(y_non_contig || !qy_needs_dequant);  // NOLINT

    GGML_ASSERT(!qx_needs_dequant || ਮੰਜ਼ਿਲ_fp16_vk_0 != ਖਾਲੀ_ਸੂਚਕ);  // NOLINT
    GGML_ASSERT(!qy_needs_dequant || ਮੰਜ਼ਿਲ_fp16_vk_1 != ਖਾਲੀ_ਸੂਚਕ);  // NOLINT
    GGML_ASSERT(dmmv != ਖਾਲੀ_ਸੂਚਕ);

    ਅਚਲ uint64_t x_ne = ggml_nelements(src0);
    ਅਚਲ uint64_t y_ne = ggml_nelements(src1);

    ਅਚਲ uint64_t qx_sz = ggml_vk_align_ਮਾਪ(ggml_type_ਮਾਪ(src0->type) * x_ne / ggml_blck_ਮਾਪ(src0->type), ctx->device->properties.limits.min_Storage_ਬਫਰ_ਵਿੱਥ_Alignment);
    ਅਚਲ uint64_t x_sz = x_non_contig ? ggml_vk_align_ਮਾਪ(ggml_type_ਮਾਪ(src0->type) * x_ne, ctx->device->properties.limits.min_Storage_ਬਫਰ_ਵਿੱਥ_Alignment) : qx_sz;
    ਅਚਲ uint64_t y_sz = quantize_y ? (ggml_vk_align_ਮਾਪ(y_ne, 128) * ggml_type_ਮਾਪ(GGML_TYPE_Q8_1) / ggml_blck_ਮਾਪ(GGML_TYPE_Q8_1)) :
                         (f16_f32_kernel ? ਆਕਾਰ(ਦਸ਼ਮਲਵ) * y_ne : ਆਕਾਰ(ggml_fp16_t) * y_ne);

    {
        ਜੇ (
                (qx_needs_dequant && x_sz > ctx->device->properties.limits.max_Storage_ਬਫਰ_Range) ||
                (qy_needs_dequant && y_sz > ctx->device->properties.limits.max_Storage_ਬਫਰ_Range)) {
            GGML_ABORT("Requested preallocation size is too large");
        }
        ਜੇ (qx_needs_dequant && ctx->prealloc_ਮਾਪ_x < x_sz) {
            ctx->prealloc_ਮਾਪ_x = x_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        ਜੇ ((qy_needs_dequant || quantize_y) && ctx->prealloc_ਮਾਪ_y < y_sz) {
            ctx->prealloc_ਮਾਪ_y = y_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }

        // Request descriptor sets
        ਜੇ (qx_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, ਮੰਜ਼ਿਲ_fp16_vk_0, 1);
        }
        ਜੇ (qy_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, ਮੰਜ਼ਿਲ_fp16_vk_1, 1);
        }
        ਜੇ (quantize_y) {
            ggml_pipeline_request_descriptor_sets(ctx, ਮੰਜ਼ਿਲ_q8_1, 1);
        }
    }

    vk_subbuffer d_D = ggml_vk_tensor_subbuffer(ctx, cgraph->nodes[ਗੰਢ_idx + ctx->ਅੰਕ_additional_fused_ops]);
    vk_subbuffer d_Qx = ggml_vk_tensor_subbuffer(ctx, src0);
    vk_subbuffer d_Qy = ggml_vk_tensor_subbuffer(ctx, src1);
    vk_subbuffer d_X, d_Y;

    ਜੇ (qx_needs_dequant) {
        d_X = { ctx->prealloc_x, 0, ctx->prealloc_x->size };
    } ਨਹੀਂਤਾਂ {
        d_X = d_Qx;
        GGML_ASSERT(qx_sz == x_sz);
    }
    ਜੇ (qy_needs_dequant || quantize_y) {
        d_Y = { ctx->prealloc_y, 0, ctx->prealloc_y->size };
    } ਨਹੀਂਤਾਂ {
        d_Y = d_Qy;
    }

    ਜੇ (x_non_contig) {
        ਜੇ (ctx->prealloc_x_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }

        GGML_ASSERT(x_sz == ggml_vk_align_ਮਾਪ(ggml_type_ਮਾਪ(src0->type) * x_ne, ctx->device->properties.limits.min_Storage_ਬਫਰ_ਵਿੱਥ_Alignment));
        ggml_vk_cpy_ਮੰਜ਼ਿਲ_contiguous(ctx, subctx, ਮੰਜ਼ਿਲ_fp16_vk_0, src0, d_Qx, d_X);
    }
    ਜੇ (y_non_contig) {
        GGML_ASSERT(y_sz == ggml_type_ਮਾਪ(y_kernel_type) * y_ne);
        ਜੇ (ctx->prealloc_y_ਆਖਰੀ_pipeline_ਵਰਤਿਆ != ਮੰਜ਼ਿਲ_fp16_vk_1.get() ||
            ctx->prealloc_y_ਆਖਰੀ_tensor_ਵਰਤਿਆ != src1 ||
            ctx->prealloc_y_ਆਖਰੀ_k_padded) {
            ਜੇ (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ggml_vk_cpy_ਮੰਜ਼ਿਲ_contiguous(ctx, subctx, ਮੰਜ਼ਿਲ_fp16_vk_1, src1, d_Qy, d_Y);
            ctx->prealloc_y_ਆਖਰੀ_pipeline_ਵਰਤਿਆ = ਮੰਜ਼ਿਲ_fp16_vk_1.get();
            ctx->prealloc_y_ਆਖਰੀ_tensor_ਵਰਤਿਆ = src1;
            ctx->prealloc_y_ਆਖਰੀ_k_padded = ਝੂਠ;
        }
    }
    ਜੇ (quantize_y) {
        ਜੇ (ctx->prealloc_y_ਆਖਰੀ_pipeline_ਵਰਤਿਆ != ਮੰਜ਼ਿਲ_q8_1.get() ||
            ctx->prealloc_y_ਆਖਰੀ_tensor_ਵਰਤਿਆ != src1 ||
            ctx->prealloc_y_ਆਖਰੀ_k_padded) {
            ਜੇ (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ggml_vk_quantize_q8_1(ctx, subctx, d_Qy, d_Y, y_ne);
            ctx->prealloc_y_ਆਖਰੀ_pipeline_ਵਰਤਿਆ = ਮੰਜ਼ਿਲ_q8_1.get();
            ctx->prealloc_y_ਆਖਰੀ_tensor_ਵਰਤਿਆ = src1;
            ctx->prealloc_y_ਆਖਰੀ_k_padded = ਝੂਠ;
        }
    }

    // For batch_n, the A matrix is the same for each batch, and B/D use the row stride as the batch stride
    uint32_t stride_batch_x = batch_n ? 0 : (qx_needs_dequant ? ne00*ne01 : ggml_vk_batch_stride(src0));
    uint32_t stride_batch_y = batch_n ? ne10 : ((qy_needs_dequant || quantize_y) ? ne10*ne11 : ggml_vk_batch_stride(src1));
    uint32_t stride_batch_d = batch_n ? ne20 : (ne20*ne21);

    ਜੇ (!ggml_vk_dim01_contiguous(src0) && !qx_needs_dequant) {
        stride_batch_x = src0->nb[0] / ggml_type_ਮਾਪ(src0->type);
    }

    ਜੇ (!ggml_vk_dim01_contiguous(src1) && !qy_needs_dequant) {
        stride_batch_y = src1->nb[0] / ggml_type_ਮਾਪ(src1->type);
    }

    ਅਚਲ uint32_t max_groups_x = ctx->device->properties.limits.max_Compute_Work_ਸਮੂਹ_ਗਿਣਤੀ[0];

    uint32_t groups_x = ne01;
    uint32_t groups_z = 1;

    ਜੇ (ne01 > max_groups_x) {
        groups_z = 64;
        groups_x = CEIL_DIV(groups_x, groups_z);
    }

    uint32_t fusion_flags = 0;

    vk_subbuffer d_F0 = d_D;
    ਜੇ (ctx->ਅੰਕ_additional_fused_ops > 0) {
        ਅਚਲ ggml_tensor * ਜੋੜੋ_2 = cgraph->nodes[ਗੰਢ_idx + 1];
        ਅਚਲ ggml_tensor * bias = ਜੋੜੋ_2->src[0] == dst ? ਜੋੜੋ_2->src[1] : ਜੋੜੋ_2->src[0];

        d_F0 = ggml_vk_tensor_subbuffer(ctx, bias);
        fusion_flags |= MAT_VEC_FUSION_FLAGS_BIAS0;
    }

    vk_subbuffer d_F1 = d_D;
    ਜੇ (ctx->ਅੰਕ_additional_fused_ops == 2) {
        ਅਚਲ ggml_tensor * ਜੋੜੋ_2 = cgraph->nodes[ਗੰਢ_idx + 2];
        ਅਚਲ ggml_tensor * bias = ਜੋੜੋ_2->src[0] == cgraph->nodes[ਗੰਢ_idx + 1] ? ਜੋੜੋ_2->src[1] : ਜੋੜੋ_2->src[0];

        d_F1 = ggml_vk_tensor_subbuffer(ctx, bias);
        fusion_flags |= MAT_VEC_FUSION_FLAGS_BIAS1;
    }

    ggml_pipeline_request_descriptor_sets(ctx, dmmv, CEIL_DIV(ne12 * ne13, ctx->device->properties.limits.max_Compute_Work_ਸਮੂਹ_ਗਿਣਤੀ[1]));

    uint32_t ਅਧਾਰ_work_ਸਮੂਹ_y = 0;
    ਜਦੋਂ (ਅਧਾਰ_work_ਸਮੂਹ_y < ne12 * ne13) {

        uint32_t groups_y = std::min((uint32_t)(ne12 * ne13) - ਅਧਾਰ_work_ਸਮੂਹ_y, ctx->device->properties.limits.max_Compute_Work_ਸਮੂਹ_ਗਿਣਤੀ[1]);
        ਅਚਲ vk_mat_vec_ਧੱਕੋ_constants pc = {
            (uint32_t)ne00, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)ne01,
            stride_batch_x, stride_batch_y, stride_batch_d,
            fusion_flags, ਅਧਾਰ_work_ਸਮੂਹ_y,
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
        ਅਧਾਰ_work_ਸਮੂਹ_y += groups_y;
    }

    ਜੇ (x_non_contig) {
        ctx->prealloc_x_need_sync = ਸੱਚ;
    }
    ਜੇ (y_non_contig || quantize_y) {
        ctx->prealloc_y_need_sync = ਸੱਚ;
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_vk_mul_mat_vec_p021_f16_f32(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ਬਣਤਰ ggml_cgraph * cgraph, ਪੂਰਨ ਗੰਢ_idx) {
    ggml_tensor * dst = cgraph->nodes[ਗੰਢ_idx];
    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];
    VK_LOG_DEBUG("ggml_vk_mul_mat_p021_f16_f32(" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::cerr << "))");
    GGML_ASSERT(ggml_is_permuted(src0) && ggml_is_permuted(src1));
    GGML_ASSERT(src0->nb[0] <= src0->nb[1] && src0->nb[2] <= src0->nb[3]);  // NOLINT
    GGML_ASSERT(src1->nb[0] <= src1->nb[1] && src1->nb[2] <= src1->nb[3]);  // NOLINT
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    ਅਚਲ uint64_t ne00 = src0->ne[0];
    ਅਚਲ uint64_t ne01 = src0->ne[1];
    ਅਚਲ uint64_t ne02 = src0->ne[2];
    // const uint64_t ne03 = src0->ne[3];

    //const uint64_t ne10 = src1->ne[0];
    ਅਚਲ uint64_t ne11 = src1->ne[1];
    ਅਚਲ uint64_t ne12 = src1->ne[2];
    // const uint64_t ne13 = src1->ne[3];

    GGML_ASSERT(ne11 == 1);

    // With grouped query attention there are > 1 Q matrices per K, V matrix.
    uint32_t gqa_ਅਨੁਪਾਤ = (uint32_t)ne12 / (uint32_t)ne02;
    ਜੇ (gqa_ਅਨੁਪਾਤ > 8 || gqa_ਅਨੁਪਾਤ == 0 || ne12 != ne02 * gqa_ਅਨੁਪਾਤ) {
        gqa_ਅਨੁਪਾਤ = 1;
    }

    vk_pipeline pipeline = ctx->device->pipeline_mul_mat_vec_p021_f16_f32[gqa_ਅਨੁਪਾਤ - 1];

    ਜੇ (ggml_nbytes(src0) > ctx->device->properties.limits.max_Storage_ਬਫਰ_Range) {
        pipeline = ggml_vk_get_64b_indexing_pipeline(ctx, pipeline);
    }

    {
        // Request descriptor sets
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    }

    vk_subbuffer d_D = ggml_vk_tensor_subbuffer(ctx, cgraph->nodes[ਗੰਢ_idx + ctx->ਅੰਕ_additional_fused_ops], ਸੱਚ);
    vk_subbuffer d_Qx = ggml_vk_tensor_subbuffer(ctx, src0);
    vk_subbuffer d_Qy = ggml_vk_tensor_subbuffer(ctx, src1, ਸੱਚ);

    vk_subbuffer d_F0 = d_D;

    uint32_t fusion_flags = 0;

    ਜੇ (ctx->ਅੰਕ_additional_fused_ops > 0) {
        ਅਚਲ ggml_tensor * ਜੋੜੋ_2 = cgraph->nodes[ਗੰਢ_idx + 1];
        ਅਚਲ ggml_tensor * bias = ਜੋੜੋ_2->src[0] == dst ? ਜੋੜੋ_2->src[1] : ਜੋੜੋ_2->src[0];

        d_F0 = ggml_vk_tensor_subbuffer(ctx, bias);
        fusion_flags |= MAT_VEC_FUSION_FLAGS_BIAS0;
    }

    vk_subbuffer d_F1 = d_D;
    ਜੇ (ctx->ਅੰਕ_additional_fused_ops > 1) {
        ਅਚਲ ggml_tensor * bias = cgraph->nodes[ਗੰਢ_idx + 2]->src[1];

        d_F1 = ggml_vk_tensor_subbuffer(ctx, bias);
        fusion_flags |= MAT_VEC_FUSION_FLAGS_BIAS1;
    }

    // compute

    vk_mat_vec_p021_ਧੱਕੋ_constants pc = {
        (uint32_t)ne00, (uint32_t)ne01, (uint32_t)ne02, (uint32_t)ne12,
        0, 0, fusion_flags
    };

    init_pushconst_tensor_offsets(ctx, pc, src0, src1, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, cgraph->nodes[ਗੰਢ_idx + ctx->ਅੰਕ_additional_fused_ops]);

    uint32_t workgroups_z = (uint32_t)ne12;
    // When gqa_ratio > 1, each invocation does multiple rows and we can launch fewer workgroups
    ਜੇ (gqa_ਅਨੁਪਾਤ > 1) {
        workgroups_z /= gqa_ਅਨੁਪਾਤ;
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

ਸਥਿਰ ਖਾਲੀ ggml_vk_mul_mat_vec_nc_f16_f32(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ਬਣਤਰ ggml_cgraph * cgraph, ਪੂਰਨ ਗੰਢ_idx) {
    ggml_tensor * dst = cgraph->nodes[ਗੰਢ_idx];
    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];
    VK_LOG_DEBUG("ggml_vk_mul_mat_nc_f16_f32((" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::cerr << "))");
    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));
    GGML_ASSERT(!ggml_is_permuted(src0));
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    ਅਚਲ uint64_t ne00 = src0->ne[0];
    ਅਚਲ uint64_t ne01 = src0->ne[1];
    ਅਚਲ uint64_t ne02 = src0->ne[2];
    ਅਚਲ uint64_t ne03 = src0->ne[3];

    ਅਚਲ uint64_t nb01 = src0->nb[1];
    ਅਚਲ uint64_t nb02 = src0->nb[2];

    ਅਚਲ uint64_t nb12 = src1->nb[2];

    // const uint64_t ne10 = src1->ne[0];
    ਅਚਲ uint64_t ne11 = src1->ne[1];
    ਅਚਲ uint64_t ne12 = src1->ne[2];
    // const uint64_t ne13 = src1->ne[3];

    ਅਚਲ uint32_t nb03 = (uint32_t)(src0->nb[3] / ਆਕਾਰ(ggml_fp16_t));
    ਅਚਲ uint32_t nb13 = (uint32_t)(src1->nb[3] / ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਅਚਲ uint32_t nb23 = (uint32_t)(dst->nb[3] / ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    GGML_ASSERT(ne11 == 1);
    GGML_ASSERT(src0->ne[3] == src1->ne[3]); // checked in supports_op

    ਅਚਲ uint32_t ਪੰਕਤੀ_stride_x = nb01 / ਆਕਾਰ(ggml_fp16_t);
    ਅਚਲ uint32_t channel_stride_x = nb02 / ਆਕਾਰ(ggml_fp16_t);
    ਅਚਲ uint32_t channel_stride_y = nb12 / ਆਕਾਰ(ਦਸ਼ਮਲਵ);

    vk_pipeline pipeline = ctx->device->pipeline_mul_mat_vec_nc_f16_f32;
    ਜੇ (ggml_nbytes(src0) > ctx->device->properties.limits.max_Storage_ਬਫਰ_Range) {
        pipeline = ggml_vk_get_64b_indexing_pipeline(ctx, pipeline);
    }

    {
        // Request descriptor sets
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    }

    vk_subbuffer d_D = ggml_vk_tensor_subbuffer(ctx, cgraph->nodes[ਗੰਢ_idx + ctx->ਅੰਕ_additional_fused_ops], ਸੱਚ);
    vk_subbuffer d_Qx = ggml_vk_tensor_subbuffer(ctx, src0);
    vk_subbuffer d_Qy = ggml_vk_tensor_subbuffer(ctx, src1, ਸੱਚ);
    vk_subbuffer d_F0 = d_D;

    uint32_t fusion_flags = 0;

    ਜੇ (ctx->ਅੰਕ_additional_fused_ops > 0) {
        ਅਚਲ ggml_tensor * ਜੋੜੋ_2 = cgraph->nodes[ਗੰਢ_idx + 1];
        ਅਚਲ ggml_tensor * bias = ਜੋੜੋ_2->src[0] == dst ? ਜੋੜੋ_2->src[1] : ਜੋੜੋ_2->src[0];

        d_F0 = ggml_vk_tensor_subbuffer(ctx, bias);
        fusion_flags |= MAT_VEC_FUSION_FLAGS_BIAS0;
    }

    vk_subbuffer d_F1 = d_D;
    ਜੇ (ctx->ਅੰਕ_additional_fused_ops > 1) {
        ਅਚਲ ggml_tensor * bias = cgraph->nodes[ਗੰਢ_idx + 2]->src[1];

        d_F1 = ggml_vk_tensor_subbuffer(ctx, bias);
        fusion_flags |= MAT_VEC_FUSION_FLAGS_BIAS1;
    }

    // compute
    vk_mat_vec_nc_ਧੱਕੋ_constants pc = {
        (uint32_t)ne00, (uint32_t)ne01,
        ਪੰਕਤੀ_stride_x, channel_stride_x, channel_stride_y,
        (uint32_t)(ne12 / ne02), (uint32_t)ne12,
        0, 0,
        nb03, nb13, nb23, fusion_flags
    };

    init_pushconst_tensor_offsets(ctx, pc, src0, src1, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, cgraph->nodes[ਗੰਢ_idx + ctx->ਅੰਕ_additional_fused_ops]);

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        {
            d_Qx,
            d_Qy,
            d_D,
            d_F0,
            d_F1,
        }, pc, { (uint32_t)ne03, (uint32_t)ne01, (uint32_t)ne12 });
}

ਸਥਿਰ ਪੂਰਨ ggml_vk_fwht_pipeline_idx(int64_t n) {
    ਚੋਣ (n) {
        ਹਾਲਤ 64:  ਵਾਪਸ 0;
        ਹਾਲਤ 128: ਵਾਪਸ 1;
        ਹਾਲਤ 256: ਵਾਪਸ 2;
        ਹਾਲਤ 512: ਵਾਪਸ 3;
        ਮੂਲ:  ਵਾਪਸ -1;
    }
}

ਤਾਰਕਿਕ ggml_vk_can_use_fwht(ਅਚਲ ggml_backend_vk_context * ctx, ਅਚਲ ggml_tensor * src1, ਅਚਲ ggml_tensor * dst) {
    ਜੇ (ctx->ਅੰਕ_additional_fused_ops != 0) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (ggml_get_op_params_i32(dst, 1) != GGML_HINT_SRC0_IS_HADAMARD) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ ਪੂਰਨ idx = ggml_vk_fwht_pipeline_idx(src1->ne[0]);
    ਜੇ (idx < 0 || ctx->device->pipeline_fwht_f32[idx] == ਖਾਲੀ_ਸੂਚਕ) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (!ggml_is_contiguous(src1)) {
        ਵਾਪਸ ਝੂਠ;
    }
    GGML_ASSERT(ggml_is_contiguous(dst));

    ਵਾਪਸ ਸੱਚ;
}

ਖਾਲੀ ggml_vk_fwht(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src, ggml_tensor * dst) {
    ਅਚਲ ਪੂਰਨ idx = ggml_vk_fwht_pipeline_idx(src->ne[0]);
    vk_pipeline pipeline = ctx->device->pipeline_fwht_f32[idx];

    ਅਚਲ uint32_t ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_workgroup = 4;
    ਅਚਲ uint32_t n_ਪੰਕਤੀਆਂ = (uint32_t)ggml_nrows(src);
    ਅਚਲ uint32_t max_workgroups_x = ctx->device->properties.limits.max_Compute_Work_ਸਮੂਹ_ਗਿਣਤੀ[0];

    ਅਚਲ uint32_t ਕੁੱਲ_workgroups = CEIL_DIV(n_ਪੰਕਤੀਆਂ, ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_workgroup);
    ਅਚਲ uint32_t workgroups_x = std::min(ਕੁੱਲ_workgroups, max_workgroups_x);
    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    ਅਚਲ vk_subbuffer src_buf = ggml_vk_tensor_subbuffer(ctx, src, ਸੱਚ);
    ਅਚਲ vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst, ਸੱਚ);

    vk_op_fwht_ਧੱਕੋ_constants pc = {
        n_ਪੰਕਤੀਆਂ,
        0,
        0,
        1.0f / std::ਵਰਗਮੂਲ((ਦਸ਼ਮਲਵ)src->ne[0]),
    };
    init_pushconst_tensor_offsets(ctx, pc, src, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, dst);

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src_buf, dst_buf }, pc, { workgroups_x, 1, 1 });
}

ਸਥਿਰ uint32_t ggml_vk_nb_elem(ਅਚਲ ggml_tensor * t, ਪੂਰਨ i) {
    ਵਾਪਸ (uint32_t)(t->nb[i] / ggml_type_ਮਾਪ(t->type));
}

ਖਾਲੀ ggml_vk_dsv4_hc_comb(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * mixes, ਅਚਲ ggml_tensor * scale, ਅਚਲ ggml_tensor * ਅਧਾਰ, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_vk_dsv4_hc_comb(" << mixes << ", " << scale << ", " << ਅਧਾਰ << ", " << dst << ")");

    vk_pipeline pipeline = ctx->device->pipeline_dsv4_hc_comb_f32;
    GGML_ASSERT(pipeline != ਖਾਲੀ_ਸੂਚਕ);

    ਅਚਲ uint32_t n_ਟੋਕਨ = (uint32_t)mixes->ne[1];

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    ਅਚਲ vk_subbuffer mixes_buf = ggml_vk_tensor_subbuffer(ctx, mixes, ਸੱਚ);
    ਅਚਲ vk_subbuffer scale_buf = ggml_vk_tensor_subbuffer(ctx, scale, ਸੱਚ);
    ਅਚਲ vk_subbuffer ਅਧਾਰ_buf  = ggml_vk_tensor_subbuffer(ctx, ਅਧਾਰ,  ਸੱਚ);
    ਅਚਲ vk_subbuffer dst_buf   = ggml_vk_tensor_subbuffer(ctx, dst,   ਸੱਚ);

    vk_op_dsv4_hc_comb_ਧੱਕੋ_constants pc = {
        n_ਟੋਕਨ,
        ggml_vk_nb_elem(mixes, 0), ggml_vk_nb_elem(mixes, 1),
        ggml_vk_nb_elem(scale, 0),
        ggml_vk_nb_elem(ਅਧਾਰ,  0),
        ggml_vk_nb_elem(dst,   0), ggml_vk_nb_elem(dst, 1), ggml_vk_nb_elem(dst, 2),
        0, 0, 0, 0,
        ggml_get_op_params_f32(dst, 0),
        (uint32_t)ggml_get_op_params_i32(dst, 1),
    };
    init_pushconst_tensor_offsets(ctx, pc, mixes, scale, ਅਧਾਰ, ਖਾਲੀ_ਸੂਚਕ, dst);

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { mixes_buf, scale_buf, ਅਧਾਰ_buf, dst_buf }, pc, { n_ਟੋਕਨ, 1, 1 });
}

ਖਾਲੀ ggml_vk_dsv4_hc_pre(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * x, ਅਚਲ ggml_tensor * weights, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_vk_dsv4_hc_pre(" << x << ", " << weights << ", " << dst << ")");

    ਅਚਲ ਦਸ਼ਮਲਵ scale = ggml_get_op_params_f32(dst, 0);
    ਅਚਲ ਤਾਰਕਿਕ  gated = ggml_get_op_params_i32(dst, 1) != 0;

    vk_pipeline pipeline = gated ? ctx->device->pipeline_dsv4_hc_pre_gated_f32 : ctx->device->pipeline_dsv4_hc_pre_f32;
    GGML_ASSERT(pipeline != ਖਾਲੀ_ਸੂਚਕ);

    ਅਚਲ uint32_t n_embd   = (uint32_t)x->ne[0];
    ਅਚਲ uint32_t n_ਟੋਕਨ = (uint32_t)x->ne[2];

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    ਅਚਲ vk_subbuffer x_buf = ggml_vk_tensor_subbuffer(ctx, x,       ਸੱਚ);
    ਅਚਲ vk_subbuffer w_buf = ggml_vk_tensor_subbuffer(ctx, weights, ਸੱਚ);
    ਅਚਲ vk_subbuffer d_buf = ggml_vk_tensor_subbuffer(ctx, dst,     ਸੱਚ);

    vk_op_dsv4_hc_pre_ਧੱਕੋ_constants pc = {
        n_embd, n_ਟੋਕਨ,
        ggml_vk_nb_elem(x, 0), ggml_vk_nb_elem(x, 1), ggml_vk_nb_elem(x, 2),
        ggml_vk_nb_elem(weights, 0), ggml_vk_nb_elem(weights, 1), ggml_vk_nb_elem(weights, 2),
        ggml_vk_nb_elem(dst, 0), ggml_vk_nb_elem(dst, 1),
        0, 0, 0,
        scale,
    };
    init_pushconst_tensor_offsets(ctx, pc, x, weights, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, dst);

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { x_buf, w_buf, d_buf }, pc, { n_embd, n_ਟੋਕਨ, 1 });
}

ਖਾਲੀ ggml_vk_dsv4_hc_post(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * x, ਅਚਲ ggml_tensor * residual, ਅਚਲ ggml_tensor * post, ਅਚਲ ggml_tensor * comb, ggml_tensor * dst, ਅਚਲ ggml_tensor * gate_scale_ਵਿੱਚ) {
    VK_LOG_DEBUG("ggml_vk_dsv4_hc_post(" << x << ", " << residual << ", " << post << ", " << comb << ", " << dst << ")");

    vk_pipeline pipeline = comb ? ctx->device->pipeline_dsv4_hc_post_f32 : ctx->device->pipeline_dsv4_hc_post_nocomb_f32;
    GGML_ASSERT(pipeline != ਖਾਲੀ_ਸੂਚਕ);

    ਅਚਲ uint32_t n_embd   = (uint32_t)x->ne[0];
    ਅਚਲ uint32_t n_ਟੋਕਨ = (uint32_t)x->ne[1];

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    ਅਚਲ vk_subbuffer x_buf = ggml_vk_tensor_subbuffer(ctx, x,        ਸੱਚ);
    ਅਚਲ vk_subbuffer r_buf = ggml_vk_tensor_subbuffer(ctx, residual, ਸੱਚ);
    // with a fused gate, post is scale(sigmoid(scale(p_src))) and the shader applies it to p_src
    ਅਚਲ ggml_tensor * p_src = gate_scale_ਵਿੱਚ ? gate_scale_ਵਿੱਚ->src[0] : post;
    ਅਚਲ vk_subbuffer p_buf = ggml_vk_tensor_subbuffer(ctx, p_src,    ਸੱਚ);
    ਅਚਲ vk_subbuffer c_buf = comb ? ggml_vk_tensor_subbuffer(ctx, comb, ਸੱਚ) : x_buf;
    ਅਚਲ vk_subbuffer d_buf = ggml_vk_tensor_subbuffer(ctx, dst,      ਸੱਚ);

    vk_op_dsv4_hc_post_ਧੱਕੋ_constants pc = {
        n_embd, n_ਟੋਕਨ,
        ggml_vk_nb_elem(x, 0), ggml_vk_nb_elem(x, 1),
        ggml_vk_nb_elem(residual, 0), ggml_vk_nb_elem(residual, 1), ggml_vk_nb_elem(residual, 2),
        ggml_vk_nb_elem(p_src, 0), ggml_vk_nb_elem(p_src, 1),
        comb ? ggml_vk_nb_elem(comb, 0) : 0, comb ? ggml_vk_nb_elem(comb, 1) : 0, comb ? ggml_vk_nb_elem(comb, 2) : 0,
        ggml_vk_nb_elem(dst,  0), ggml_vk_nb_elem(dst,  1), ggml_vk_nb_elem(dst,  2),
        0, 0, 0, 0, 0,
        gate_scale_ਵਿੱਚ ? 1u : 0u,
        gate_scale_ਵਿੱਚ ? ggml_get_op_params_f32(gate_scale_ਵਿੱਚ, 0) : 1.0f,
        gate_scale_ਵਿੱਚ ? ggml_get_op_params_f32(post, 0) : 1.0f,
    };
    init_pushconst_tensor_offsets(ctx, pc, x, residual, p_src, comb, dst);

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { x_buf, r_buf, p_buf, c_buf, d_buf }, pc, { n_embd, n_ਟੋਕਨ, 1 });
}

ਖਾਲੀ ggml_vk_mul_mat(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ਬਣਤਰ ggml_cgraph * cgraph, ਪੂਰਨ ਗੰਢ_idx) {
    ggml_tensor * dst = cgraph->nodes[ਗੰਢ_idx];
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];
    VK_LOG_DEBUG("ggml_vk_mul_mat(" << src0 << ", " << src1 << ", " << dst << ")");

    // Handle huge A matrix by splitting the M dimensions. This works well for convolution use cases
    // where the M dimension is very large.
    // Split_k doesn't work with M splitting.
    // This only supports batchsize == 1.
    ਅਚਲ size_t nbytes = ggml_nbytes(src0);
    ਅਚਲ ਤਾਰਕਿਕ needs_ਵੰਡੋ = dst->ne[2] == 1 && dst->ne[3] == 1 && nbytes > ctx->device->properties.limits.max_Storage_ਬਫਰ_Range;
    ਜੇ (needs_ਵੰਡੋ) {
        // Choose the number of rows that can fit (and divide by two, to allow for any additional offsets)
        ਅਚਲ uint32_t M_ਵੰਡੋ = ctx->device->properties.limits.max_Storage_ਬਫਰ_Range / (2 * src0->nb[1]);
        uint32_t m_ਵਿੱਥ = 0;
        ਜਦੋਂ (m_ਵਿੱਥ < dst->ne[0]) {
            ਅਚਲ uint32_t cur_M_ਮਾਪ = std::min(M_ਵੰਡੋ, (uint32_t)(dst->ne[0] - m_ਵਿੱਥ));
            ggml_tensor dst2 = *dst;
            ggml_tensor src02 = *src0;

            dst2.view_src = dst->view_src ? dst->view_src : dst;
            src02.view_src = src0->view_src ? src0->view_src : src0;

            dst2.view_offs += m_ਵਿੱਥ * dst->nb[0];
            src02.view_offs += m_ਵਿੱਥ * src0->nb[1];
            dst2.ne[0] = cur_M_ਮਾਪ;
            src02.ne[1] = cur_M_ਮਾਪ;

            ggml_vk_mul_mat_q_f16(ctx, subctx, &src02, src1, &dst2, ਸੱਚ);

            m_ਵਿੱਥ += cur_M_ਮਾਪ;
        }
    } ਨਹੀਂਤਾਂ ਜੇ (ggml_vk_can_use_fwht(ctx, src1, dst)) {
        ggml_vk_fwht(ctx, subctx, src1, dst);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && ggml_is_permuted(src0) && ggml_is_permuted(src1) && dst->ne[1] == 1 &&
        // detect 0213 permutation, and batch size of 1
        src0->nb[0] <= src0->nb[2] &&
        src0->nb[2] <= src0->nb[1] &&
        src0->nb[1] <= src0->nb[3] &&
        src1->nb[0] <= src1->nb[2] &&
        src1->nb[2] <= src1->nb[1] &&
        src1->nb[1] <= src1->nb[3] &&
        src0->ne[3] == 1 &&
        src1->ne[3] == 1 &&
        src0->ne[1] <= ctx->device->properties.limits.max_Compute_Work_ਸਮੂਹ_ਗਿਣਤੀ[1] &&
        src1->ne[2] <= ctx->device->properties.limits.max_Compute_Work_ਸਮੂਹ_ਗਿਣਤੀ[2]) {
        ggml_vk_mul_mat_vec_p021_f16_f32(ctx, subctx, cgraph, ਗੰਢ_idx);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && !ggml_is_contiguous(src0) && !ggml_is_transposed(src1) && dst->ne[1] == 1 &&
               !ggml_is_permuted(src0) && !ggml_is_permuted(src1) &&
               src0->ne[3] <= ctx->device->properties.limits.max_Compute_Work_ਸਮੂਹ_ਗਿਣਤੀ[0] &&
               src0->ne[1] <= ctx->device->properties.limits.max_Compute_Work_ਸਮੂਹ_ਗਿਣਤੀ[1] &&
               src1->ne[2] <= ctx->device->properties.limits.max_Compute_Work_ਸਮੂਹ_ਗਿਣਤੀ[2]) {
        ggml_vk_mul_mat_vec_nc_f16_f32(ctx, subctx, cgraph, ਗੰਢ_idx);
    // With one output row, B^T*A has the same flat output as A^T*B.
    } ਨਹੀਂਤਾਂ ਜੇ (ctx->ਅੰਕ_additional_fused_ops == 0 &&
               (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
               (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_BF16 || ggml_is_quantized(src1->type)) &&
               dst->ne[0] == 1 && dst->ne[1] > mul_mat_vec_max_ਖਾਨੇ_ਗਿਣਤੀ &&
               src0->ne[2] == 1 && src0->ne[3] == 1 &&
               src1->ne[2] == 1 && src1->ne[3] == 1 &&
               ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst) &&
               get_misalign_ਬਾਈਟਾਂ(ctx, src0) == 0 && get_misalign_ਬਾਈਟਾਂ(ctx, src1) == 0 && get_misalign_ਬਾਈਟਾਂ(ctx, dst) == 0) {
        ggml_vk_mul_mat_vec_q_f16(ctx, subctx, cgraph, ਗੰਢ_idx, ਸੱਚ);
    // mul_mat_vec supports batching ne12*ne13 when ne11==1, or treating ne11 as the batch size (up to four)
    // when ne12 and ne13 are one.
    } ਨਹੀਂਤਾਂ ਜੇ ((dst->ne[1] == 1 || (dst->ne[1] <= mul_mat_vec_max_ਖਾਨੇ_ਗਿਣਤੀ && src1->ne[2] * src1->ne[3] == 1)) &&
               (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16 || ggml_is_quantized(src0->type))) {
        ggml_vk_mul_mat_vec_q_f16(ctx, subctx, cgraph, ਗੰਢ_idx);
    } ਨਹੀਂਤਾਂ {
        ggml_vk_mul_mat_q_f16(ctx, subctx, src0, src1, dst, ਝੂਠ);
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_vk_mul_mat_id_q_f16(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ਅਚਲ ggml_tensor * ids, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_vk_mul_mat_id_q_f16((" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << ids << ", name=" << ids->name << ", type=" << ids->type << ", ne0=" << ids->ne[0] << ", ne1=" << ids->ne[1] << ", ne2=" << ids->ne[2] << ", ne3=" << ids->ne[3] << ", nb0=" << ids->nb[0] << ", nb1=" << ids->nb[1] << ", nb2=" << ids->nb[2] << ", nb3=" << ids->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3] << "),)");
    GGML_ASSERT(ggml_vk_dim01_contiguous(src1) || src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);  // NOLINT
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    ਅਚਲ uint64_t ne00 = src0->ne[0];
    ਅਚਲ uint64_t ne01 = src0->ne[1];
    ਅਚਲ uint64_t ne02 = src0->ne[2];
    // const uint64_t ne03 = src0->ne[3];

    ਅਚਲ uint64_t ne10 = src1->ne[0];
    ਅਚਲ uint64_t ne11 = src1->ne[1];
    ਅਚਲ uint64_t ne12 = src1->ne[2];
    ਅਚਲ uint64_t ne13 = src1->ne[3];

    ਅਚਲ uint64_t nei0 = ids->ne[0];
    ਅਚਲ uint64_t nei1 = ids->ne[1];

    ਅਚਲ uint32_t nbi0 = ids->nb[0];
    ਅਚਲ uint32_t nbi1 = ids->nb[1];

    ਅਚਲ uint64_t ne20 = dst->ne[0];
    ਅਚਲ uint64_t ne21 = dst->ne[1];
    // const uint64_t ne22 = dst->ne[2];
    // const uint64_t ne23 = dst->ne[3];

    ਅਚਲ uint64_t n_as = ne02;

    ggml_backend_vk_ਬਫਰ_context * dst_buf_ctx = (ggml_backend_vk_ਬਫਰ_context *)dst->ਬਫਰ->context;
    ggml_backend_vk_ਬਫਰ_context * src0_buf_ctx = (ggml_backend_vk_ਬਫਰ_context *)src0->ਬਫਰ->context;
    ggml_backend_vk_ਬਫਰ_context * src1_buf_ctx = (ggml_backend_vk_ਬਫਰ_context *)src1->ਬਫਰ->context;

    vk_ਬਫਰ d_Qx = ਖਾਲੀ_ਸੂਚਕ;
    size_t qx_buf_ਵਿੱਥ = 0;
    vk_ਬਫਰ d_Qy = ਖਾਲੀ_ਸੂਚਕ;
    size_t qy_buf_ਵਿੱਥ = 0;

    ਤਾਰਕਿਕ src0_uma = ਝੂਠ;
    ਤਾਰਕਿਕ src1_uma = ਝੂਠ;

    ਜੇ (ctx->device->uma) {
        ggml_vk_host_get(ctx->device, src0->ਅੰਕੜੇ, d_Qx, qx_buf_ਵਿੱਥ);
        ggml_vk_host_get(ctx->device, src1->ਅੰਕੜੇ, d_Qy, qy_buf_ਵਿੱਥ);
        src0_uma = d_Qx != ਖਾਲੀ_ਸੂਚਕ;
        src1_uma = d_Qy != ਖਾਲੀ_ਸੂਚਕ;
    }

    // Reformat and convert to fp16 if non-contiguous, or for coopmat2 for better perf
    ਅਚਲ ਤਾਰਕਿਕ x_non_contig = (ctx->device->coopmat2 && src0->type == GGML_TYPE_F32) ||
                              !ggml_vk_dim01_contiguous(src0);
    // If src0 is BF16, try to use a BF16 x BF16 multiply
    ggml_type f16_type = src0->type == GGML_TYPE_BF16 ? GGML_TYPE_BF16 : GGML_TYPE_F16;
#ਜੇ defined(GGML_VULKAN_COOPMAT2_DECODE_ਸਦਿਸ਼_GLSLC_SUPPORT)
    // B must already be, or be convertible to, the matmul B type used by this path.
    ਅਚਲ ਤਾਰਕਿਕ y_decode_ਸਦਿਸ਼_supported = ctx->device->coopmat2_decode_ਸਦਿਸ਼ &&
                                           (f16_type != GGML_TYPE_BF16 || ctx->device->coopmat2_bf16_support) &&
                                           (src1->type == GGML_TYPE_F32 || src1->type == f16_type);
    // If B is copied to prealloc_y, we can choose a 4-element-aligned row stride.
    ਅਚਲ ਤਾਰਕਿਕ y_decode_ਸਦਿਸ਼_ਵਰਤਦਾ_prealloc = !ggml_vk_dim01_contiguous(src1) || src1->type != f16_type;
    // Direct B reads are safe only if row starts and the original buffer offset are 4-element aligned.
    ਅਚਲ ਤਾਰਕਿਕ y_decode_ਸਦਿਸ਼_aligned =
        (ne10 % 4 == 0) &&
        (y_decode_ਸਦਿਸ਼_ਵਰਤਦਾ_prealloc || get_misalign_ਬਾਈਟਾਂ(ctx, src1) % (4 * ggml_type_ਮਾਪ(src1->type)) == 0);
    // Stage B only when decode-vector is available and direct B reads would be misaligned.
    ਅਚਲ ਤਾਰਕਿਕ y_decode_ਸਦਿਸ਼_staging = y_decode_ਸਦਿਸ਼_supported && !y_decode_ਸਦਿਸ਼_aligned;
#ਨਹੀਂਤਾਂ
    ਅਚਲ ਤਾਰਕਿਕ y_decode_ਸਦਿਸ਼_staging = ਝੂਠ;
#ਜੇ_ਅੰਤ
    ਅਚਲ ਤਾਰਕਿਕ can_quantize_y = ctx->device->integer_dot_ਗੁਣਨਫਲ && src1->type == GGML_TYPE_F32 &&
                                ggml_is_contiguous(src1) && (ne11 * ne10) % 4 == 0;
    // Intel coopmat1 matrix cores prefer f16 activations for quantized weights
    ਅਚਲ ਤਾਰਕਿਕ prefer_f16_b = ctx->device->coopmat_support && !ctx->device->coopmat2 &&
                              ctx->device->vendor_id == VK_VENDOR_ID_INTEL &&
                              ggml_is_quantized(src0->type) && src1->type == GGML_TYPE_F32;

    ਅਚਲ ggml_type y_kernel_type = ggml_vk_mul_mat_b_type(ctx, src0->type, src1, f16_type,
                                        (ggml_prec)dst->op_params[0], ਸੱਚ, can_quantize_y, prefer_f16_b);

    ਅਚਲ ਤਾਰਕਿਕ quantize_y   = y_kernel_type == GGML_TYPE_Q8_1;
    ਅਚਲ ਤਾਰਕਿਕ y_f32_kernel = y_kernel_type == GGML_TYPE_F32;

    ਅਚਲ std::vector<vk_matmul_pipeline_pair>* mmp_map =
        ggml_vk_get_mul_mat_mat_pipeline_map(ctx, src0->type, y_kernel_type, (ggml_prec)dst->op_params[0], ਸੱਚ);

    ਅਚਲ ਤਾਰਕਿਕ qx_needs_dequant = mmp_map == ਖਾਲੀ_ਸੂਚਕ || x_non_contig;

    ਜੇ (qx_needs_dequant) {
        // dequant src0 to f16
        mmp_map = ggml_vk_get_mul_mat_mat_pipeline_map(ctx, f16_type, y_kernel_type, (ggml_prec)dst->op_params[0], ਸੱਚ);
    }

    GGML_ASSERT(mmp_map != ਖਾਲੀ_ਸੂਚਕ);

    // Coopmat2 MUL_MAT_ID BK specialization constants in ggml_vk_load_shaders are at most 64.
    ਅਚਲ uint32_t y_staged_ਪੰਕਤੀ_stride = ctx->device->coopmat2 && !quantize_y ? ggml_vk_align_ਮਾਪ(ne10, 64) : ne10;
    ਅਚਲ ਤਾਰਕਿਕ y_needs_k_padding = ne10 != y_staged_ਪੰਕਤੀ_stride;

    ਅਚਲ ਤਾਰਕਿਕ y_needs_reformat = !quantize_y && (!ggml_vk_dim01_contiguous(src1) || y_kernel_type != src1->type ||
                                                  y_decode_ਸਦਿਸ਼_staging || y_needs_k_padding);
    ਅਚਲ ਤਾਰਕਿਕ qy_needs_dequant = y_needs_reformat;

    ਅਚਲ uint32_t kpad = quantize_y ? 0 : ggml_vk_align_ਮਾਪ(ne10, ggml_vk_guess_matmul_pipeline_align_map(ctx, *mmp_map, ne01, nei1, ਸੱਚ));
    ਅਚਲ ਤਾਰਕਿਕ aligned = !quantize_y && ne10 == kpad && ne01 > 8 && nei1 > 8;

    vk_pipeline pipeline = ggml_vk_guess_matmul_pipeline_map(ctx, *mmp_map, ne01, nei1, aligned, ਸੱਚ);

    ਅਚਲ uint32_t bn = pipeline->wg_denoms[1];
    ਅਚਲ uint64_t max_tiles = ggml_vk_mul_mat_id_max_tiles(n_as, nei0 * nei1, bn);
    ਅਚਲ uint64_t ਪੰਕਤੀ_ids_ਵਿੱਥ = 1 + 2 * max_tiles;
    ਅਚਲ uint64_t ਪੰਕਤੀ_map_ਸ਼ਬਦ = ਪੰਕਤੀ_ids_ਵਿੱਥ + max_tiles * bn;

    ਜੇ (ggml_nbytes(src0) > ctx->device->properties.limits.max_Storage_ਬਫਰ_Range) {
        pipeline = ggml_vk_get_64b_indexing_pipeline(ctx, pipeline);
    }
    ਅਚਲ uint64_t x_ne = ggml_nelements(src0);
    ਅਚਲ uint64_t y_ne = (uint64_t)y_staged_ਪੰਕਤੀ_stride * ne11 * ne12 * ne13;
    ਅਚਲ uint64_t d_ne = ggml_nelements(dst);

    ਅਚਲ uint64_t qx_sz = ggml_type_ਮਾਪ(src0->type) * x_ne / ggml_blck_ਮਾਪ(src0->type);
    ਅਚਲ uint64_t qy_sz = ggml_type_ਮਾਪ(src1->type) * ggml_nelements(src1) / ggml_blck_ਮਾਪ(src1->type);
    ਅਚਲ uint64_t x_sz = !qx_needs_dequant ? qx_sz : ਆਕਾਰ(ggml_fp16_t) * x_ne;
    ਅਚਲ uint64_t y_sz = quantize_y ? (ggml_vk_align_ਮਾਪ(y_ne, 128) * ggml_type_ਮਾਪ(GGML_TYPE_Q8_1) / ggml_blck_ਮਾਪ(GGML_TYPE_Q8_1)) : (y_f32_kernel ? ਆਕਾਰ(ਦਸ਼ਮਲਵ) * y_ne : ਆਕਾਰ(ggml_fp16_t) * y_ne);
    ਅਚਲ uint64_t d_sz = ਆਕਾਰ(ਦਸ਼ਮਲਵ) * d_ne;

    vk_pipeline ਮੰਜ਼ਿਲ_fp16_vk_0 = ਖਾਲੀ_ਸੂਚਕ;
    vk_pipeline ਮੰਜ਼ਿਲ_fp16_vk_1 = ਖਾਲੀ_ਸੂਚਕ;
    vk_pipeline ਮੰਜ਼ਿਲ_q8_1 = ਖਾਲੀ_ਸੂਚਕ;

    ਸਵੈਚਾਲਿਤ make_y_staged_dst = [&]() {
        ggml_tensor y_staged_dst = *src1;
        y_staged_dst.type = f16_type;
        y_staged_dst.nb[0] = ggml_type_ਮਾਪ(f16_type);
        y_staged_dst.nb[1] = y_staged_dst.nb[0] * y_staged_ਪੰਕਤੀ_stride;
        y_staged_dst.nb[2] = y_staged_dst.nb[1] * ne11;
        y_staged_dst.nb[3] = y_staged_dst.nb[2] * y_staged_dst.ne[2];
        ਵਾਪਸ y_staged_dst;
    };

    ਜੇ (x_non_contig) {
        ਮੰਜ਼ਿਲ_fp16_vk_0 = ggml_vk_get_cpy_pipeline(ctx, src0, ਖਾਲੀ_ਸੂਚਕ, f16_type);
    } ਨਹੀਂਤਾਂ {
        ਮੰਜ਼ਿਲ_fp16_vk_0 = ggml_vk_get_ਮੰਜ਼ਿਲ_fp16(ctx, src0->type);
    }
    ਜੇ (y_needs_reformat) {
        ggml_tensor y_staged_dst;
        ਅਚਲ ggml_tensor * y_staged_dst_ptr = ਖਾਲੀ_ਸੂਚਕ;
        ਜੇ (y_needs_k_padding) {
            y_staged_dst = make_y_staged_dst();
            y_staged_dst_ptr = &y_staged_dst;
        }

        ਮੰਜ਼ਿਲ_fp16_vk_1 = ggml_vk_get_cpy_pipeline(ctx, src1, y_staged_dst_ptr, f16_type);
    } ਨਹੀਂਤਾਂ {
        ਮੰਜ਼ਿਲ_fp16_vk_1 = ggml_vk_get_ਮੰਜ਼ਿਲ_fp16(ctx, src1->type);
    }
    GGML_ASSERT(!qx_needs_dequant || ਮੰਜ਼ਿਲ_fp16_vk_0 != ਖਾਲੀ_ਸੂਚਕ);  // NOLINT
    GGML_ASSERT(!qy_needs_dequant || ਮੰਜ਼ਿਲ_fp16_vk_1 != ਖਾਲੀ_ਸੂਚਕ);  // NOLINT

    ਜੇ (quantize_y) {
        ਮੰਜ਼ਿਲ_q8_1 = ggml_vk_get_quantize_pipeline(ctx, GGML_TYPE_Q8_1);
    }
    vk_pipeline ਗਿਣਤੀ_experts = ctx->device->pipeline_ਗਿਣਤੀ_experts;

    ਅਚਲ size_t expert_ਅੰਕੜੇ_ਮਾਪ = ਆਕਾਰ(uint32_t) * ਪੰਕਤੀ_map_ਸ਼ਬਦ;

    {
        ਜੇ (
                (qx_needs_dequant && x_sz > ctx->device->properties.limits.max_Storage_ਬਫਰ_Range) ||
                (qy_needs_dequant && y_sz > ctx->device->properties.limits.max_Storage_ਬਫਰ_Range)) {
            GGML_ABORT("Requested preallocation size is too large");
        }
        ਜੇ (qx_needs_dequant && ctx->prealloc_ਮਾਪ_x < x_sz) {
            ctx->prealloc_ਮਾਪ_x = x_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        ਜੇ ((qy_needs_dequant || quantize_y) && ctx->prealloc_ਮਾਪ_y < y_sz) {
            ctx->prealloc_ਮਾਪ_y = y_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        ਜੇ (ctx->prealloc_ਮਾਪ_ਵੰਡੋ_k < expert_ਅੰਕੜੇ_ਮਾਪ) {
            ctx->prealloc_ਮਾਪ_ਵੰਡੋ_k = expert_ਅੰਕੜੇ_ਮਾਪ;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }

        // Request descriptor sets
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
        ਜੇ (qx_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, ਮੰਜ਼ਿਲ_fp16_vk_0, 1);
        }
        ਜੇ (qy_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, ਮੰਜ਼ਿਲ_fp16_vk_1, 1);
        }
        ਜੇ (quantize_y) {
            ggml_pipeline_request_descriptor_sets(ctx, ਮੰਜ਼ਿਲ_q8_1, 1);
        }
        ggml_pipeline_request_descriptor_sets(ctx, ਗਿਣਤੀ_experts, 1);
    }

    vk_ਬਫਰ d_D = dst_buf_ctx->dev_ਬਫਰ;
    ਅਚਲ uint64_t d_buf_ਵਿੱਥ = vk_tensor_ਵਿੱਥ(dst) + dst->view_offs;
    GGML_ASSERT(d_D != ਖਾਲੀ_ਸੂਚਕ);
    vk_ਬਫਰ d_X;
    uint64_t x_buf_ਵਿੱਥ = 0;
    vk_ਬਫਰ d_Y;
    uint64_t y_buf_ਵਿੱਥ = 0;
    ਜੇ (!src0_uma) {
        d_Qx = src0_buf_ctx->dev_ਬਫਰ;
        qx_buf_ਵਿੱਥ = vk_tensor_ਵਿੱਥ(src0) + src0->view_offs;
        GGML_ASSERT(d_Qx != ਖਾਲੀ_ਸੂਚਕ);
    }
    ਜੇ (!src1_uma) {
        d_Qy = src1_buf_ctx->dev_ਬਫਰ;
        qy_buf_ਵਿੱਥ = vk_tensor_ਵਿੱਥ(src1) + src1->view_offs;
        GGML_ASSERT(d_Qy != ਖਾਲੀ_ਸੂਚਕ);
    }
    ਜੇ (qx_needs_dequant) {
        d_X = ctx->prealloc_x;
        GGML_ASSERT(d_X->size >= x_sz);
    } ਨਹੀਂਤਾਂ {
        d_X = d_Qx;
        x_buf_ਵਿੱਥ = qx_buf_ਵਿੱਥ;
        GGML_ASSERT(qx_sz == x_sz);
    }
    ਜੇ (qy_needs_dequant) {
        d_Y = ctx->prealloc_y;
        GGML_ASSERT(d_Y->size >= y_sz);
    } ਨਹੀਂਤਾਂ ਜੇ (quantize_y) {
        d_Y = ctx->prealloc_y;
        GGML_ASSERT(d_Y->size >= CEIL_DIV(y_sz, 144) * 144);
    } ਨਹੀਂਤਾਂ {
        d_Y = d_Qy;
        y_buf_ਵਿੱਥ = qy_buf_ਵਿੱਥ;
        GGML_ASSERT(qy_sz == y_sz);
    }

    ਜੇ (x_non_contig || qx_needs_dequant) {
        ਜੇ (ctx->prealloc_x_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }
    }
    vk_subbuffer d_ids = ggml_vk_tensor_subbuffer(ctx, ids, ਸੱਚ);

    // Count how many times each expert is used
    vk_subbuffer expert_ਗਿਣਤੀ_buf = { ctx->prealloc_ਵੰਡੋ_k, 0, expert_ਅੰਕੜੇ_ਮਾਪ };
    ਜੇ (ctx->prealloc_ਵੰਡੋ_k_need_sync) {
        ggml_vk_sync_buffers(ctx, subctx);
    }
    {
        vk_op_ਗਿਣਤੀ_experts_ਧੱਕੋ_constants pc = { (uint32_t)nei0,
                                           (uint32_t)nei1,
                                           (uint32_t)(nbi0 / ggml_type_ਮਾਪ(ids->type)),
                                           (uint32_t)(nbi1 / ggml_type_ਮਾਪ(ids->type)),
                                           (uint32_t)(get_misalign_ਬਾਈਟਾਂ(ctx, ids) / ggml_type_ਮਾਪ(ids->type)),
                                           (uint32_t)n_as,
                                           0, 0, bn, (uint32_t)ਪੰਕਤੀ_ids_ਵਿੱਥ };
        init_pushconst_fastdiv(pc);
        ggml_vk_dispatch_pipeline(ctx, subctx, ਗਿਣਤੀ_experts,
            { d_ids, expert_ਗਿਣਤੀ_buf }, pc,
            { 1, 1, 1 });
    }

    ਜੇ (x_non_contig) {
        ggml_vk_cpy_ਮੰਜ਼ਿਲ_contiguous(ctx, subctx, ਮੰਜ਼ਿਲ_fp16_vk_0, src0, ggml_vk_subbuffer(ctx, d_Qx, qx_buf_ਵਿੱਥ), ggml_vk_subbuffer(ctx, d_X, 0));
    } ਨਹੀਂਤਾਂ ਜੇ (qx_needs_dequant) {
        ਅਚਲ std::vector<uint32_t> pc = { (uint32_t)ne01, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)(ggml_nelements(src0)) };
        ggml_vk_dispatch_pipeline(ctx, subctx, ਮੰਜ਼ਿਲ_fp16_vk_0,
            { vk_subbuffer{ d_Qx, qx_buf_ਵਿੱਥ, qx_sz }, vk_subbuffer{ d_X, 0, x_sz } }, pc, { (uint32_t)x_ne, 1, 1});
    }
    ਜੇ (y_needs_reformat) {
        ਜੇ (ctx->prealloc_y_ਆਖਰੀ_pipeline_ਵਰਤਿਆ != ਮੰਜ਼ਿਲ_fp16_vk_1.get() ||
            ctx->prealloc_y_ਆਖਰੀ_tensor_ਵਰਤਿਆ != src1 ||
            ctx->prealloc_y_ਆਖਰੀ_k_padded != y_needs_k_padding) {
            ਜੇ (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ਜੇ (y_needs_k_padding) {
                GGML_ASSERT(y_sz % 4 == 0);
                // Zero B padding because clamping only A can produce 0 * Inf or NaN.
                subctx->s->ਬਫਰ->buf.fill_ਬਫਰ(d_Y->ਬਫਰ, 0, y_sz, 0);
                ggml_vk_sync_buffers(ctx, subctx);
                ਅਚਲ ggml_tensor y_staged_dst = make_y_staged_dst();
                ਅਚਲ uint32_t y_staged_dst_type_ਮਾਪ = ggml_type_ਮਾਪ(y_staged_dst.type);
                ggml_vk_cpy_ਮੰਜ਼ਿਲ_strided(
                    ctx, subctx, ਮੰਜ਼ਿਲ_fp16_vk_1, src1,
                    ggml_vk_subbuffer(ctx, d_Qy, qy_buf_ਵਿੱਥ), ggml_vk_subbuffer(ctx, d_Y, 0),
                    (uint32_t)(y_staged_dst.nb[0] / y_staged_dst_type_ਮਾਪ),
                    (uint32_t)(y_staged_dst.nb[1] / y_staged_dst_type_ਮਾਪ),
                    (uint32_t)(y_staged_dst.nb[2] / y_staged_dst_type_ਮਾਪ),
                    (uint32_t)(y_staged_dst.nb[3] / y_staged_dst_type_ਮਾਪ));
            } ਨਹੀਂਤਾਂ {
                ggml_vk_cpy_ਮੰਜ਼ਿਲ_contiguous(ctx, subctx, ਮੰਜ਼ਿਲ_fp16_vk_1, src1, ggml_vk_subbuffer(ctx, d_Qy, qy_buf_ਵਿੱਥ), ggml_vk_subbuffer(ctx, d_Y, 0));
            }
            ctx->prealloc_y_ਆਖਰੀ_pipeline_ਵਰਤਿਆ = ਮੰਜ਼ਿਲ_fp16_vk_1.get();
            ctx->prealloc_y_ਆਖਰੀ_tensor_ਵਰਤਿਆ = src1;
            ctx->prealloc_y_ਆਖਰੀ_k_padded = y_needs_k_padding;
        }
    }
    ਜੇ (quantize_y) {
        ਜੇ (ctx->prealloc_y_ਆਖਰੀ_pipeline_ਵਰਤਿਆ != ਮੰਜ਼ਿਲ_q8_1.get() ||
            ctx->prealloc_y_ਆਖਰੀ_tensor_ਵਰਤਿਆ != src1 ||
            ctx->prealloc_y_ਆਖਰੀ_k_padded) {
            ਜੇ (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ggml_vk_quantize_q8_1(ctx, subctx, ggml_vk_subbuffer(ctx, d_Qy, qy_buf_ਵਿੱਥ), ggml_vk_subbuffer(ctx, d_Y, 0), y_ne);
            ctx->prealloc_y_ਆਖਰੀ_pipeline_ਵਰਤਿਆ = ਮੰਜ਼ਿਲ_q8_1.get();
            ctx->prealloc_y_ਆਖਰੀ_tensor_ਵਰਤਿਆ = src1;
            ctx->prealloc_y_ਆਖਰੀ_k_padded = ਝੂਠ;
        }
    }
    ggml_vk_sync_buffers(ctx, subctx);

    ਅਚਲ uint64_t x_range = qx_needs_dequant ? x_sz : ggml_nbytes(src0);
    ਅਚਲ uint64_t y_range = (qy_needs_dequant || quantize_y) ? y_sz : ggml_nbytes(src1);

    uint32_t stride_batch_x = qx_needs_dequant ? ne00*ne01 : ggml_vk_batch_stride(src0);
    uint32_t stride_b_y = y_needs_k_padding ? y_staged_ਪੰਕਤੀ_stride : ne10;
    uint32_t stride_batch_y = y_needs_k_padding ? y_staged_ਪੰਕਤੀ_stride * ne11 : ((qy_needs_dequant || quantize_y) ? ne10*ne11 : ggml_vk_batch_stride(src1));

    ਜੇ (!ggml_vk_dim01_contiguous(src0) && !qx_needs_dequant) {
        stride_batch_x = src0->nb[0] / ggml_type_ਮਾਪ(src0->type);
    }

    ਜੇ (!ggml_vk_dim01_contiguous(src1) && !qy_needs_dequant && !quantize_y) {
        stride_batch_y = src1->nb[0] / ggml_type_ਮਾਪ(src1->type);
    }

    // compute
    ggml_vk_matmul_id(
        ctx, subctx, pipeline,
        { d_X, x_buf_ਵਿੱਥ, x_range }, { d_Y, y_buf_ਵਿੱਥ, y_range },
        { d_D, d_buf_ਵਿੱਥ, d_sz }, std::move(d_ids), expert_ਗਿਣਤੀ_buf,
        ne01, ne21, ne10, ne10, stride_b_y, ne01,
        stride_batch_x, stride_batch_y, ne20*ne21,
        n_as, nei0, nei1, nbi1 / ggml_type_ਮਾਪ(ids->type), ne11, ਪੰਕਤੀ_ids_ਵਿੱਥ, max_tiles
    );  // NOLINT

    ਜੇ (x_non_contig || qx_needs_dequant) {
        ctx->prealloc_x_need_sync = ਸੱਚ;
    }
    ਜੇ (y_needs_reformat || quantize_y) {
        ctx->prealloc_y_need_sync = ਸੱਚ;
    }
    ctx->prealloc_ਵੰਡੋ_k_need_sync = ਸੱਚ;
}

ਸਥਿਰ ਖਾਲੀ ggml_vk_mul_mat_vec_id_q_f16(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ਬਣਤਰ ggml_cgraph * cgraph, ਪੂਰਨ ਗੰਢ_idx) {
    ggml_tensor * dst = cgraph->nodes[ਗੰਢ_idx];
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];
    ggml_tensor * ids = dst->src[2];
    VK_LOG_DEBUG("ggml_vk_mul_mat_vec_id_q_f16((" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << ids << ", name=" << ids->name << ", type=" << ids->type << ", ne0=" << ids->ne[0] << ", ne1=" << ids->ne[1] << ", ne2=" << ids->ne[2] << ", ne3=" << ids->ne[3] << ", nb0=" << ids->nb[0] << ", nb1=" << ids->nb[1] << ", nb2=" << ids->nb[2] << ", nb3=" << ids->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::cerr << "))");
    GGML_ASSERT(ggml_vk_dim01_contiguous(src0) || src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16);  // NOLINT
    GGML_ASSERT(ggml_vk_dim01_contiguous(src1) || src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);  // NOLINT
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    ਅਚਲ uint64_t ne00 = src0->ne[0];
    ਅਚਲ uint64_t ne01 = src0->ne[1];
    // const uint64_t ne02 = src0->ne[2];
    // const uint64_t ne03 = src0->ne[3];

    ਅਚਲ uint64_t ne10 = src1->ne[0];
    ਅਚਲ uint64_t ne11 = src1->ne[1];
    ਅਚਲ uint64_t ne12 = src1->ne[2];
    // const uint64_t ne13 = src1->ne[3];

    ਅਚਲ uint64_t nei0 = ids->ne[0];
    ਅਚਲ uint64_t nei1 = ids->ne[1];
    ਅਚਲ uint32_t nbi1 = (uint32_t)(ids->nb[1] / ਆਕਾਰ(ਪੂਰਨ));

    ਅਚਲ uint64_t ne20 = dst->ne[0];
    ਅਚਲ uint64_t ne21 = dst->ne[1];
    // const uint64_t ne22 = dst->ne[2];
    // const uint64_t ne23 = dst->ne[3];

    ਅਚਲ ਤਾਰਕਿਕ x_non_contig = !ggml_vk_dim01_contiguous(src0);
    ਅਚਲ ਤਾਰਕਿਕ y_non_contig = !ggml_vk_dim01_contiguous(src1);

    ਅਚਲ ਤਾਰਕਿਕ f16_f32_kernel = src1->type == GGML_TYPE_F32;
    ਤਾਰਕਿਕ quantize_y = ctx->device->integer_dot_ਗੁਣਨਫਲ && src1->type == GGML_TYPE_F32 && ggml_is_contiguous(src1) && !y_non_contig && (ne11 * ne10) % 4 == 0 && ggml_vk_should_use_mmvq(ctx->device, ne01, ne12, ne10, src0->type);

    vk_pipeline ਮੰਜ਼ਿਲ_fp16_vk_0 = ਖਾਲੀ_ਸੂਚਕ;
    vk_pipeline ਮੰਜ਼ਿਲ_fp16_vk_1 = ਖਾਲੀ_ਸੂਚਕ;
    ਜੇ (x_non_contig) {
        ਮੰਜ਼ਿਲ_fp16_vk_0 = ggml_vk_get_cpy_pipeline(ctx, src0, ਖਾਲੀ_ਸੂਚਕ, src0->type);
    }
    ਜੇ (y_non_contig) {
        ਮੰਜ਼ਿਲ_fp16_vk_1 = ggml_vk_get_cpy_pipeline(ctx, src1, ਖਾਲੀ_ਸੂਚਕ, src1->type);
    } ਨਹੀਂਤਾਂ {
        ਮੰਜ਼ਿਲ_fp16_vk_1 = ggml_vk_get_ਮੰਜ਼ਿਲ_fp16(ctx, src1->type);
    }

    // Check for mmq first
    vk_pipeline dmmv = quantize_y ? ggml_vk_get_dequantize_mul_mat_vec_id(ctx, src0->type, GGML_TYPE_Q8_1, ne20, ne00) : ਖਾਲੀ_ਸੂਚਕ;
    vk_pipeline ਮੰਜ਼ਿਲ_q8_1 = ਖਾਲੀ_ਸੂਚਕ;

    ਜੇ (dmmv == ਖਾਲੀ_ਸੂਚਕ) {
        // Fall back to f16 dequant mul mat
        dmmv = ggml_vk_get_dequantize_mul_mat_vec_id(ctx, src0->type, src1->type, ne20, ne00);
        quantize_y = ਝੂਠ;
    }

    ਜੇ (quantize_y) {
        ਮੰਜ਼ਿਲ_q8_1 = ggml_vk_get_quantize_pipeline(ctx, GGML_TYPE_Q8_1);
    }

    ਅਚਲ ਤਾਰਕਿਕ qx_needs_dequant = x_non_contig;
    ਅਚਲ ਤਾਰਕਿਕ qy_needs_dequant = !quantize_y && ((src1->type != GGML_TYPE_F16 && !f16_f32_kernel) || y_non_contig);

    ਜੇ (ggml_nbytes(src0) > ctx->device->properties.limits.max_Storage_ਬਫਰ_Range) {
        dmmv = ggml_vk_get_64b_indexing_pipeline(ctx, dmmv);
    }

    // Not implemented
    GGML_ASSERT(y_non_contig || !qy_needs_dequant);  // NOLINT
    GGML_ASSERT(!qx_needs_dequant || ਮੰਜ਼ਿਲ_fp16_vk_0 != ਖਾਲੀ_ਸੂਚਕ);  // NOLINT
    GGML_ASSERT(!qy_needs_dequant || ਮੰਜ਼ਿਲ_fp16_vk_1 != ਖਾਲੀ_ਸੂਚਕ);  // NOLINT
    GGML_ASSERT(dmmv != ਖਾਲੀ_ਸੂਚਕ);

    ਅਚਲ uint64_t x_ne = ggml_nelements(src0);
    ਅਚਲ uint64_t y_ne = ggml_nelements(src1);

    ਅਚਲ uint64_t qx_sz = ggml_vk_align_ਮਾਪ(ggml_type_ਮਾਪ(src0->type) * x_ne / ggml_blck_ਮਾਪ(src0->type), ctx->device->properties.limits.min_Storage_ਬਫਰ_ਵਿੱਥ_Alignment);
    ਅਚਲ uint64_t x_sz = x_non_contig ? ggml_vk_align_ਮਾਪ(ggml_type_ਮਾਪ(src0->type) * x_ne, ctx->device->properties.limits.min_Storage_ਬਫਰ_ਵਿੱਥ_Alignment) : qx_sz;
    ਅਚਲ uint64_t y_sz = quantize_y ? (ggml_vk_align_ਮਾਪ(y_ne, 128) * ggml_type_ਮਾਪ(GGML_TYPE_Q8_1) / ggml_blck_ਮਾਪ(GGML_TYPE_Q8_1)) :
                                       (f16_f32_kernel ? ਆਕਾਰ(ਦਸ਼ਮਲਵ) * y_ne : ਆਕਾਰ(ggml_fp16_t) * y_ne);

    {
        ਜੇ (
                (qx_needs_dequant && x_sz > ctx->device->properties.limits.max_Storage_ਬਫਰ_Range) ||
                (qy_needs_dequant && y_sz > ctx->device->properties.limits.max_Storage_ਬਫਰ_Range)) {
            GGML_ABORT("Requested preallocation size is too large");
        }
        ਜੇ (qx_needs_dequant && ctx->prealloc_ਮਾਪ_x < x_sz) {
            ctx->prealloc_ਮਾਪ_x = x_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        ਜੇ ((qy_needs_dequant || quantize_y) && ctx->prealloc_ਮਾਪ_y < y_sz) {
            ctx->prealloc_ਮਾਪ_y = y_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }

        // Request descriptor sets
        ਜੇ (qx_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, ਮੰਜ਼ਿਲ_fp16_vk_0, 1);
        }
        ਜੇ (qy_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, ਮੰਜ਼ਿਲ_fp16_vk_1, 1);
        }
        ਜੇ (quantize_y) {
            ggml_pipeline_request_descriptor_sets(ctx, ਮੰਜ਼ਿਲ_q8_1, 1);
        }
        ggml_pipeline_request_descriptor_sets(ctx, dmmv, nei1);
    }

    vk_subbuffer d_D = ggml_vk_tensor_subbuffer(ctx, cgraph->nodes[ਗੰਢ_idx + ctx->ਅੰਕ_additional_fused_ops]);
    vk_subbuffer d_Qx = ggml_vk_tensor_subbuffer(ctx, src0);
    vk_subbuffer d_Qy = ggml_vk_tensor_subbuffer(ctx, src1);
    vk_subbuffer d_ids = ggml_vk_tensor_subbuffer(ctx, ids);
    vk_subbuffer d_F0 = d_D;
    vk_subbuffer d_X, d_Y;

    ਜੇ (qx_needs_dequant) {
        d_X = { ctx->prealloc_x, 0, ctx->prealloc_x->size };
    } ਨਹੀਂਤਾਂ {
        d_X = d_Qx;
    }
    ਜੇ (qy_needs_dequant || quantize_y) {
        d_Y = { ctx->prealloc_y, 0, ctx->prealloc_y->size };
    } ਨਹੀਂਤਾਂ {
        d_Y = d_Qy;
    }

    ਜੇ (x_non_contig) {
        ਜੇ (ctx->prealloc_x_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }
    }

    ਜੇ (x_non_contig) {
        GGML_ASSERT(x_sz == ggml_vk_align_ਮਾਪ(ggml_type_ਮਾਪ(src0->type) * x_ne, ctx->device->properties.limits.min_Storage_ਬਫਰ_ਵਿੱਥ_Alignment));
        ggml_vk_cpy_ਮੰਜ਼ਿਲ_contiguous(ctx, subctx, ਮੰਜ਼ਿਲ_fp16_vk_0, src0, d_Qx, d_X);
    }
    ਜੇ (y_non_contig) {
        GGML_ASSERT(y_sz == ggml_type_ਮਾਪ(src1->type) * y_ne);
        ਜੇ (ctx->prealloc_y_ਆਖਰੀ_pipeline_ਵਰਤਿਆ != ਮੰਜ਼ਿਲ_fp16_vk_1.get() ||
            ctx->prealloc_y_ਆਖਰੀ_tensor_ਵਰਤਿਆ != src1 ||
            ctx->prealloc_y_ਆਖਰੀ_k_padded) {
            ਜੇ (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ggml_vk_cpy_ਮੰਜ਼ਿਲ_contiguous(ctx, subctx, ਮੰਜ਼ਿਲ_fp16_vk_1, src1, d_Qy, d_Y);
            ctx->prealloc_y_ਆਖਰੀ_pipeline_ਵਰਤਿਆ = ਮੰਜ਼ਿਲ_fp16_vk_1.get();
            ctx->prealloc_y_ਆਖਰੀ_tensor_ਵਰਤਿਆ = src1;
            ctx->prealloc_y_ਆਖਰੀ_k_padded = ਝੂਠ;
        }
    }
    ਜੇ (quantize_y) {
        ਜੇ (ctx->prealloc_y_ਆਖਰੀ_pipeline_ਵਰਤਿਆ != ਮੰਜ਼ਿਲ_q8_1.get() ||
            ctx->prealloc_y_ਆਖਰੀ_tensor_ਵਰਤਿਆ != src1 ||
            ctx->prealloc_y_ਆਖਰੀ_k_padded) {
            ਜੇ (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ggml_vk_quantize_q8_1(ctx, subctx, d_Qy, d_Y, y_ne);
            ctx->prealloc_y_ਆਖਰੀ_pipeline_ਵਰਤਿਆ = ਮੰਜ਼ਿਲ_q8_1.get();
            ctx->prealloc_y_ਆਖਰੀ_tensor_ਵਰਤਿਆ = src1;
            ctx->prealloc_y_ਆਖਰੀ_k_padded = ਝੂਠ;
        }
    }

    uint32_t stride_batch_x = qx_needs_dequant ? ne00*ne01 : ggml_vk_batch_stride(src0);
    uint32_t stride_batch_y = (qy_needs_dequant || quantize_y) ? ne10*ne11 : ggml_vk_batch_stride(src1);

    ਜੇ (!ggml_vk_dim01_contiguous(src1) && !qy_needs_dequant) {
        stride_batch_y = src1->nb[2] / ggml_type_ਮਾਪ(src1->type);
    }

    ਅਚਲ uint32_t max_groups_x = ctx->device->properties.limits.max_Compute_Work_ਸਮੂਹ_ਗਿਣਤੀ[0];

    uint32_t groups_x = ne01;
    uint32_t groups_z = 1;

    ਜੇ (ne01 > max_groups_x) {
        groups_z = 64;
        groups_x = CEIL_DIV(groups_x, groups_z);
    }

    uint32_t fusion_flags = 0;

    ਜੇ (ctx->ਅੰਕ_additional_fused_ops > 0) {
        ਅਚਲ ggml_tensor * bias = cgraph->nodes[ਗੰਢ_idx + 1]->src[1];

        d_F0 = ggml_vk_tensor_subbuffer(ctx, bias);

        ਜੇ (cgraph->nodes[ਗੰਢ_idx + 1]->op == GGML_OP_MUL) {
            fusion_flags |= MAT_VEC_FUSION_FLAGS_SCALE0;
        } ਨਹੀਂਤਾਂ {
            GGML_ASSERT(cgraph->nodes[ਗੰਢ_idx + 1]->op == GGML_OP_ਜੋੜੋ_ID);
            fusion_flags |= MAT_VEC_FUSION_FLAGS_BIAS0;
        }
    }

    vk_subbuffer d_F1 = d_D;
    ਜੇ (ctx->ਅੰਕ_additional_fused_ops > 1) {
        ਅਚਲ ggml_tensor * scale = cgraph->nodes[ਗੰਢ_idx + 2]->src[1];

        d_F1 = ggml_vk_tensor_subbuffer(ctx, scale);
        fusion_flags |= MAT_VEC_FUSION_FLAGS_SCALE1;
    }

    // Loop over the batch dimension
    ਲਈ (uint32_t expert_i1 = 0; expert_i1 < nei1; ++expert_i1) {
        ਅਚਲ vk_mat_vec_id_ਧੱਕੋ_constants pc = {
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

    ਜੇ (x_non_contig) {
        ctx->prealloc_x_need_sync = ਸੱਚ;
    }
    ਜੇ (y_non_contig || quantize_y) {
        ctx->prealloc_y_need_sync = ਸੱਚ;
    }
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_vk_use_mul_mat_vec_id(ਅਚਲ ggml_tensor * dst) {
    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * ids = dst->src[2];
    ਵਾਪਸ ids->ne[1] <= 8 && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type));
}

ਤਾਰਕਿਕ ggml_vk_use_mul_mat_vec_id(ਅਚਲ ਬਣਤਰ ggml_cgraph * cgraph, ਪੂਰਨ ਗੰਢ_idx) {
    ਵਾਪਸ ggml_vk_use_mul_mat_vec_id(cgraph->nodes[ਗੰਢ_idx]);
}

ਖਾਲੀ ggml_vk_mul_mat_id(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ਬਣਤਰ ggml_cgraph * cgraph, ਪੂਰਨ ਗੰਢ_idx) {
    ggml_tensor * dst = cgraph->nodes[ਗੰਢ_idx];
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];
    ggml_tensor * src2 = dst->src[2];
    VK_LOG_DEBUG("ggml_vk_mul_mat_id(" << src0 << ", " << src1 << ", " << src2 << ", " << dst << ")");
    ਜੇ (ggml_vk_use_mul_mat_vec_id(cgraph, ਗੰਢ_idx)) {
        ggml_vk_mul_mat_vec_id_q_f16(ctx, subctx, cgraph, ਗੰਢ_idx);
    } ਨਹੀਂਤਾਂ {
        ggml_vk_mul_mat_id_q_f16(ctx, subctx, src0, src1, src2, dst);
    }
}

ਤਾਰਕਿਕ ggml_vk_flash_attn_scalar_shmem_support(ਅਚਲ vk_device& device, ਅਚਲ vk_fa_tuning_params& params, uint32_t hsk, uint32_t hsv, ਤਾਰਕਿਕ f32acc, ggml_type k_type, ggml_type v_type) {
    GGML_ਅਣਵਰਤਿਆ(f32acc);
    // Needs to be kept up to date on shader changes
    ਅਚਲ uint32_t wg_ਮਾਪ = params.workgroup_ਮਾਪ;
    ਅਚਲ uint32_t Br = params.ਖੰਡ_ਪੰਕਤੀਆਂ;
    ਅਚਲ uint32_t Bc = params.ਖੰਡ_ਖਾਨੇ_ਗਿਣਤੀ;

    // BF16 uses the fp32 shader (FLOAT_TYPE=float)
    ਅਚਲ uint32_t float_type_ਮਾਪ = (device->fp16 && k_type != GGML_TYPE_BF16) ? ਆਕਾਰ(ggml_fp16_t) : ਆਕਾਰ(ਦਸ਼ਮਲਵ);

    ਅਚਲ ਤਾਰਕਿਕ mmq = ggml_vk_fa_scalar_ਵਰਤਦਾ_mmq(device, k_type, v_type);

    // tmpsh is overestimated slightly
    ਅਚਲ uint32_t tmpsh = wg_ਮਾਪ * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ uint32_t tmpshv4 = wg_ਮਾਪ * 4 * float_type_ਮਾਪ;

    ਅਚਲ uint32_t masksh = Bc * (Br + 1) * float_type_ਮਾਪ;
    // DATA_A_IQ4_NL is compiled into the FA shaders unconditionally, so its shared table is always allocated.
    ਅਚਲ uint32_t iq_shmem = 16 * float_type_ਮਾਪ;

    uint32_t Qf, kvsh, kblocksh_ਮਾਪ;
    ਜੇ (mmq) {
        // block_b_cache: int32_t qs[8] + FLOAT_TYPEV2 ds
        ਅਚਲ uint32_t ਖੰਡ_b_ਮਾਪ = 8 * ਆਕਾਰ(int32_t) + 2 * float_type_ਮਾਪ;
        Qf = Br * (hsk / 32) * ਖੰਡ_b_ਮਾਪ;

        // kvsh uses D = HSV (K goes through kblocksh instead)
        kvsh = params.shmem_staging ? Bc * (hsv / 4 + 1) * 4 * float_type_ਮਾਪ : 4 * float_type_ਮਾਪ;

        // The mixed MMQ shader uses a superset block_a_cache that fits every
        // FA-supported quant: int32_t qs[8] + uint32_t qh + FLOAT_TYPEV2 dm.
        // Single-scale types leave dm.y unused; non-Q5_* leave qh unused.
        ਅਚਲ uint32_t ਖੰਡ_a_ਮਾਪ = 8 * ਆਕਾਰ(int32_t) + ਆਕਾਰ(uint32_t) + 2 * float_type_ਮਾਪ;
        kblocksh_ਮਾਪ = params.shmem_staging ? Bc * (hsk / 32) * ਖੰਡ_a_ਮਾਪ : ਖੰਡ_a_ਮਾਪ;
    } ਨਹੀਂਤਾਂ {
        Qf = Br * (hsk / 4 + 1) * 4 * float_type_ਮਾਪ;

        ਅਚਲ uint32_t D = std::max(hsk, hsv);
        kvsh = params.shmem_staging ? Bc * (D / 4 + 1) * 4 * float_type_ਮਾਪ : 4 * float_type_ਮਾਪ;

        kblocksh_ਮਾਪ = 0;
    }

    ਅਚਲ uint32_t ਕੁੱਲ_ਮਾਪ = tmpsh + tmpshv4 + masksh + iq_shmem + Qf + kvsh + kblocksh_ਮਾਪ;
    ਅਚਲ ਤਾਰਕਿਕ supported = ਕੁੱਲ_ਮਾਪ <= device->properties.limits.max_Compute_Shared_ਸਮ੍ਰਿਤੀ_ਮਾਪ;

    VK_LOG_DEBUG("ggml_vk_flash_attn_scalar_shmem_support(HSK=" << hsk << ", HSV=" << hsv << ", mmq=" << mmq << ", total_size=" << ਕੁੱਲ_ਮਾਪ << ", supported=" << supported);

    ਵਾਪਸ supported;
}

ਤਾਰਕਿਕ ggml_vk_flash_attn_coopmat_shmem_support(ਅਚਲ vk_device& device, ਅਚਲ vk_fa_tuning_params& params, uint32_t hsk, uint32_t hsv, ਤਾਰਕਿਕ f32acc, ggml_type k_type, ggml_type v_type) {
    GGML_ਅਣਵਰਤਿਆ(v_type);
    // Needs to be kept up to date on shader changes
    ਅਚਲ uint32_t Br = params.ਖੰਡ_ਪੰਕਤੀਆਂ;
    ਅਚਲ uint32_t Bc = params.ਖੰਡ_ਖਾਨੇ_ਗਿਣਤੀ;

    ਅਚਲ uint32_t MatBr = 16, MatBc = 16;

    ਅਚਲ uint32_t ਪੰਕਤੀ_ਵੰਡੋ = Bc / MatBc;

    ਅਚਲ uint32_t hsk_pad = ROUNDUP_POW2(hsk, 16);
    ਅਚਲ uint32_t hsv_pad = ROUNDUP_POW2(hsv, 16);

    ਅਚਲ uint32_t acctype = f32acc ? 4 : 2;
    ਅਚਲ uint32_t f16vec4 = 8;

    ਅਚਲ uint32_t tmpsh = (Bc / MatBc) * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    // DATA_A_IQ4_NL is compiled into the FA shaders unconditionally, so its shared table is always allocated.
    ਅਚਲ uint32_t iq_shmem = 16 * ਆਕਾਰ(ggml_fp16_t);

    ਅਚਲ uint32_t qstride = hsk_pad / 4 + 2;
    ਅਚਲ uint32_t Qf = Br * qstride * f16vec4;

    ਅਚਲ uint32_t psh_stride = Br / 4 + 2;
    ਅਚਲ uint32_t Psh = Bc * psh_stride * f16vec4;

    ਅਚਲ uint32_t sfshstride = (hsk <= 128) ? (Br + 8) : Br;
    ਅਚਲ uint32_t sfsh = Bc * sfshstride * acctype;

    ਅਚਲ uint32_t kvshstride = (params.shmem_staging ? std::max(hsk_pad, hsv_pad) : MatBr) / 4 + 2;
    ਅਚਲ uint32_t vsh_stride = MatBc / 4 * ਪੰਕਤੀ_ਵੰਡੋ;
    ਅਚਲ uint32_t ksh = ((kvshstride >= vsh_stride) ? (Bc * kvshstride) : (Bc * vsh_stride)) * f16vec4;

    // BF16 PVMat accumulator is f32 (no bf16 accumulator support), so pvsh is vec4 (16 bytes)
    ਅਚਲ uint32_t pvsh_elem_ਮਾਪ = (k_type == GGML_TYPE_BF16) ? 16u : f16vec4;
    ਅਚਲ uint32_t osh_stride = params.ਪੰਕਤੀ_ਵੰਡੋ * MatBr / 4;
    ਅਚਲ uint32_t pvsh = MatBc * osh_stride * pvsh_elem_ਮਾਪ;

    ਅਚਲ uint32_t ਢਲਾਨ = Br * acctype;

    ਅਚਲ uint32_t ਕੁੱਲ_ਮਾਪ = tmpsh + iq_shmem + Qf + Psh + sfsh + ksh + pvsh + ਢਲਾਨ;
    ਅਚਲ ਤਾਰਕਿਕ supported = ਕੁੱਲ_ਮਾਪ <= device->properties.limits.max_Compute_Shared_ਸਮ੍ਰਿਤੀ_ਮਾਪ;

    VK_LOG_DEBUG("ggml_vk_flash_attn_coopmat_shmem_support(HSK=" << hsk << ", HSV=" << hsv << ", f32acc=" << f32acc << ", total_size=" << ਕੁੱਲ_ਮਾਪ << ", supported=" << supported);

    ਵਾਪਸ supported;
}

ਖਾਲੀ ggml_vk_flash_attn(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * q, ਅਚਲ ggml_tensor * k, ਅਚਲ ggml_tensor * v, ਅਚਲ ggml_tensor * ਨਕਾਬ, ਅਚਲ ggml_tensor * sinks, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_vk_flash_attn((" << q << ", name=" << q->name << ", type=" << q->type << ", ne0=" << q->ne[0] << ", ne1=" << q->ne[1] << ", ne2=" << q->ne[2] << ", ne3=" << q->ne[3] << ", nb0=" << q->nb[0] << ", nb1=" << q->nb[1] << ", nb2=" << q->nb[2] << ", nb3=" << q->nb[3];
    std::cerr << "), (" << k << ", name=" << k->name << ", type=" << k->type << ", ne0=" << k->ne[0] << ", ne1=" << k->ne[1] << ", ne2=" << k->ne[2] << ", ne3=" << k->ne[3] << ", nb0=" << k->nb[0] << ", nb1=" << k->nb[1] << ", nb2=" << k->nb[2] << ", nb3=" << k->nb[3];
    std::cerr << "), (" << v << ", name=" << v->name << ", type=" << v->type << ", ne0=" << v->ne[0] << ", ne1=" << v->ne[1] << ", ne2=" << v->ne[2] << ", ne3=" << v->ne[3] << ", nb0=" << v->nb[0] << ", nb1=" << v->nb[1] << ", nb2=" << v->nb[2] << ", nb3=" << v->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    ਜੇ (sinks) {
        std::cerr << "), (" << sinks << ", name=" << sinks->name << ", type=" << sinks->type << ", ne0=" << sinks->ne[0] << ", ne1=" << sinks->ne[1] << ", ne2=" << sinks->ne[2] << ", ne3=" << sinks->ne[3] << ", nb0=" << sinks->nb[0] << ", nb1=" << sinks->nb[1] << ", nb2=" << sinks->nb[2] << ", nb3=" << sinks->nb[3];
    }
    std::cerr << "))");

    GGML_TENSOR_LOCALS(int64_t, neq, q,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbq, q,   nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbk, k,   nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbv, v,   nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst, nb)

    ਅਚਲ uint32_t nem0 = ਨਕਾਬ ? ਨਕਾਬ->ne[0] : 0;
    ਅਚਲ uint32_t nem1 = ਨਕਾਬ ? ਨਕਾਬ->ne[1] : 0;
    ਅਚਲ uint32_t nem2 = ਨਕਾਬ ? ਨਕਾਬ->ne[2] : 0;
    ਅਚਲ uint32_t nem3 = ਨਕਾਬ ? ਨਕਾਬ->ne[3] : 0;

    ਅਚਲ uint32_t HSK = nek0;
    ਅਚਲ uint32_t HSV = nev0;
    uint32_t N = neq1;
    ਅਚਲ uint32_t KV = nek1;

    GGML_ASSERT(ne0 == HSV);
    GGML_ASSERT(ne2 == N);

    // input tensor rows must be contiguous
    GGML_ASSERT(nbq0 == ggml_type_ਮਾਪ(q->type));
    GGML_ASSERT(nbk0 == ggml_type_ਮਾਪ(k->type));
    GGML_ASSERT(nbv0 == ggml_type_ਮਾਪ(v->type));

    GGML_ASSERT(neq0 == HSK);

    GGML_ASSERT(neq1 == N);

    GGML_ASSERT(nev1 == nek1);

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    ਪ੍ਰਤਿਜ੍ਞਾ(dst->type == GGML_TYPE_F32);
    ਪ੍ਰਤਿਜ੍ਞਾ(q->type == GGML_TYPE_F32);
    uint32_t gqa_ਅਨੁਪਾਤ = 1;
    uint32_t qk_ਅਨੁਪਾਤ = neq2 / nek2;
    uint32_t workgroups_x = (uint32_t)neq1;
    uint32_t workgroups_y = (uint32_t)neq2;
    uint32_t workgroups_z = (uint32_t)neq3;

    ਅਚਲ ਤਾਰਕਿਕ f32acc = !ctx->device->fp16 || dst->op_params[3] == GGML_PREC_F32 || k->type == GGML_TYPE_BF16;

    // dequant K/V once into an f16 scratch, reordered KV layout so FA can read without a stride
    ਸਵੈਚਾਲਿਤ is_dense_kv_cache = [](ਅਚਲ ggml_tensor * t) {
        ਵਾਪਸ t->nb[0] == ggml_type_ਮਾਪ(t->type) &&
               t->nb[2] == ggml_ਪੰਕਤੀ_ਮਾਪ(t->type, t->ne[0]) &&
               t->nb[1] == t->nb[2] * t->ne[2] &&
               (t->ne[3] == 1 || t->nb[3] == t->nb[1] * t->ne[1]);
    };
    ਅਚਲ ਤਾਰਕਿਕ k_quant = k->type != GGML_TYPE_F16 && k->type != GGML_TYPE_BF16 && k->type != GGML_TYPE_F32;
    ਅਚਲ ਤਾਰਕਿਕ v_quant = v->type != GGML_TYPE_F16 && v->type != GGML_TYPE_BF16 && v->type != GGML_TYPE_F32;
    ਅਚਲ ਤਾਰਕਿਕ kv_raw = (k->type == GGML_TYPE_F16 || k->type == GGML_TYPE_BF16) && k->type == v->type;
    // gated to AMD, except RDNA4, which regressed in benchmarks
    ਅਚਲ ਤਾਰਕਿਕ kv_raw_strided = ctx->device->vendor_id == VK_VENDOR_ID_AMD &&
                                ctx->device->architecture != vk_device_architecture::AMD_RDNA4 &&
                                neq1 >= 256 &&
                                kv_raw &&
                                (k->nb[1] != (uint64_t)HSK * ggml_type_ਮਾਪ(k->type) ||
                                 v->nb[1] != (uint64_t)HSV * ggml_type_ਮਾਪ(v->type)) &&
                                (HSK % 8) == 0 && (HSV % 8) == 0;
    ਅਚਲ size_t kv_scratch_ts = kv_raw_strided ? ggml_type_ਮਾਪ(k->type) : ਆਕਾਰ(ggml_fp16_t);
    ਅਚਲ ਤਾਰਕਿਕ use_dequant_kv = ((k_quant && v_quant) || kv_raw_strided) && neq1 >= 64 &&
                                is_dense_kv_cache(k) && is_dense_kv_cache(v) &&
                                (uint64_t)ggml_nelements(k) * kv_scratch_ts <= ctx->device->properties.limits.max_Storage_ਬਫਰ_Range &&
                                (uint64_t)ggml_nelements(v) * kv_scratch_ts <= ctx->device->properties.limits.max_Storage_ਬਫਰ_Range &&
                                (kv_raw_strided ||
                                 (ctx->device->pipeline_dequant_transpose[k->type] != ਖਾਲੀ_ਸੂਚਕ &&
                                  ctx->device->pipeline_dequant_transpose[v->type] != ਖਾਲੀ_ਸੂਚਕ)) &&
                                // coopmat2 path does not benefit from the f16 scratch
                                !ctx->device->coopmat2 &&
                                // Intel Xe1 regresses, see PR 25494
                                (ctx->device->vendor_id != VK_VENDOR_ID_INTEL ||
                                 (ctx->device->coopmat_support && ctx->device->architecture != vk_device_architecture::INTEL_XE1));
    ਅਚਲ ggml_type k_type_eff = (use_dequant_kv && !kv_raw_strided) ? GGML_TYPE_F16 : k->type;
    ਅਚਲ ggml_type v_type_eff = (use_dequant_kv && !kv_raw_strided) ? GGML_TYPE_F16 : v->type;

    // For scalar/coopmat1 FA, we can use the "large" size to accommodate qga.
    // For coopmat2 FA, we always use the small size (which is still pretty large for gqa).
    vk_fa_tuning_params tuning_params = get_fa_tuning_params(ctx->device, HSK, HSV, 512, KV, k_type_eff, v_type_eff, f32acc);
    ਅਚਲ uint32_t max_gqa = std::min(tuning_params.ਖੰਡ_ਪੰਕਤੀਆਂ, 32u);

    ਜੇ (N <= 8 && qk_ਅਨੁਪਾਤ > 1 && qk_ਅਨੁਪਾਤ <= max_gqa &&
        qk_ਅਨੁਪਾਤ * nek2 == neq2 && nek2 == nev2 && nem2 <= 1) {
        // grouped query attention - make the N dimension equal to gqa_ratio, reduce
        // workgroups proportionally in y dimension. The shader will detect gqa_ratio > 1
        // and change addressing calculations to index Q's dimension 2.
        gqa_ਅਨੁਪਾਤ = qk_ਅਨੁਪਾਤ;
        N = gqa_ਅਨੁਪਾਤ;
        workgroups_y /= gqa_ਅਨੁਪਾਤ;
    }

    tuning_params = get_fa_tuning_params(ctx->device, HSK, HSV, N, KV, k_type_eff, v_type_eff, f32acc);

    ਦਸ਼ਮਲਵ scale         = 1.0f;
    ਦਸ਼ਮਲਵ max_bias      = 0.0f;
    ਦਸ਼ਮਲਵ logit_softcap = 0.0f;

    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&scale,         (ਅਚਲ ਦਸ਼ਮਲਵ *) dst->op_params + 0, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&max_bias,      (ਅਚਲ ਦਸ਼ਮਲਵ *) dst->op_params + 1, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&logit_softcap, (ਅਚਲ ਦਸ਼ਮਲਵ *) dst->op_params + 2, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਜੇ (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    // Sparse mask hint (op_params[4]): compact the <= n_kv_max finite positions and gather only those.
    ਅਚਲ int32_t n_kv_max = ਨਕਾਬ ? ggml_get_op_params_i32(dst, 4) : 0;
    ਸਥਿਰ ਅਚਲ ਤਾਰਕਿਕ disable_sparse = getenv("GGML_VK_FA_SPARSE_DISABLE") != ਖਾਲੀ_ਸੂਚਕ;
    ਅਚਲ ਤਾਰਕਿਕ kv_f16 = k_type_eff == GGML_TYPE_F16 && v_type_eff == GGML_TYPE_F16;
    // cm2 dense is fast, so it needs a larger reduction to win.
    // With quantized K/V, sparse only breaks even around 16x (measured on RDNA3/RDNA4).
    ਅਚਲ int64_t min_ਅਨੁਪਾਤ = tuning_params.path == FA_COOPMAT2 ? 4 : (kv_f16 ? 2 : 16);
    // coopmat2 vector decode requires 8B strides.
    ਸਵੈਚਾਲਿਤ sparse_gather_aligned = [](ਅਚਲ ggml_tensor * t) {
        ਵਾਪਸ (t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_BF16) ||
               (t->nb[1] | t->nb[2] | t->nb[3]) % (4 * ਆਕਾਰ(ggml_fp16_t)) == 0;
    };
    ਅਚਲ ਤਾਰਕਿਕ use_sparse = !disable_sparse && n_kv_max > 0 && ਨਕਾਬ &&
                            max_bias == 0.0f && logit_softcap == 0.0f &&
                            (tuning_params.path != FA_COOPMAT2 || (sparse_gather_aligned(k) && sparse_gather_aligned(v))) &&
                            nem0 == KV &&
                            (int64_t)KV >= std::max<int64_t>(4096, min_ਅਨੁਪਾਤ * (int64_t)n_kv_max) &&
                            (gqa_ਅਨੁਪਾਤ > 1 || (tuning_params.path == FA_SCALAR && N == 1));

    ਅਚਲ uint32_t q_stride = (uint32_t)(nbq1 / ggml_type_ਮਾਪ(q->type));
    uint32_t k_stride = (uint32_t)(nbk1 / ggml_type_ਮਾਪ(k->type));
    uint32_t v_stride = (uint32_t)(nbv1 / ggml_type_ਮਾਪ(v->type));

    // For F32, the shader treats it as a block of size 4 (for vec4 loads)
    ਜੇ (k->type == GGML_TYPE_F32) {
        k_stride /= 4;
    }
    ਜੇ (v->type == GGML_TYPE_F32) {
        v_stride /= 4;
    }

    uint32_t nbk2_eff = (uint32_t)nbk2, nbk3_eff = (uint32_t)nbk3;
    uint32_t nbv2_eff = (uint32_t)nbv2, nbv3_eff = (uint32_t)nbv3;
    ਜੇ (use_dequant_kv) {
        k_stride = HSK;
        v_stride = HSV;
        nbk2_eff = (uint32_t)((uint64_t)HSK * KV * ਆਕਾਰ(ggml_fp16_t));
        nbk3_eff = (uint32_t)((uint64_t)HSK * KV * nek2 * ਆਕਾਰ(ggml_fp16_t));
        nbv2_eff = (uint32_t)((uint64_t)HSV * KV * ਆਕਾਰ(ggml_fp16_t));
        nbv3_eff = (uint32_t)((uint64_t)HSV * KV * nev2 * ਆਕਾਰ(ggml_fp16_t));
    }
    ਅਚਲ uint32_t alignment = tuning_params.ਖੰਡ_ਖਾਨੇ_ਗਿਣਤੀ;
    ਤਾਰਕਿਕ aligned = (KV % alignment) == 0 &&
                   // the "aligned" shader variant will forcibly align strides, for performance
                   (q_stride & 7) == 0 && (k_stride & 7) == 0 && (v_stride & 7) == 0;

    // Need to use the coopmat2 variant that clamps loads when HSK/HSV aren't sufficiently aligned.
    ਜੇ (((HSK | HSV) % 16) != 0 && tuning_params.path == FA_COOPMAT2) {
        aligned = ਝੂਠ;
    }

    // Only use mask opt when the mask is fairly large. This hasn't been tuned extensively.
    ਤਾਰਕਿਕ use_ਨਕਾਬ_opt = ਨਕਾਬ && !use_sparse && nem1 >= 32 && nem0 * nem1 > 32768 && nem0 >= tuning_params.ਖੰਡ_ਖਾਨੇ_ਗਿਣਤੀ * 16
                        && (ctx->device->architecture != vk_device_architecture::AMD_GCN || HSK > 256 || HSV > 256);
    vk_fa_pipeline_state fa_pipeline_state = get_fa_pipeline_state(ctx->device, tuning_params, HSK, HSV, aligned, f32acc,
                                                                   ਨਕਾਬ != ਖਾਲੀ_ਸੂਚਕ, use_ਨਕਾਬ_opt, logit_softcap != 0, use_sparse, k_type_eff, v_type_eff);

    vk_pipeline pipeline = ਖਾਲੀ_ਸੂਚਕ;

    ਤਾਰਕਿਕ xe_fa_opt = ਝੂਠ;
    ਤਾਰਕਿਕ fa_copy_qstate = ਝੂਠ;
    ਤਾਰਕਿਕ xe_fa_supported_platform =
        (ctx->device.get()->architecture == INTEL_XE2 && ctx->device.get()->properties.deviceID != 0xFD80 && ctx->device.get()->properties.deviceID != 0xFD81) ||
        (ctx->device.get()->architecture == INTEL_XE1 && ctx->device.get()->coopmat_support && ctx->device.get()->uma);
    ਤਾਰਕਿਕ xe_fa_supported_usage = neq0 % 32 == 0 && nev0 % 16 == 0 && q->nb[1] > q->nb[2] && k->nb[1] > k->nb[2] && v->nb[1] > v->nb[2] && ਨਕਾਬ != ਖਾਲੀ_ਸੂਚਕ;
    ਤਾਰਕਿਕ xe_fa_supported_dtype = q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16 && (ਨਕਾਬ != ਖਾਲੀ_ਸੂਚਕ && ਨਕਾਬ->type == GGML_TYPE_F16);
    std::pair<vk_pipeline, vk_pipeline> xe_fa_pipeline_dual_phases = { ਖਾਲੀ_ਸੂਚਕ , ਖਾਲੀ_ਸੂਚਕ };
    vk_pipeline xe_fa_pipeline = ਖਾਲੀ_ਸੂਚਕ;
    size_t ਮਾਪ_p = 0;
    size_t ਮਾਪ_ਸਮੂਹ_max = 0;

    {
        std::lock_guard<std::mutex> guard(ctx->device->compile_mutex);
        ਸਵੈਚਾਲਿਤ &pipelines = ctx->device->pipeline_flash_attn_f32_f16;
        ਸਵੈਚਾਲਿਤ it = pipelines.find(fa_pipeline_state);
        ਜੇ (it != pipelines.end()) {
            pipeline = it->second;
        } ਨਹੀਂਤਾਂ {
            pipelines[fa_pipeline_state] = pipeline = std::make_shared<vk_pipeline_struct>();
        }
    }

    ਪ੍ਰਤਿਜ੍ਞਾ(pipeline);
    // Compile early to initialize wg_denoms.
    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    uint32_t ਵੰਡੋ_kv = KV;
    uint32_t ਵੰਡੋ_k = 1;

    // Intel Alchemist prefers more workgroups
    ਅਚਲ uint32_t shader_core_ਗਿਣਤੀ_multiplier = (ctx->device->vendor_id == VK_VENDOR_ID_INTEL && ctx->device->architecture != INTEL_XE2) ? 2 : 1;

    // Use a placeholder core count if one isn't available. split_k is a big help for perf.
    ਅਚਲ uint32_t shader_core_ਗਿਣਤੀ = ctx->device->shader_core_ਗਿਣਤੀ ? ctx->device->shader_core_ਗਿਣਤੀ * shader_core_ਗਿਣਤੀ_multiplier : 16;

    ਅਚਲ uint32_t Br = fa_pipeline_state.Br;
    ਅਚਲ uint32_t Bc = fa_pipeline_state.Bc;

    GGML_ASSERT(Br == pipeline->wg_denoms[0]);
    ਅਚਲ uint32_t Tr = CEIL_DIV(N, Br);

    // Try to use split_k when KV is large enough to be worth the overhead.
    // Sparse: split_kv carries n_kv_max, split_k partitions its blocks for occupancy.
    ਜੇ (use_sparse) {
        ਵੰਡੋ_kv = (uint32_t)n_kv_max;
        ਅਚਲ uint32_t ਕੁੱਲ_ਖੰਡਾਂ = CEIL_DIV((uint32_t)n_kv_max, Bc);
        ਅਚਲ uint32_t ਅਧਾਰ_wgs = (gqa_ਅਨੁਪਾਤ > 1 ? workgroups_x : Tr) * workgroups_y * workgroups_z;
        ਜੇ (ਅਧਾਰ_wgs < shader_core_ਗਿਣਤੀ * 2) {
            ਵੰਡੋ_k = shader_core_ਗਿਣਤੀ * 2 / ਅਧਾਰ_wgs;
        }
        ਵੰਡੋ_k = std::max(1u, std::min(ਵੰਡੋ_k, ਕੁੱਲ_ਖੰਡਾਂ));
        // Match the shader's per-split block count so no split is empty.
        ਅਚਲ uint32_t ਪ੍ਰਤੀ_ਖੰਡਾਂ = CEIL_DIV(ਕੁੱਲ_ਖੰਡਾਂ, ਵੰਡੋ_k);
        ਵੰਡੋ_k = CEIL_DIV(ਕੁੱਲ_ਖੰਡਾਂ, ਪ੍ਰਤੀ_ਖੰਡਾਂ);
    } ਨਹੀਂਤਾਂ ਜੇ (gqa_ਅਨੁਪਾਤ > 1 && workgroups_x <= Br) {
        ਵੰਡੋ_k = shader_core_ਗਿਣਤੀ * 2 / (workgroups_x * workgroups_y * workgroups_z);
    } ਨਹੀਂਤਾਂ ਜੇ (gqa_ਅਨੁਪਾਤ <= 1) {
        uint32_t ਕੁੱਲ_wgs_no_ਵੰਡੋ = Tr * workgroups_y * workgroups_z;
        ਜੇ (ਕੁੱਲ_wgs_no_ਵੰਡੋ < shader_core_ਗਿਣਤੀ * 2) {
            ਵੰਡੋ_k = shader_core_ਗਿਣਤੀ * 2 / ਕੁੱਲ_wgs_no_ਵੰਡੋ;
        }
    }

    ਜੇ (!use_sparse && ਵੰਡੋ_k > 1) {
        // Try to evenly split KV into split_k chunks, but it needs to be a multiple
        // of "align", so recompute split_k based on that.
        ਵੰਡੋ_kv = ROUNDUP_POW2(std::max(1u, KV / ਵੰਡੋ_k), alignment);
        ਵੰਡੋ_k = CEIL_DIV(KV, ਵੰਡੋ_kv);
        xe_fa_opt = xe_fa_supported_platform && xe_fa_supported_usage && xe_fa_supported_dtype;
        ਜੇ (xe_fa_opt) {
            ਅਚਲ uint32_t ਵੰਡੋ_p_ਮਾਪ = 32;
            ਅਚਲ size_t max_dim = (nek1 + ਵੰਡੋ_p_ਮਾਪ - 1) / ਵੰਡੋ_p_ਮਾਪ;
            ਅਚਲ size_t p_dim = max_dim * ਵੰਡੋ_p_ਮਾਪ;
#ਜੇ defined(VK_KHR_cooperative_ਮੈਟ੍ਰਿਕਸ) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
            std::lock_guard<std::mutex> guard(ctx->device->compile_mutex);
            ਸਵੈਚਾਲਿਤ& pipelines = ctx->device->pipeline_xe_fa_decode_dual_phases;
            ਸਵੈਚਾਲਿਤ it = pipelines.find({ (uint32_t)neq0, (uint32_t)nev0, qk_ਅਨੁਪਾਤ, (uint32_t)neq1 });
            ਜੇ (it != pipelines.end()) {
                xe_fa_pipeline_dual_phases = it->second;
            } ਨਹੀਂਤਾਂ {
                pipelines[{(uint32_t)neq0, (uint32_t)nev0, qk_ਅਨੁਪਾਤ, (uint32_t)neq1}] = xe_fa_pipeline_dual_phases = std::make_pair(std::make_shared<vk_pipeline_struct>(), std::make_shared<vk_pipeline_struct>());
            }
#ਜੇ_ਅੰਤ
            ਜੇ (xe_fa_pipeline_dual_phases.ਪਹਿਲਾ == ਖਾਲੀ_ਸੂਚਕ || xe_fa_pipeline_dual_phases.second == ਖਾਲੀ_ਸੂਚਕ) {
                xe_fa_opt = ਝੂਠ;
                fa_copy_qstate = ਝੂਠ;
            } ਨਹੀਂਤਾਂ {
                ਮਾਪ_p = neq1 * neq2 * p_dim * neq3 * ਆਕਾਰ(ggml_fp16_t);
                ਮਾਪ_ਸਮੂਹ_max = neq1 * neq2 * max_dim * neq3 * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
                size_t ਆਰਜ਼ੀ_ਮਾਪ = ggml_nelements(q) * ਆਕਾਰ(ggml_fp16_t) + ਮਾਪ_p + ਮਾਪ_ਸਮੂਹ_max;
                fa_copy_qstate = ਸੱਚ;
                ਜੇ (ctx->prealloc_ਮਾਪ_x < ਆਰਜ਼ੀ_ਮਾਪ) {
                    ctx->prealloc_ਮਾਪ_x = ਆਰਜ਼ੀ_ਮਾਪ;
                    ggml_vk_preallocate_buffers(ctx, subctx);
                }

                ਜੇ (ctx->prealloc_x_need_sync) {
                    ggml_vk_sync_buffers(ctx, subctx);
                }
            }
        }
    }

    ਜੇ (xe_fa_opt == ਸੱਚ) {
        use_ਨਕਾਬ_opt = ਝੂਠ;
    }

    // Reserve space for split_k temporaries. For each split x batch, we need to store the O matrix (D x ne1)
    // and the per-row m and L values (ne1 rows). We store all the matrices first, followed by the rows.
    // For matrices, the order is (inner to outer) [HSV, ne1, k, ne2, ne3].
    // For L/M, the order is (inner to outer) [ne1, k, ne2, ne3].
    ਅਚਲ uint64_t ਵੰਡੋ_k_ਮਾਪ = ਵੰਡੋ_k > 1 ? (HSV * ne1 * ਆਕਾਰ(ਦਸ਼ਮਲਵ) + ne1 * ਆਕਾਰ(ਦਸ਼ਮਲਵ) * 2) * ਵੰਡੋ_k * ne2 * ne3 : 0;
    ਜੇ (ਵੰਡੋ_k_ਮਾਪ > ctx->device->properties.limits.max_Storage_ਬਫਰ_Range) {
        GGML_ABORT("Requested preallocation size is too large");
    }
    ਜੇ (ctx->prealloc_ਮਾਪ_ਵੰਡੋ_k < ਵੰਡੋ_k_ਮਾਪ) {
        ctx->prealloc_ਮਾਪ_ਵੰਡੋ_k = ਵੰਡੋ_k_ਮਾਪ;
        ggml_vk_preallocate_buffers(ctx, subctx);
    }

    ਅਚਲ uint32_t ਨਕਾਬ_opt_ਅੰਕ_dwords = CEIL_DIV(nem0, 16 * Bc);
    ਅਚਲ uint64_t ਨਕਾਬ_opt_ਮਾਪ = ਆਕਾਰ(uint32_t) * ਨਕਾਬ_opt_ਅੰਕ_dwords * CEIL_DIV(nem1, Br) * nem2 * nem3;

    vk_pipeline pipeline_fa_ਨਕਾਬ_opt = ਖਾਲੀ_ਸੂਚਕ;
    ਜੇ (use_ਨਕਾਬ_opt) {
        {
            std::lock_guard<std::mutex> guard(ctx->device->compile_mutex);
            ਸਵੈਚਾਲਿਤ &pipelines = ctx->device->pipeline_fa_ਨਕਾਬ_opt;
            ਸਵੈਚਾਲਿਤ it = pipelines.find({Br, Bc});
            ਜੇ (it != pipelines.end()) {
                pipeline_fa_ਨਕਾਬ_opt = it->second;
            } ਨਹੀਂਤਾਂ {
                pipelines[{Br, Bc}] = pipeline_fa_ਨਕਾਬ_opt = std::make_shared<vk_pipeline_struct>();
            }
        }
        ਪ੍ਰਤਿਜ੍ਞਾ(pipeline_fa_ਨਕਾਬ_opt);
        ggml_pipeline_request_descriptor_sets(ctx, pipeline_fa_ਨਕਾਬ_opt, 1);

        ਜੇ (ctx->prealloc_ਮਾਪ_y < ਨਕਾਬ_opt_ਮਾਪ) {
            ctx->prealloc_ਮਾਪ_y = ਨਕਾਬ_opt_ਮਾਪ;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        ਜੇ (ctx->prealloc_y_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }
    }

    // Sparse index scratch reuses prealloc_y (mutually exclusive with mask opt).
    ਅਚਲ uint64_t sparse_idx_ਮਾਪ = use_sparse
        ? ਆਕਾਰ(int32_t) * (uint64_t)n_kv_max * nem1 * nem2 * nem3
        : 0;
    vk_pipeline sparse_compact_pipeline = ctx->device->fa_sparse_compact_use_subgroups
        ? ctx->device->pipeline_fa_sparse_compact_subgroup
        : ctx->device->pipeline_fa_sparse_compact;
    ਜੇ (use_sparse) {
        ggml_pipeline_request_descriptor_sets(ctx, sparse_compact_pipeline, 1);
        ਜੇ (ctx->prealloc_ਮਾਪ_y < sparse_idx_ਮਾਪ) {
            ctx->prealloc_ਮਾਪ_y = sparse_idx_ਮਾਪ;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        ਜੇ (ctx->prealloc_y_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }
    }

    ਅਚਲ uint32_t n_ਸਿਰ_kv   = neq2;
    ਅਚਲ uint32_t n_ਸਿਰ_log2 = 1u << (uint32_t) floorf(log2f((ਦਸ਼ਮਲਵ) n_ਸਿਰ_kv));
    ਅਚਲ ਦਸ਼ਮਲਵ m0 = powf(2.0f, -(max_bias       ) / n_ਸਿਰ_log2);
    ਅਚਲ ਦਸ਼ਮਲਵ m1 = powf(2.0f, -(max_bias / 2.0f) / n_ਸਿਰ_log2);

    vk_subbuffer q_buf = ggml_vk_tensor_subbuffer(ctx, q);
    vk_subbuffer k_buf = ggml_vk_tensor_subbuffer(ctx, k);
    vk_subbuffer v_buf = ggml_vk_tensor_subbuffer(ctx, v);
    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
    vk_subbuffer ਨਕਾਬ_buf = ਨਕਾਬ ? ggml_vk_tensor_subbuffer(ctx, ਨਕਾਬ) : q_buf;
    vk_subbuffer sinks_buf = sinks ? ggml_vk_tensor_subbuffer(ctx, sinks) : q_buf;
    vk_subbuffer ਨਕਾਬ_opt_buf = use_ਨਕਾਬ_opt ? ggml_vk_subbuffer(ctx, ctx->prealloc_y, 0) : q_buf;
    vk_subbuffer sparse_buf = use_sparse ? ggml_vk_subbuffer(ctx, ctx->prealloc_y, 0) : q_buf;
    ਜੇ (use_ਨਕਾਬ_opt || use_sparse) {
        // the mask opt bits and the sparse index list overwrite a matmul input converted into prealloc_y
        ctx->prealloc_y_ਆਖਰੀ_pipeline_ਵਰਤਿਆ = ਖਾਲੀ_ਸੂਚਕ;
        ctx->prealloc_y_ਆਖਰੀ_tensor_ਵਰਤਿਆ = ਖਾਲੀ_ਸੂਚਕ;
    }

    ਜੇ (use_dequant_kv) {
        ਅਚਲ uint64_t k_sz = (uint64_t)ggml_nelements(k) * kv_scratch_ts;
        ਅਚਲ uint64_t v_sz = (uint64_t)ggml_nelements(v) * kv_scratch_ts;
        ਜੇ (ctx->prealloc_ਮਾਪ_x < k_sz + v_sz) {
            ctx->prealloc_ਮਾਪ_x = k_sz + v_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        vk_subbuffer k_dst = vk_subbuffer{ ctx->prealloc_x, 0,    k_sz };
        vk_subbuffer v_dst = vk_subbuffer{ ctx->prealloc_x, k_sz, v_sz };
        ਅਚਲ uint32_t k_nel = (uint32_t)ggml_nelements(k);
        ਅਚਲ uint32_t v_nel = (uint32_t)ggml_nelements(v);
        ਜੇ (kv_raw_strided) {
            vk_pipeline cp_k = ggml_vk_get_cpy_pipeline(ctx, k, ਖਾਲੀ_ਸੂਚਕ, k->type);
            vk_pipeline cp_v = ggml_vk_get_cpy_pipeline(ctx, v, ਖਾਲੀ_ਸੂਚਕ, v->type);
            ggml_pipeline_request_descriptor_sets(ctx, cp_k, 1);
            ggml_pipeline_request_descriptor_sets(ctx, cp_v, 1);
            ਜੇ (ctx->prealloc_x_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ਸਵੈਚਾਲਿਤ make_pc = [](uint32_t hs, uint32_t nh, uint32_t kv, uint32_t nel) {
                vk_op_unary_ਧੱਕੋ_constants pc{};
                pc.ne = nel;
                pc.ne00 = hs; pc.ne01 = nh;      pc.ne02 = kv; pc.ne03 = nel / (hs * nh * kv);
                pc.nb00 = 1;  pc.nb01 = hs;      pc.nb02 = hs * nh; pc.nb03 = hs * nh * kv;
                pc.ne10 = hs; pc.ne11 = nh;      pc.ne12 = kv; pc.ne13 = pc.ne03;
                pc.nb10 = 1;  pc.nb11 = hs * kv; pc.nb12 = hs; pc.nb13 = hs * kv * nh;
                init_pushconst_fastdiv(pc);
                ਵਾਪਸ pc;
            };
            ਸਵੈਚਾਲਿਤ cpy_elems = [](uint32_t ne) -> std::array<uint32_t, 3> {
                ਜੇ (ne > 262144) { ਵਾਪਸ { 512, 512, CEIL_DIV(ne, 262144) }; }
                ਜੇ (ne > 512)    { ਵਾਪਸ { 512, CEIL_DIV(ne, 512), 1 }; }
                ਵਾਪਸ { ne, 1, 1 };
            };
            ggml_vk_dispatch_pipeline(ctx, subctx, cp_k, { k_buf, k_dst },
                make_pc((uint32_t)HSK, (uint32_t)nek2, (uint32_t)KV, k_nel), cpy_elems(k_nel));
            ggml_vk_dispatch_pipeline(ctx, subctx, cp_v, { v_buf, v_dst },
                make_pc((uint32_t)HSV, (uint32_t)nev2, (uint32_t)KV, v_nel), cpy_elems(v_nel));
        } ਨਹੀਂਤਾਂ {
            vk_pipeline tr_k = ctx->device->pipeline_dequant_transpose[k->type];
            vk_pipeline tr_v = ctx->device->pipeline_dequant_transpose[v->type];
            ggml_pipeline_request_descriptor_sets(ctx, tr_k, 1);
            ggml_pipeline_request_descriptor_sets(ctx, tr_v, 1);
            ਜੇ (ctx->prealloc_x_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            { ਅਚਲ std::vector<uint32_t> pc = { (uint32_t)HSK, (uint32_t)nek2, (uint32_t)KV, 0, k_nel };
              ggml_vk_dispatch_pipeline(ctx, subctx, tr_k, { k_buf, k_dst }, pc, { k_nel, 1, 1 }); }
            { ਅਚਲ std::vector<uint32_t> pc = { (uint32_t)HSV, (uint32_t)nev2, (uint32_t)KV, 0, v_nel };
              ggml_vk_dispatch_pipeline(ctx, subctx, tr_v, { v_buf, v_dst }, pc, { v_nel, 1, 1 }); }
        }
        ggml_vk_sync_buffers(ctx, subctx);
        k_buf = k_dst;
        v_buf = v_dst;
    }

    uint32_t ਨਕਾਬ_n_ਸਿਰ_log2 = ((sinks != ਖਾਲੀ_ਸੂਚਕ) << 24) | n_ਸਿਰ_log2;

    ਜੇ (use_ਨਕਾਬ_opt)
    {
        ਅਚਲ vk_op_flash_attn_ਨਕਾਬ_opt_ਧੱਕੋ_constants opt_pc = {
            nem0,
            nem1,
            nem2,
            (uint32_t)(ਨਕਾਬ->nb[1] / ਆਕਾਰ(ggml_fp16_t)),
            (uint32_t)(ਨਕਾਬ->nb[2] / ਆਕਾਰ(ggml_fp16_t)),
            (uint32_t)(ਨਕਾਬ->nb[3] / ਆਕਾਰ(ggml_fp16_t)),
            ਨਕਾਬ_opt_ਅੰਕ_dwords,
            ਨਕਾਬ_opt_ਅੰਕ_dwords * CEIL_DIV(nem1, Br),
            ਨਕਾਬ_opt_ਅੰਕ_dwords * CEIL_DIV(nem1, Br) * nem2,
        };

        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline_fa_ਨਕਾਬ_opt,
                                  { ਨਕਾਬ_buf, ਨਕਾਬ_opt_buf }, opt_pc,
                                  { ਨਕਾਬ_opt_ਅੰਕ_dwords, CEIL_DIV(nem1, Br), nem2 * nem3 });
        ggml_vk_sync_buffers(ctx, subctx);
    }

    ਜੇ (use_sparse)
    {
        ਅਚਲ vk_op_flash_attn_sparse_compact_ਧੱਕੋ_constants sc_pc = {
            KV,
            nem1,
            nem2,
            (uint32_t)(ਨਕਾਬ->nb[1] / ਆਕਾਰ(ggml_fp16_t)),
            (uint32_t)(ਨਕਾਬ->nb[2] / ਆਕਾਰ(ggml_fp16_t)),
            (uint32_t)(ਨਕਾਬ->nb[3] / ਆਕਾਰ(ggml_fp16_t)),
            (uint32_t)n_kv_max,
        };

        ggml_vk_dispatch_pipeline(ctx, subctx, sparse_compact_pipeline,
                                  { ਨਕਾਬ_buf, sparse_buf }, sc_pc,
                                  { nem1, nem2, nem3 });
        ggml_vk_sync_buffers(ctx, subctx);
    }

    ਅਚਲ vk_flash_attn_ਧੱਕੋ_constants pc = { N, KV,
                                              (uint32_t)ne1, (uint32_t)ne2, (uint32_t)ne3,
                                              (uint32_t)neq2, (uint32_t)neq3,
                                              (uint32_t)nek2, (uint32_t)nek3,
                                              (uint32_t)nev2, (uint32_t)nev3,
                                              nem1, nem2, nem3,
                                              q_stride, (uint32_t)nbq2, (uint32_t)nbq3,
                                              k_stride, nbk2_eff, nbk3_eff,
                                              v_stride, nbv2_eff, nbv3_eff,
                                              scale, max_bias, logit_softcap,
                                              ਨਕਾਬ_n_ਸਿਰ_log2, m0, m1,
                                              gqa_ਅਨੁਪਾਤ, ਵੰਡੋ_kv, ਵੰਡੋ_k };

    ਜੇ (xe_fa_opt && ਵੰਡੋ_k > 1) {
        ਸਵੈਚਾਲਿਤ ਉੱਤਲਾ_power_ਦਾ_2 = [&](uint32_t ਵਿੱਚ) {
            GGML_ASSERT(ਵਿੱਚ != 0);
            ਜੇ (ਵਿੱਚ <= 1) ਵਾਪਸ 1u;
            uint32_t ret = ਵਿੱਚ - 1;
            ret |= ret >> 1;
            ret |= ret >> 2;
            ret |= ret >> 4;
            ret |= ret >> 8;
            ret |= ret >> 16;
            ਵਾਪਸ ret + 1;
        };
        ਸਵੈਚਾਲਿਤ ਮੰਜ਼ਿਲ_fp16_vk_0 = ggml_vk_get_ਮੰਜ਼ਿਲ_fp16(ctx, q->type);
        ਅਚਲ uint32_t ਬਾਹਰਲਾ_dim_ਪ੍ਰਤੀ_wg = qk_ਅਨੁਪਾਤ > 16 ? 8 : 16;
        size_t x_ne = ggml_nelements(q);
        size_t ਆਰਜ਼ੀ_buf_ਵਿੱਥ = 0;
        uint32_t ਸਿਰ_stride_k = uint32_t(nbk2 / ggml_type_ਮਾਪ(k->type));
        uint32_t ਸਿਰ_stride_v = uint32_t(nbv2 / ggml_type_ਮਾਪ(v->type));
        uint32_t batch_stride_q = uint32_t(nbq3 / ggml_type_ਮਾਪ(q->type));
        uint32_t batch_stride_k = uint32_t(nbk3 / ggml_type_ਮਾਪ(k->type));
        uint32_t batch_stride_v = uint32_t(nbv3 / ggml_type_ਮਾਪ(v->type));
        uint32_t batch_stride_m = ਨਕਾਬ ? uint32_t(ਨਕਾਬ->nb[3] / ggml_type_ਮਾਪ(ਨਕਾਬ->type)) : 0u;
        uint32_t batch_stride_o = uint32_t(nb3 / ggml_type_ਮਾਪ(dst->type));
        vk_fa_xe_opt_ਧੱਕੋ_constants pc_ph1 = { (uint32_t)nek1, (uint32_t)neq1, (uint32_t)neq2, (uint32_t)nek2, qk_ਅਨੁਪਾਤ, 1, (sinks != ਖਾਲੀ_ਸੂਚਕ) ? 1u : 0u, (uint32_t)k_stride, ਸਿਰ_stride_k,
            batch_stride_q, batch_stride_k, batch_stride_v, batch_stride_m, batch_stride_o, scale };
        vk_fa_xe_opt_ਧੱਕੋ_constants pc_ph2 = pc_ph1;
        pc_ph2.nbkv_tok = v_stride;
        pc_ph2.nbkv_ਸਿਰ = ਸਿਰ_stride_v;
        vk_subbuffer q_ਆਰਜ਼ੀ_buf = fa_copy_qstate ? ggml_vk_subbuffer(ctx, ctx->prealloc_x, ਆਰਜ਼ੀ_buf_ਵਿੱਥ) : q_buf;
        ਆਰਜ਼ੀ_buf_ਵਿੱਥ += fa_copy_qstate ? x_ne * ਆਕਾਰ(ggml_fp16_t) : 0;
        vk_subbuffer p_ਆਰਜ਼ੀ_buf = ggml_vk_subbuffer(ctx, ctx->prealloc_x, ਆਰਜ਼ੀ_buf_ਵਿੱਥ);
        ਆਰਜ਼ੀ_buf_ਵਿੱਥ += ਮਾਪ_p;
        vk_subbuffer max_ਆਰਜ਼ੀ_buf = ggml_vk_subbuffer(ctx, ctx->prealloc_x, ਆਰਜ਼ੀ_buf_ਵਿੱਥ);
        ਆਰਜ਼ੀ_buf_ਵਿੱਥ += ਮਾਪ_ਸਮੂਹ_max;
        uint32_t xe_native_sub_ਸਮੂਹ_ਮਾਪ = ctx->device.get()->architecture == INTEL_XE1 ? 8 : 16;
        uint32_t aligned_gqa_ਅਨੁਪਾਤ = ਉੱਤਲਾ_power_ਦਾ_2(qk_ਅਨੁਪਾਤ);
        uint32_t ਬਾਹਰਲਾ_ਪ੍ਰਤੀ_wg_ph1 = std::min(256u / xe_native_sub_ਸਮੂਹ_ਮਾਪ, (uint32_t)neq1);
        uint32_t ਬਾਹਰਲਾ_ਪ੍ਰਤੀ_wg_ph2 = std::min(std::max(16u / aligned_gqa_ਅਨੁਪਾਤ, 1u), (uint32_t)neq1);
        uint32_t ph1_wg = ((neq1 + ਬਾਹਰਲਾ_ਪ੍ਰਤੀ_wg_ph1 - 1) / ਬਾਹਰਲਾ_ਪ੍ਰਤੀ_wg_ph1) * nek2;
        uint32_t ph2_wg = ((neq1 + ਬਾਹਰਲਾ_ਪ੍ਰਤੀ_wg_ph2 - 1) / ਬਾਹਰਲਾ_ਪ੍ਰਤੀ_wg_ph2) * ne0 / ਬਾਹਰਲਾ_dim_ਪ੍ਰਤੀ_wg;
        ਜੇ (fa_copy_qstate) {
            ਅਚਲ std::vector<uint32_t> pc_cpy_fp16 =
            { (uint32_t)q->ne[0], (uint32_t)q->ne[1], (uint32_t)q->ne[2], (uint32_t)q->ne[3], (uint32_t)(x_ne) };
            ggml_vk_sync_buffers(ctx, subctx);
            ggml_pipeline_request_descriptor_sets(ctx, ਮੰਜ਼ਿਲ_fp16_vk_0, 1);
            ggml_vk_dispatch_pipeline(ctx, subctx, ਮੰਜ਼ਿਲ_fp16_vk_0, { q_buf, q_ਆਰਜ਼ੀ_buf }, pc_cpy_fp16, { (uint32_t)(x_ne), 1, 1 });
        }

        ggml_vk_sync_buffers(ctx, subctx);
        ggml_pipeline_request_descriptor_sets(ctx, xe_fa_pipeline_dual_phases.ਪਹਿਲਾ, 1);
        ggml_vk_dispatch_pipeline(ctx, subctx, xe_fa_pipeline_dual_phases.ਪਹਿਲਾ,
            { q_ਆਰਜ਼ੀ_buf, k_buf, ਨਕਾਬ_buf, p_ਆਰਜ਼ੀ_buf, max_ਆਰਜ਼ੀ_buf },
            pc_ph1, { (uint32_t)ph1_wg, (uint32_t)nek1, (uint32_t)neq3 });

        ggml_vk_sync_buffers(ctx, subctx);
        ggml_pipeline_request_descriptor_sets(ctx, xe_fa_pipeline_dual_phases.second, 1);
        ggml_vk_dispatch_pipeline(ctx, subctx, xe_fa_pipeline_dual_phases.second,
            { p_ਆਰਜ਼ੀ_buf, v_buf, max_ਆਰਜ਼ੀ_buf, sinks_buf, dst_buf },
            pc_ph2, { (uint32_t)ph2_wg, (uint32_t)nev2, (uint32_t)neq3 });

        ctx->prealloc_x_need_sync = ਸੱਚ;
    } ਨਹੀਂਤਾਂ ਜੇ (ਵੰਡੋ_k > 1) {
        ggml_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_flash_attn_ਵੰਡੋ_k_reduce, 1);

        ਜੇ (ctx->prealloc_ਵੰਡੋ_k_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }

        // We reuse workgroups_x to mean the number of splits, so we need to
        // cancel out the divide by wg_denoms[0].
        uint32_t dispatch_x;
        ਜੇ (gqa_ਅਨੁਪਾਤ > 1) {
            workgroups_x *= pipeline->wg_denoms[0];
            dispatch_x = ਵੰਡੋ_k * workgroups_x;
        } ਨਹੀਂਤਾਂ {
            dispatch_x = Tr * ਵੰਡੋ_k * pipeline->wg_denoms[0];
        }

        vk_subbuffer ਵੰਡੋ_k_buf = ggml_vk_subbuffer(ctx, ctx->prealloc_ਵੰਡੋ_k, 0);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
                                    {q_buf, k_buf, v_buf, ਨਕਾਬ_buf, sinks_buf, ਵੰਡੋ_k_buf, ਨਕਾਬ_opt_buf, sparse_buf},
                                    pc, { dispatch_x, workgroups_y, workgroups_z });

        ggml_vk_sync_buffers(ctx, subctx);
        ਅਚਲ vk_op_flash_attn_ਵੰਡੋ_k_reduce_ਧੱਕੋ_constants pc2 = { HSV, (uint32_t)ne1, (uint32_t)ne2, (uint32_t)ne3, ਵੰਡੋ_k, (sinks != ਖਾਲੀ_ਸੂਚਕ) };
        ggml_vk_dispatch_pipeline(ctx, subctx, ctx->device->pipeline_flash_attn_ਵੰਡੋ_k_reduce,
                                    {ਵੰਡੋ_k_buf, sinks_buf, dst_buf},
                                    pc2, { (uint32_t)ne1, HSV, (uint32_t)(ne2 * ne3) });
        ctx->prealloc_ਵੰਡੋ_k_need_sync = ਸੱਚ;
    } ਨਹੀਂਤਾਂ {
        ਜੇ (gqa_ਅਨੁਪਾਤ > 1) {
            // When using gqa, we want one actual workgroup per batch, so cancel out wg_denoms
            workgroups_x *= pipeline->wg_denoms[0];
        }
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
                                    {q_buf, k_buf, v_buf, ਨਕਾਬ_buf, sinks_buf, dst_buf, ਨਕਾਬ_opt_buf, sparse_buf},
                                    pc, { workgroups_x, workgroups_y, workgroups_z });
    }

    ਜੇ (use_dequant_kv) {
        ctx->prealloc_x_need_sync = ਸੱਚ;
    }
    ਜੇ (use_ਨਕਾਬ_opt || use_sparse) {
        ctx->prealloc_y_need_sync = ਸੱਚ;
    }
}

ਸਥਿਰ vk_conv_shapes ggml_vk_conv_select_shape(ggml_backend_vk_context * ctx, uint32_t K, uint32_t NPQ) {
    ਸਵੈਚਾਲਿਤ n_tiles = [&](vk_conv_shapes s) {
        ਵਾਪਸ CEIL_DIV(K, vk_conv_ਖੰਡ_sizes[s].K)
            * CEIL_DIV(NPQ, vk_conv_ਖੰਡ_sizes[s].NPQ);
    };

    // We can't query number of shader cores on Intel, use 32 as a placeholder
    // so small convolutions will still choose a smaller tile.
    ਅਚਲ uint32_t shader_core_ਗਿਣਤੀ = ctx->device->shader_core_ਗਿਣਤੀ > 0 ? ctx->device->shader_core_ਗਿਣਤੀ : 32;

    // 128x128 isn't used with cm1 due to shared memory size; fall through to a smaller tile.
    ਤਾਰਕਿਕ allow_128x128 = ਸੱਚ;
#ਜੇ defined(VK_KHR_cooperative_ਮੈਟ੍ਰਿਕਸ) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    ਜੇ (!ctx->device->coopmat2 && ctx->device->coopmat_support && ctx->device->coopmat_support_16x16x16_f16acc) {
        allow_128x128 = ਝੂਠ;
    }
#ਜੇ_ਅੰਤ

    ਜੇ (allow_128x128 && K > 64 && n_tiles(CONV_SHAPE_128x128) >= shader_core_ਗਿਣਤੀ * 2) {
        ਵਾਪਸ CONV_SHAPE_128x128;
    } ਨਹੀਂਤਾਂ ਜੇ (K <= 32 && n_tiles(CONV_SHAPE_32x256) >= shader_core_ਗਿਣਤੀ * 2) {
        ਵਾਪਸ CONV_SHAPE_32x256;
    } ਨਹੀਂਤਾਂ ਜੇ (K <= 64 && n_tiles(CONV_SHAPE_64x128) >= shader_core_ਗਿਣਤੀ * 2) {
        ਵਾਪਸ CONV_SHAPE_64x128;
    } ਨਹੀਂਤਾਂ ਜੇ (!allow_128x128 && K > 64 && n_tiles(CONV_SHAPE_64x128) >= shader_core_ਗਿਣਤੀ * 2) {
        // cm1 fallback for large K when 128x128 isn't available
        ਵਾਪਸ CONV_SHAPE_64x128;
    } ਨਹੀਂਤਾਂ {
        ਵਾਪਸ CONV_SHAPE_64x32;
    }
}

ਸਥਿਰ vk_pipeline ggml_vk_op_get_pipeline(ggml_backend_vk_context * ctx, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ਅਚਲ ggml_tensor * src2, ਅਚਲ ggml_tensor * dst, ggml_op op) {
    ਚੋਣ (op) {
    ਹਾਲਤ GGML_OP_GET_ਪੰਕਤੀਆਂ:
        GGML_ASSERT(src1->type == GGML_TYPE_I32);
        ਜੇ (src0->type == GGML_TYPE_I32) {
            // i32 src only supports i32 result
            GGML_ASSERT(dst->type == GGML_TYPE_I32);
            ਵਾਪਸ ctx->device->pipeline_get_ਪੰਕਤੀਆਂ[src0->type];
        }
        ਜੇ (dst->type == GGML_TYPE_F16) {
            ਵਾਪਸ ctx->device->pipeline_get_ਪੰਕਤੀਆਂ[src0->type];
        }
        ਜੇ (dst->type == GGML_TYPE_F32) {
            ਵਾਪਸ ctx->device->pipeline_get_ਪੰਕਤੀਆਂ_f32[src0->type];
        }
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    ਹਾਲਤ GGML_OP_GET_ਪੰਕਤੀਆਂ_ਪਿੱਠ:
        ਜੇ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32) {
            ਵਾਪਸ ctx->device->pipeline_get_ਪੰਕਤੀਆਂ_ਪਿੱਠ_f32;
        }
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    ਹਾਲਤ GGML_OP_ACC:
        ਜੇ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ਵਾਪਸ ctx->device->pipeline_acc_f32;
        }
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    ਹਾਲਤ GGML_OP_ਸੈੱਟ:
        ਜੇ (src0->type == src1->type && src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_I32)) {
            ਵਾਪਸ ctx->device->pipeline_ਸੈੱਟ_f32;
        }
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    ਹਾਲਤ GGML_OP_ਜੋੜੋ:
    ਹਾਲਤ GGML_OP_SUB:
    ਹਾਲਤ GGML_OP_MUL:
    ਹਾਲਤ GGML_OP_DIV:
        ਜੇ ((src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) ||
            (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) ||
            (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16)) {
            ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
        }
        ਚੋਣ (op) {
        ਹਾਲਤ GGML_OP_ਜੋੜੋ:
        {
            ਜੇ (ctx->ਅੰਕ_additional_fused_ops > 0) {
                ਜੇ (ctx->do_ਜੋੜੋ_rms_partials) {
                    ਵਾਪਸ ctx->device->pipeline_multi_ਜੋੜੋ_rms[ctx->ਅੰਕ_additional_fused_ops];
                } ਨਹੀਂਤਾਂ {
                    ਵਾਪਸ ctx->device->pipeline_multi_ਜੋੜੋ[ctx->ਅੰਕ_additional_fused_ops];
                }
            }
            ਜੇ (ctx->do_ਜੋੜੋ_rms_partials) {
                ਸਵੈਚਾਲਿਤ pipelines = ggml_are_same_shape(src0, src1) ? ctx->device->pipeline_ਜੋੜੋ_rms_norepeat : ctx->device->pipeline_ਜੋੜੋ_rms;
                ਵਾਪਸ pipelines[src0->type == GGML_TYPE_F16][src1->type == GGML_TYPE_F16][dst->type == GGML_TYPE_F16];
            } ਨਹੀਂਤਾਂ {
                ਸਵੈਚਾਲਿਤ pipelines = ggml_are_same_shape(src0, src1) ? ctx->device->pipeline_ਜੋੜੋ_norepeat : ctx->device->pipeline_ਜੋੜੋ;
                ਵਾਪਸ pipelines[src0->type == GGML_TYPE_F16][src1->type == GGML_TYPE_F16][dst->type == GGML_TYPE_F16];
            }
        }
        ਹਾਲਤ GGML_OP_SUB:
        {
            ਸਵੈਚਾਲਿਤ pipelines = ggml_are_same_shape(src0, src1) ? ctx->device->pipeline_sub_norepeat : ctx->device->pipeline_sub;
            ਵਾਪਸ pipelines[src0->type == GGML_TYPE_F16][src1->type == GGML_TYPE_F16][dst->type == GGML_TYPE_F16];
        }
        ਹਾਲਤ GGML_OP_MUL:
        {
            ਸਵੈਚਾਲਿਤ pipelines = ggml_are_same_shape(src0, src1) ? ctx->device->pipeline_mul_norepeat : ctx->device->pipeline_mul;
            ਵਾਪਸ pipelines[src0->type == GGML_TYPE_F16][src1->type == GGML_TYPE_F16][dst->type == GGML_TYPE_F16];
        }
        ਹਾਲਤ GGML_OP_DIV:
        {
            ਸਵੈਚਾਲਿਤ pipelines = ggml_are_same_shape(src0, src1) ? ctx->device->pipeline_div_norepeat : ctx->device->pipeline_div;
            ਵਾਪਸ pipelines[src0->type == GGML_TYPE_F16][src1->type == GGML_TYPE_F16][dst->type == GGML_TYPE_F16];
        }
        ਮੂਲ:
            ਰੋਕੋ;
        }
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    ਹਾਲਤ GGML_OP_ਜੋੜੋ_ID:
        ਜੇ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && src2->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32) {
            ਵਾਪਸ ctx->device->pipeline_ਜੋੜੋ_id_f32;
        }
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    ਹਾਲਤ GGML_OP_ਬਾਹਰਲਾ_PROD:
        ਜੇ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ਵਾਪਸ ctx->device->pipeline_ਬਾਹਰਲਾ_prod_f32;
        }
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    ਹਾਲਤ GGML_OP_CONCAT: {
        ਜੇ (!ggml_vk_concat_supported(src0, src1, dst)) {
            ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
        }
        ਚੋਣ (ggml_vk_concat_unit_ਮਾਪ(src0->type)) {
        ਹਾਲਤ 1:
            ਵਾਪਸ ctx->device->pipeline_concat_i8;
        ਹਾਲਤ 2:
            ਵਾਪਸ ctx->device->pipeline_concat_i16;
        ਹਾਲਤ 4:
            ਵਾਪਸ ctx->device->pipeline_concat_i32;
        ਹਾਲਤ 8:
            ਵਾਪਸ ctx->device->pipeline_concat_i64;
        ਮੂਲ:
            ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
        }
    }
    ਹਾਲਤ GGML_OP_UPSCALE:
        ਜੇ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            uint32_t mode = (ggml_get_op_params_i32(dst, 0) & (0xFF | GGML_SCALE_FLAG_ANTIALIAS));
            ਚੋਣ (mode) {
                ਹਾਲਤ GGML_SCALE_ਬਹੁਲਕ_ਸਭ_ਤੋਂ_ਨੇੜੇ:
                    ਵਾਪਸ ctx->device->pipeline_upscale_ਸਭ_ਤੋਂ_ਨੇੜੇ_f32;
                ਹਾਲਤ GGML_SCALE_ਬਹੁਲਕ_BILINEAR:
                    ਵਾਪਸ ctx->device->pipeline_upscale_bilinear_f32;
                ਹਾਲਤ GGML_SCALE_ਬਹੁਲਕ_BICUBIC:
                    ਵਾਪਸ ctx->device->pipeline_upscale_bicubic_f32;
                ਹਾਲਤ GGML_SCALE_ਬਹੁਲਕ_BILINEAR | GGML_SCALE_FLAG_ANTIALIAS:
                    ਵਾਪਸ ctx->device->pipeline_upscale_bilinear_antialias_f32;
                ਮੂਲ:
                    ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
            }
        }
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    ਹਾਲਤ GGML_OP_SCALE:
        ਜੇ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ਵਾਪਸ ctx->device->pipeline_scale_f32;
        }
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    ਹਾਲਤ GGML_OP_SQR:
        ਜੇ (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            ਵਾਪਸ ctx->device->pipeline_sqr[dst->type == GGML_TYPE_F16];
        }
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    ਹਾਲਤ GGML_OP_SQRT:
        ਜੇ (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            ਵਾਪਸ ctx->device->pipeline_sqrt[dst->type == GGML_TYPE_F16];
        }
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    ਹਾਲਤ GGML_OP_SIN:
        ਜੇ (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            ਵਾਪਸ ctx->device->pipeline_sin[dst->type == GGML_TYPE_F16];
        }
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    ਹਾਲਤ GGML_OP_COS:
        ਜੇ (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            ਵਾਪਸ ctx->device->pipeline_cos[dst->type == GGML_TYPE_F16];
        }
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    ਹਾਲਤ GGML_OP_LOG:
        ਜੇ (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            ਵਾਪਸ ctx->device->pipeline_log[dst->type == GGML_TYPE_F16];
        }
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    ਹਾਲਤ GGML_OP_TRI:
        ਜੇ (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            ਵਾਪਸ ctx->device->pipeline_tri[dst->type == GGML_TYPE_F16];
        }
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    ਹਾਲਤ GGML_OP_DIAG:
        ਜੇ (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            ਵਾਪਸ ctx->device->pipeline_diag[dst->type == GGML_TYPE_F16];
        }
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    ਹਾਲਤ GGML_OP_CLAMP:
        ਜੇ (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            ਵਾਪਸ ctx->device->pipeline_clamp[dst->type == GGML_TYPE_F16];
        }
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    ਹਾਲਤ GGML_OP_PAD:
        ਜੇ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ਵਾਪਸ ctx->device->pipeline_pad_f32;
        }
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    ਹਾਲਤ GGML_OP_PAD_REFLECT_1D:
        ਜੇ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ਵਾਪਸ ctx->device->pipeline_pad_reflect_1d_f32;
        }
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    ਹਾਲਤ GGML_OP_ROLL:
        ਜੇ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ਵਾਪਸ ctx->device->pipeline_roll_f32;
        }
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    ਹਾਲਤ GGML_OP_REPEAT:
        ਜੇ (ggml_type_ਮਾਪ(src0->type) == ਆਕਾਰ(ਦਸ਼ਮਲਵ) && ggml_type_ਮਾਪ(dst->type) == ਆਕਾਰ(ਦਸ਼ਮਲਵ)) {
            ਵਾਪਸ ctx->device->pipeline_repeat_i32;
        }
        ਜੇ (ggml_type_ਮਾਪ(src0->type) == 2 && ggml_type_ਮਾਪ(dst->type) == 2) {
            ਵਾਪਸ ctx->device->pipeline_repeat_i16;
        }
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    ਹਾਲਤ GGML_OP_REPEAT_ਪਿੱਠ:
        ਜੇ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ਵਾਪਸ ctx->device->pipeline_repeat_ਪਿੱਠ_f32;
        }
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    ਹਾਲਤ GGML_OP_CPY:
    ਹਾਲਤ GGML_OP_CONT:
    ਹਾਲਤ GGML_OP_DUP:
        ਵਾਪਸ ggml_vk_get_cpy_pipeline(ctx, src0, dst, dst->type);
    ਹਾਲਤ GGML_OP_ਸੈੱਟ_ਪੰਕਤੀਆਂ:
        {
            ਜੇ (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) {
                ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
            }
            ਅਚਲ ਪੂਰਨ src_idx = src0->type == GGML_TYPE_F16;
            ਜੇ (src1->type == GGML_TYPE_I64) {
                ਵਾਪਸ ctx->device->pipeline_ਸੈੱਟ_ਪੰਕਤੀਆਂ_i64[src_idx][dst->type];
            } ਨਹੀਂਤਾਂ ਜੇ (src1->type == GGML_TYPE_I32) {
                ਵਾਪਸ ctx->device->pipeline_ਸੈੱਟ_ਪੰਕਤੀਆਂ_i32[src_idx][dst->type];
            }
            ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
        }
    ਹਾਲਤ GGML_OP_SILU_ਪਿੱਠ:
        ਜੇ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ਵਾਪਸ ctx->device->pipeline_silu_ਪਿੱਠ_f32;
        }
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    ਹਾਲਤ GGML_OP_NORM:
        ਜੇ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ਵਾਪਸ ctx->device->pipeline_norm_f32;
        }
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    ਹਾਲਤ GGML_OP_ਸਮੂਹ_NORM:
        ਜੇ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ਵਾਪਸ ctx->device->pipeline_ਸਮੂਹ_norm_f32;
        }
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    ਹਾਲਤ GGML_OP_RMS_NORM:
        ਜੇ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ਜੇ (ctx->do_ਜੋੜੋ_rms_partials) {
                ਵਾਪਸ ctx->fused_rms_norm_ਬਹੁਲਕ == RMS_NORM_MUL ? ctx->device->pipeline_rms_norm_mul_partials_f32 : ctx->device->pipeline_rms_norm_partials_f32;
            }
            ਵਾਪਸ ctx->fused_rms_norm_ਬਹੁਲਕ == RMS_NORM_MUL ? ctx->device->pipeline_rms_norm_mul_f32 : ctx->device->pipeline_rms_norm_f32;
        }
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    ਹਾਲਤ GGML_OP_RMS_NORM_ਪਿੱਠ:
        ਜੇ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ਵਾਪਸ ctx->device->pipeline_rms_norm_ਪਿੱਠ_f32;
        }
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    ਹਾਲਤ GGML_OP_L2_NORM:
        ਜੇ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ਵਾਪਸ ctx->device->pipeline_l2_norm_f32;
        }
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    ਹਾਲਤ GGML_OP_UNARY:
        ਜੇ ((src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) ||
            (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) ||
            (src0->type != dst->type)) {
            ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
        }

        ਚੋਣ (ggml_get_unary_op(dst)) {
            ਹਾਲਤ GGML_UNARY_OP_EXP:
                ਵਾਪਸ ctx->device->pipeline_exp[dst->type == GGML_TYPE_F16];
            ਹਾਲਤ GGML_UNARY_OP_EXPM1:
                ਵਾਪਸ ctx->device->pipeline_expm1[dst->type == GGML_TYPE_F16];
            ਹਾਲਤ GGML_UNARY_OP_ELU:
                ਵਾਪਸ ctx->device->pipeline_elu[dst->type == GGML_TYPE_F16];
            ਹਾਲਤ GGML_UNARY_OP_SILU:
                ਵਾਪਸ ctx->device->pipeline_silu[dst->type == GGML_TYPE_F16];
            ਹਾਲਤ GGML_UNARY_OP_GELU:
                ਵਾਪਸ ctx->device->pipeline_gelu[dst->type == GGML_TYPE_F16];
            ਹਾਲਤ GGML_UNARY_OP_GELU_ERF:
                ਵਾਪਸ ctx->device->pipeline_gelu_erf[dst->type == GGML_TYPE_F16];
            ਹਾਲਤ GGML_UNARY_OP_GELU_ਤੇਜ਼:
                ਵਾਪਸ ctx->device->pipeline_gelu_ਤੇਜ਼[dst->type == GGML_TYPE_F16];
            ਹਾਲਤ GGML_UNARY_OP_RELU:
                ਵਾਪਸ ctx->device->pipeline_relu[dst->type == GGML_TYPE_F16];
            ਹਾਲਤ GGML_UNARY_OP_XIELU:
                ਵਾਪਸ ctx->device->pipeline_xielu[dst->type == GGML_TYPE_F16];
            ਹਾਲਤ GGML_UNARY_OP_NEG:
                ਵਾਪਸ ctx->device->pipeline_neg[dst->type == GGML_TYPE_F16];
            ਹਾਲਤ GGML_UNARY_OP_TANH:
                ਵਾਪਸ ctx->device->pipeline_tanh[dst->type == GGML_TYPE_F16];
            ਹਾਲਤ GGML_UNARY_OP_SIGMOID:
                ਵਾਪਸ ctx->device->pipeline_sigmoid[dst->type == GGML_TYPE_F16];
            ਹਾਲਤ GGML_UNARY_OP_HARDSIGMOID:
                ਵਾਪਸ ctx->device->pipeline_hardsigmoid[dst->type == GGML_TYPE_F16];
            ਹਾਲਤ GGML_UNARY_OP_HARDSWISH:
                ਵਾਪਸ ctx->device->pipeline_hardswish[dst->type == GGML_TYPE_F16];
            ਹਾਲਤ GGML_UNARY_OP_ABS:
                ਵਾਪਸ ctx->device->pipeline_abs[dst->type == GGML_TYPE_F16];
            ਹਾਲਤ GGML_UNARY_OP_SOFTPLUS:
                ਵਾਪਸ ctx->device->pipeline_softplus[dst->type == GGML_TYPE_F16];
            ਹਾਲਤ GGML_UNARY_OP_ਕਦਮ:
                ਵਾਪਸ ctx->device->pipeline_ਕਦਮ[dst->type == GGML_TYPE_F16];
            ਹਾਲਤ GGML_UNARY_OP_ਗੇੜਾ:
                ਵਾਪਸ ctx->device->pipeline_ਗੇੜਾ[dst->type == GGML_TYPE_F16];
            ਹਾਲਤ GGML_UNARY_OP_CEIL:
                ਵਾਪਸ ctx->device->pipeline_ceil[dst->type == GGML_TYPE_F16];
            ਹਾਲਤ GGML_UNARY_OP_FLOOR:
                ਵਾਪਸ ctx->device->pipeline_floor[dst->type == GGML_TYPE_F16];
            ਹਾਲਤ GGML_UNARY_OP_TRUNC:
                ਵਾਪਸ ctx->device->pipeline_trunc[dst->type == GGML_TYPE_F16];
            ਹਾਲਤ GGML_UNARY_OP_SGN:
                ਵਾਪਸ ctx->device->pipeline_sgn[dst->type == GGML_TYPE_F16];
            ਮੂਲ:
                ਰੋਕੋ;
        }
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    ਹਾਲਤ GGML_OP_GLU:
        ਜੇ ((src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) ||
            (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) ||
            (src0->type != dst->type)) {
            ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
        }

        ਚੋਣ (ggml_get_glu_op(dst)) {
            ਹਾਲਤ GGML_GLU_OP_GEGLU:
                ਵਾਪਸ ctx->device->pipeline_geglu[dst->type == GGML_TYPE_F16];
            ਹਾਲਤ GGML_GLU_OP_REGLU:
                ਵਾਪਸ ctx->device->pipeline_reglu[dst->type == GGML_TYPE_F16];
            ਹਾਲਤ GGML_GLU_OP_SWIGLU:
                ਵਾਪਸ ctx->device->pipeline_swiglu[dst->type == GGML_TYPE_F16];
            ਹਾਲਤ GGML_GLU_OP_SWIGLU_OAI:
                ਵਾਪਸ ctx->device->pipeline_swiglu_oai[dst->type == GGML_TYPE_F16];
            ਹਾਲਤ GGML_GLU_OP_SWIGLU_CLAMP:
                ਵਾਪਸ ctx->device->pipeline_swiglu_clamp[dst->type == GGML_TYPE_F16];
            ਹਾਲਤ GGML_GLU_OP_GEGLU_ERF:
                ਵਾਪਸ ctx->device->pipeline_geglu_erf[dst->type == GGML_TYPE_F16];
            ਹਾਲਤ GGML_GLU_OP_GEGLU_ਤੇਜ਼:
                ਵਾਪਸ ctx->device->pipeline_geglu_ਤੇਜ਼[dst->type == GGML_TYPE_F16];
            ਮੂਲ:
                ਰੋਕੋ;
        }
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    ਹਾਲਤ GGML_OP_DIAG_ਨਕਾਬ_INF:
        ਜੇ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ਵਾਪਸ ctx->device->pipeline_diag_ਨਕਾਬ_inf_f32;
        }
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    ਹਾਲਤ GGML_OP_SOFT_MAX:
        GGML_ASSERT(!src1 || src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
        GGML_ASSERT(!src2 || src2->type == GGML_TYPE_F32);

        ਜੇ (ctx->ਅੰਕ_additional_fused_ops) {
            uint32_t idx = (uint32_t)ceilf(log2f(ਦਸ਼ਮਲਵ(dst->ne[0])));
            GGML_ASSERT(idx < ਅੰਕ_topk_moe_pipelines);
            // use n_experts from push constant if it's not equal to the power of two spec constant
            ਤਾਰਕਿਕ use_ਧੱਕੋ = dst->ne[0] != (1u << idx);
            ਵਾਪਸ ctx->device->pipeline_topk_moe[idx][use_ਧੱਕੋ];
        }

        ਜੇ (src0->type == GGML_TYPE_F32 && (src1 == ਖਾਲੀ_ਸੂਚਕ || src1->type == GGML_TYPE_F32) && dst->type == GGML_TYPE_F32) {
            ਵਾਪਸ src0->ne[0] > 1024 ? ctx->device->pipeline_soft_max_f32_wg512 : ctx->device->pipeline_soft_max_f32;
        }
        ਜੇ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) {
            ਵਾਪਸ src0->ne[0] > 1024 ? ctx->device->pipeline_soft_max_f32_f16_wg512 : ctx->device->pipeline_soft_max_f32_f16;
        }
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    ਹਾਲਤ GGML_OP_SOFT_MAX_ਪਿੱਠ:
        ਜੇ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ਵਾਪਸ ctx->device->pipeline_soft_max_ਪਿੱਠ_f32;
        }
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    ਹਾਲਤ GGML_OP_ROPE:
    ਹਾਲਤ GGML_OP_ROPE_ਪਿੱਠ:
        {
            ਅਚਲ ggml_tensor *rope = ctx->ਅੰਕ_additional_fused_ops == 2 ? dst->src[0]->src[0] : dst;
            ਅਚਲ ਪੂਰਨ mode = ((ਅਚਲ int32_t *) rope->op_params)[2];
            ਅਚਲ ਤਾਰਕਿਕ is_neox = mode & GGML_ROPE_TYPE_NEOX;
            ਅਚਲ ਤਾਰਕਿਕ is_mrope = mode & GGML_ROPE_TYPE_MROPE;
            ਅਚਲ ਤਾਰਕਿਕ is_vision = mode == GGML_ROPE_TYPE_VISION;

            ਜੇ (is_neox) {
                ਜੇ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
                    ਵਾਪਸ ctx->device->pipeline_rope_neox_f32;
                }
                ਜੇ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
                    ਵਾਪਸ ctx->device->pipeline_rope_neox_f32_f16;
                }
                ਜੇ (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
                    ਵਾਪਸ ctx->device->pipeline_rope_neox_f16;
                }
            } ਨਹੀਂਤਾਂ ਜੇ (is_mrope && !is_vision) {
                ਜੇ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
                    ਵਾਪਸ ctx->device->pipeline_rope_multi_f32;
                }
                ਜੇ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
                    ਵਾਪਸ ctx->device->pipeline_rope_multi_f32_f16;
                }
                ਜੇ (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
                    ਵਾਪਸ ctx->device->pipeline_rope_multi_f16;
                }
            } ਨਹੀਂਤਾਂ ਜੇ (is_vision) {
                ਜੇ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
                    ਵਾਪਸ ctx->device->pipeline_rope_vision_f32;
                }
                ਜੇ (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
                    ਵਾਪਸ ctx->device->pipeline_rope_vision_f16;
                }
            } ਨਹੀਂਤਾਂ {
                ਜੇ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
                    ਵਾਪਸ ctx->device->pipeline_rope_norm_f32;
                }
                ਜੇ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
                    ਵਾਪਸ ctx->device->pipeline_rope_norm_f32_f16;
                }
                ਜੇ (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
                    ਵਾਪਸ ctx->device->pipeline_rope_norm_f16;
                }
            }
            ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
        }
    ਹਾਲਤ GGML_OP_ਜੋੜ:
    ਹਾਲਤ GGML_OP_ਜੋੜ_ਪੰਕਤੀਆਂ:
    ਹਾਲਤ GGML_OP_ਔਸਤ:
        ਜੇ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ਵਾਪਸ ctx->device->pipeline_ਜੋੜ_ਪੰਕਤੀਆਂ_f32;
        }
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    ਹਾਲਤ GGML_OP_CROSS_ENTROPY_LOSS:
        ਜੇ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ਵਾਪਸ src0->ne[0] > 1024 ? ctx->device->pipeline_cross_entropy_loss_f32_wg512 : ctx->device->pipeline_cross_entropy_loss_f32;
        }
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    ਹਾਲਤ GGML_OP_CROSS_ENTROPY_LOSS_ਪਿੱਠ:
        // src0 is the scalar grad; src1 is logits
        ਜੇ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && src2 && src2->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ਵਾਪਸ src1->ne[0] > 1024 ? ctx->device->pipeline_cross_entropy_loss_ਪਿੱਠ_f32_wg512 : ctx->device->pipeline_cross_entropy_loss_ਪਿੱਠ_f32;
        }
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    ਹਾਲਤ GGML_OP_CUMSUM:
        ਜੇ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ਜੇ (src0->ne[0] <= 512) {
                ਵਾਪਸ ctx->device->pipeline_cumsum_small_f32;
            } ਨਹੀਂਤਾਂ {
                ਵਾਪਸ ctx->device->pipeline_cumsum_f32;
            }
        }
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    ਹਾਲਤ GGML_OP_SOLVE_TRI:
        ਜੇ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {

            vk_solve_tri_pipeline_state solve_tri_pipeline_state(src0->ne[0], src1->ne[0]);

            vk_pipeline pipeline = ਖਾਲੀ_ਸੂਚਕ;

            {
                std::lock_guard<std::mutex> guard(ctx->device->compile_mutex);
                ਸਵੈਚਾਲਿਤ it = ctx->device->pipeline_solve_tri_f32.find(solve_tri_pipeline_state);
                ਜੇ (it != ctx->device->pipeline_solve_tri_f32.end()) {
                    pipeline = it->second;
                } ਨਹੀਂਤਾਂ {
                    ctx->device->pipeline_solve_tri_f32[solve_tri_pipeline_state] = pipeline = std::make_shared<vk_pipeline_struct>();
                }
            }

            ਵਾਪਸ pipeline;
        }
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    ਹਾਲਤ GGML_OP_ARGMAX:
        ਜੇ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_I32) {
            ਵਾਪਸ ctx->device->pipeline_argmax_f32;
        }
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    ਹਾਲਤ GGML_OP_ਗਿਣਤੀ_EQUAL:
        ਜੇ (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_I64) {
            ਵਾਪਸ ctx->device->pipeline_ਗਿਣਤੀ_equal_i32;
        }
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    ਹਾਲਤ GGML_OP_IM2_ਖਾਨਾ:
        ਜੇ (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ਵਾਪਸ ctx->device->pipeline_im2col_f32;
        }
        ਜੇ (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
            ਵਾਪਸ ctx->device->pipeline_im2col_f32_f16;
        }
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    ਹਾਲਤ GGML_OP_IM2_ਖਾਨਾ_3_D:
        ਜੇ (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ਵਾਪਸ ctx->device->pipeline_im2col_3d_f32;
        }
        ਜੇ (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
            ਵਾਪਸ ctx->device->pipeline_im2col_3d_f32_f16;
        }
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    ਹਾਲਤ GGML_OP_TIMESTEP_EMBEDDING:
        ਜੇ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ਵਾਪਸ ctx->device->pipeline_timestep_embedding_f32;
        }
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    ਹਾਲਤ GGML_OP_CONV_TRANSPOSE_1D:
        ਜੇ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ਵਾਪਸ ctx->device->pipeline_conv_transpose_1d_f32;
        }
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    ਹਾਲਤ GGML_OP_COL2IM_1D:
        ਚੋਣ (src0->type) {
            ਹਾਲਤ GGML_TYPE_F32:  ਵਾਪਸ ctx->device->pipeline_col2im_1d_f32;
            ਹਾਲਤ GGML_TYPE_F16:  ਵਾਪਸ ctx->device->pipeline_col2im_1d_f16;
            ਹਾਲਤ GGML_TYPE_BF16: ਵਾਪਸ ctx->device->pipeline_col2im_1d_bf16;
            ਮੂਲ:             ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
        }
    ਹਾਲਤ GGML_OP_POOL_1D:
        ਜੇ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ਵਾਪਸ ctx->device->pipeline_pool1d_f32;
        }
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    ਹਾਲਤ GGML_OP_POOL_2D:
        ਜੇ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            // Wide windows use neighboring subgroup lanes to read neighboring inputs.
            ਜੇ (ctx->device->pipeline_pool2d_subgroup_f32 && dst->op_params[1] >= int32_t(ctx->device->subgroup_ਮਾਪ)) {
                ਵਾਪਸ ctx->device->pipeline_pool2d_subgroup_f32;
            }
            ਵਾਪਸ ctx->device->pipeline_pool2d_f32;
        }
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    ਹਾਲਤ GGML_OP_RWKV_WKV6:
        ਜੇ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ਵਾਪਸ ctx->device->pipeline_rwkv_wkv6_f32;
        }
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    ਹਾਲਤ GGML_OP_RWKV_WKV7:
        ਜੇ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ਵਾਪਸ ctx->device->pipeline_rwkv_wkv7_f32;
        }
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    ਹਾਲਤ GGML_OP_GATED_ਰੇਖਿਕ_ATTN:
        ਜੇ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ਵਾਪਸ ctx->device->pipeline_gated_ਰੇਖਿਕ_attn_f32;
        }
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    ਹਾਲਤ GGML_OP_LIGHTNING_INDEXER:
        // only the k type selects a pipeline, the other types are fixed by ggml_lightning_indexer()
        ਜੇ (ggml_vk_lightning_indexer_k_type_supported(src1->type)) {
            ਵਾਪਸ ctx->device->pipeline_lightning_indexer_f32[src1->type];
        }
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    ਹਾਲਤ GGML_OP_GATED_DELTA_NET:
        ਜੇ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ਅਚਲ uint32_t S_v = dst->src[2]->ne[0];
            ਅਚਲ uint32_t kda = (dst->src[3]->ne[0] == (int64_t)S_v) ? 1 : 0;
            uint32_t si;
            ਚੋਣ (S_v) {
                ਹਾਲਤ 16:  si = 0; ਰੋਕੋ;
                ਹਾਲਤ 32:  si = 1; ਰੋਕੋ;
                ਹਾਲਤ 64:  si = 2; ਰੋਕੋ;
                ਹਾਲਤ 128: si = 3; ਰੋਕੋ;
                ਮੂਲ: ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
            }
            ਵਾਪਸ ctx->device->pipeline_gated_delta_net[si][kda];
        }
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    ਹਾਲਤ GGML_OP_SSM_ਜਾਂਚ:
        ਜੇ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ਅਚਲ uint32_t d_state = src0->ne[0];
            ਜੇ (d_state == 128) {
                ਵਾਪਸ ctx->device->pipeline_ssm_ਜਾਂਚ_f32_d128;
            } ਨਹੀਂਤਾਂ ਜੇ (d_state == 256) {
                ਵਾਪਸ ctx->device->pipeline_ssm_ਜਾਂਚ_f32_d256;
            }
        }
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    ਹਾਲਤ GGML_OP_SSM_CONV:
        ਜੇ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ਚੋਣ (ctx->ਅੰਕ_additional_fused_ops) {
                ਹਾਲਤ 0:  ਵਾਪਸ ctx->device->pipeline_ssm_conv_f32;
                ਹਾਲਤ 1:  ਵਾਪਸ ctx->device->pipeline_ssm_conv_silu_f32;
                ਹਾਲਤ 2:  ਵਾਪਸ ctx->device->pipeline_ssm_conv_bias_silu_f32;
                ਮੂਲ: ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
            }
        }
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    ਹਾਲਤ GGML_OP_OPT_ਕਦਮ_ADAMW:
        ਜੇ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ਵਾਪਸ ctx->device->pipeline_opt_ਕਦਮ_adamw_f32;
        }
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    ਹਾਲਤ GGML_OP_OPT_ਕਦਮ_SGD:
        ਜੇ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ਵਾਪਸ ctx->device->pipeline_opt_ਕਦਮ_sgd_f32;
        }
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    ਹਾਲਤ GGML_OP_LEAKY_RELU:
        ਜੇ (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            ਵਾਪਸ ctx->device->pipeline_leaky_relu[dst->type == GGML_TYPE_F16];
        }
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    ਹਾਲਤ GGML_OP_CONV_2D:
    ਹਾਲਤ GGML_OP_CONV_TRANSPOSE_2D:
        ਜੇ (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            uint32_t K = dst->ne[2]; // Cout
            uint32_t NPQ = dst->ne[3] * dst->ne[1] * dst->ne[0]; // N * OH * OW
            vk_conv_shapes shape = ggml_vk_conv_select_shape(ctx, K, NPQ);

            ਤਾਰਕਿਕ transpose = dst->op == GGML_OP_CONV_TRANSPOSE_2D;
            uint32_t KW = (uint32_t)src0->ne[0];
            uint32_t KH = (uint32_t)src0->ne[1];
            uint32_t s0 = (uint32_t)(ggml_get_op_params_i32(dst, 0));
            uint32_t s1 = !transpose ? (uint32_t)ggml_get_op_params_i32(dst, 1) : s0;
            uint32_t p0 = !transpose ? (uint32_t)ggml_get_op_params_i32(dst, 2) : 0;
            uint32_t p1 = !transpose ? (uint32_t)ggml_get_op_params_i32(dst, 3) : 0;
            uint32_t d0 = !transpose ? (uint32_t)ggml_get_op_params_i32(dst, 4) : 1;
            uint32_t d1 = !transpose ? (uint32_t)ggml_get_op_params_i32(dst, 5) : 1;

            // tile-aligned shapes let the shader skip bounds checks
            ਅਚਲ uint32_t Cin = (uint32_t)src1->ne[2];
            ਅਚਲ uint32_t CRS = Cin * KW * KH;
            ਅਚਲ uint32_t BS_K   = vk_conv_ਖੰਡ_sizes[shape].K;
            ਅਚਲ uint32_t BS_CRS = vk_conv_ਖੰਡ_sizes[shape].CRS;
            ਅਚਲ uint32_t BS_NPQ = vk_conv_ਖੰਡ_sizes[shape].NPQ;
            ਅਚਲ uint32_t aligned = ((K   % BS_K   == 0) &&
                                      (CRS % BS_CRS == 0) &&
                                      (NPQ % BS_NPQ == 0)) ? 1u : 0u;

            vk_conv2d_pipeline_state conv2d_pipeline_state(s0, s1, p0, p1, d0, d1, KW, KH, aligned);

            std::map<vk_conv2d_pipeline_state, vk_pipeline> *pipelines = ਖਾਲੀ_ਸੂਚਕ;
            ਜੇ (op == GGML_OP_CONV_2D) {
                ਜੇ (src0->type == GGML_TYPE_F32) {
                    pipelines = &ctx->device->pipeline_conv2d_f32[shape];
                } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_F16) {
                    pipelines = &ctx->device->pipeline_conv2d_f16_f32[shape];
                }
            } ਨਹੀਂਤਾਂ ਜੇ (op == GGML_OP_CONV_TRANSPOSE_2D) {
                ਜੇ (src0->type == GGML_TYPE_F32) {
                    pipelines = &ctx->device->pipeline_conv_transpose_2d_f32[shape];
                } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_F16) {
                    pipelines = &ctx->device->pipeline_conv_transpose_2d_f16_f32[shape];
                }
            }

            vk_pipeline pipeline = ਖਾਲੀ_ਸੂਚਕ;

            {
                std::lock_guard<std::mutex> guard(ctx->device->compile_mutex);
                ਸਵੈਚਾਲਿਤ it = pipelines->find(conv2d_pipeline_state);
                ਜੇ (it != pipelines->end()) {
                    pipeline = it->second;
                } ਨਹੀਂਤਾਂ {
                    (*pipelines)[conv2d_pipeline_state] = pipeline = std::make_shared<vk_pipeline_struct>();
                }
            }

            ਵਾਪਸ pipeline;
        }
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    ਹਾਲਤ GGML_OP_CONV_2D_DW:
        ਜੇ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ਜੇ (ggml_is_contiguous(src1)) {
                ਵਾਪਸ ctx->device->pipeline_conv2d_dw_whcn_f32;
            } ਨਹੀਂਤਾਂ ਜੇ (ggml_is_contiguous_channels(src1)) {
                ਵਾਪਸ ctx->device->pipeline_conv2d_dw_cwhn_f32;
            }
        } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) {
            ਜੇ (ggml_is_contiguous(src1)) {
                ਵਾਪਸ ctx->device->pipeline_conv2d_dw_whcn_f16_f32;
            } ਨਹੀਂਤਾਂ ਜੇ (ggml_is_contiguous_channels(src1)) {
                ਵਾਪਸ ctx->device->pipeline_conv2d_dw_cwhn_f16_f32;
            }
        }
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    ਹਾਲਤ GGML_OP_CONV_3D:
        ਜੇ (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ਅਚਲ uint32_t OC = (uint32_t)ggml_get_op_params_i32(dst, 11);
            ਅਚਲ uint32_t IC = (uint32_t)ggml_get_op_params_i32(dst, 9);
            ਅਚਲ uint32_t N  = (uint32_t)ggml_get_op_params_i32(dst, 10);
            ਅਚਲ uint32_t NPQ = N * dst->ne[2] * dst->ne[1] * dst->ne[0];
            ਅਚਲ vk_conv_shapes shape = ggml_vk_conv_select_shape(ctx, OC, NPQ);

            ਅਚਲ uint32_t KW = (uint32_t)src0->ne[0];
            ਅਚਲ uint32_t KH = (uint32_t)src0->ne[1];
            ਅਚਲ uint32_t KD = (uint32_t)src0->ne[2];
            ਅਚਲ uint32_t s0 = (uint32_t)ggml_get_op_params_i32(dst, 0);
            ਅਚਲ uint32_t s1 = (uint32_t)ggml_get_op_params_i32(dst, 1);
            ਅਚਲ uint32_t s2 = (uint32_t)ggml_get_op_params_i32(dst, 2);
            ਅਚਲ uint32_t p0 = (uint32_t)ggml_get_op_params_i32(dst, 3);
            ਅਚਲ uint32_t p1 = (uint32_t)ggml_get_op_params_i32(dst, 4);
            ਅਚਲ uint32_t p2 = (uint32_t)ggml_get_op_params_i32(dst, 5);
            ਅਚਲ uint32_t d0 = (uint32_t)ggml_get_op_params_i32(dst, 6);
            ਅਚਲ uint32_t d1 = (uint32_t)ggml_get_op_params_i32(dst, 7);
            ਅਚਲ uint32_t d2 = (uint32_t)ggml_get_op_params_i32(dst, 8);

            ਅਚਲ uint32_t CRS = IC * KW * KH * KD;
            ਅਚਲ uint32_t BS_K   = vk_conv_ਖੰਡ_sizes[shape].K;
            ਅਚਲ uint32_t BS_CRS = vk_conv_ਖੰਡ_sizes[shape].CRS;
            ਅਚਲ uint32_t BS_NPQ = vk_conv_ਖੰਡ_sizes[shape].NPQ;
            ਅਚਲ uint32_t aligned = ((OC  % BS_K   == 0) &&
                                      (CRS % BS_CRS == 0) &&
                                      (NPQ % BS_NPQ == 0)) ? 1u : 0u;

            vk_conv3d_pipeline_state conv3d_pipeline_state(s0, s1, s2, p0, p1, p2, d0, d1, d2, KW, KH, KD, aligned);

            std::map<vk_conv3d_pipeline_state, vk_pipeline> *pipelines = ਖਾਲੀ_ਸੂਚਕ;
            ਜੇ (src0->type == GGML_TYPE_F32) {
                pipelines = &ctx->device->pipeline_conv3d_f32[shape];
            } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_F16) {
                pipelines = &ctx->device->pipeline_conv3d_f16_f32[shape];
            } ਨਹੀਂਤਾਂ {
                ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
            }

            vk_pipeline pipeline = ਖਾਲੀ_ਸੂਚਕ;

            {
                std::lock_guard<std::mutex> guard(ctx->device->compile_mutex);
                ਸਵੈਚਾਲਿਤ it = pipelines->find(conv3d_pipeline_state);
                ਜੇ (it != pipelines->end()) {
                    pipeline = it->second;
                } ਨਹੀਂਤਾਂ {
                    (*pipelines)[conv3d_pipeline_state] = pipeline = std::make_shared<vk_pipeline_struct>();
                }
            }

            ਵਾਪਸ pipeline;
        }
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    ਹਾਲਤ GGML_OP_ADD1:
        ਜੇ (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
            ਵਾਪਸ ctx->device->pipeline_add1_f16_f16;
        }
        ਜੇ (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
            ਵਾਪਸ ctx->device->pipeline_add1_f16_f32;
        }
        ਜੇ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ਵਾਪਸ ctx->device->pipeline_add1_f32_f32;
        }
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    ਹਾਲਤ GGML_OP_ARANGE:
        ਜੇ (dst->type == GGML_TYPE_F32) {
            ਵਾਪਸ ctx->device->pipeline_arange_f32;
        }
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    ਹਾਲਤ GGML_OP_FILL:
        ਜੇ (dst->type == GGML_TYPE_F32) {
            ਵਾਪਸ ctx->device->pipeline_fill_f32;
        }
        ਜੇ (dst->type == GGML_TYPE_F16) {
            ਵਾਪਸ ctx->device->pipeline_fill_f16;
        }
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    ਮੂਲ:
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }

    GGML_ਅਣਵਰਤਿਆ(src2);
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ PC>
ਸਥਿਰ ਖਾਲੀ ggml_vk_op_f32(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ਅਚਲ ggml_tensor * src2, ਅਚਲ ggml_tensor * src3, ggml_tensor * dst, ggml_op op, PC&& pc, vk_pipeline pipeline_override = ਖਾਲੀ_ਸੂਚਕ) {
    VK_LOG_DEBUG("ggml_vk_op_f32((" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    ਜੇ (src1 != ਖਾਲੀ_ਸੂਚਕ) {
        std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    }
    ਜੇ (src2 != ਖਾਲੀ_ਸੂਚਕ) {
        std::cerr << "), (" << src2 << ", name=" << src2->name << ", type=" << src2->type << ", ne0=" << src2->ne[0] << ", ne1=" << src2->ne[1] << ", ne2=" << src2->ne[2] << ", ne3=" << src2->ne[3] << ", nb0=" << src2->nb[0] << ", nb1=" << src2->nb[1] << ", nb2=" << src2->nb[2] << ", nb3=" << src2->nb[3];
    }
    ਜੇ (src3 != ਖਾਲੀ_ਸੂਚਕ) {
        std::cerr << "), (" << src3 << ", name=" << src3->name << ", type=" << src3->type << ", ne0=" << src3->ne[0] << ", ne1=" << src3->ne[1] << ", ne2=" << src3->ne[2] << ", ne3=" << src3->ne[3] << ", nb0=" << src3->nb[0] << ", nb1=" << src3->nb[1] << ", nb2=" << src3->nb[2] << ", nb3=" << src3->nb[3];
    }
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::cerr << "), " << ggml_op_name(op) << ")");
    GGML_ASSERT(op == GGML_OP_GET_ਪੰਕਤੀਆਂ || op == GGML_OP_CPY || op == GGML_OP_CONCAT || (!ggml_is_quantized(src0->type) && (src1 == ਖਾਲੀ_ਸੂਚਕ || !ggml_is_quantized(src1->type))));  // NOLINT
    GGML_ASSERT(dst->ਬਫਰ != ਖਾਲੀ_ਸੂਚਕ);
    ਅਚਲ uint64_t ne00 = src0->ne[0];
    ਅਚਲ uint64_t ne01 = src0->ne[1];
    ਅਚਲ uint64_t ne02 = src0->ne[2];
    ਅਚਲ uint64_t ne03 = src0->ne[3];

    ਅਚਲ ਤਾਰਕਿਕ use_src1 = src1 != ਖਾਲੀ_ਸੂਚਕ;
    ਅਚਲ uint64_t ne10 = use_src1 ? src1->ne[0] : 0;
    ਅਚਲ uint64_t ne11 = use_src1 ? src1->ne[1] : 0;
    ਅਚਲ uint64_t ne12 = use_src1 ? src1->ne[2] : 0;
    ਅਚਲ uint64_t ne13 = use_src1 ? src1->ne[3] : 0;

    ਅਚਲ ਤਾਰਕਿਕ use_src2 = src2 != ਖਾਲੀ_ਸੂਚਕ;
    ਅਚਲ ਤਾਰਕਿਕ use_src3 = src3 != ਖਾਲੀ_ਸੂਚਕ;

    init_pushconst_fastdiv(pc);

    vk_pipeline pipeline;
    ਜੇ (pipeline_override) {
        pipeline = pipeline_override;
    } ਨਹੀਂਤਾਂ {
        pipeline = ggml_vk_op_get_pipeline(ctx, src0, src1, src2, dst, op);
    }

    ਜੇ (pipeline == ਖਾਲੀ_ਸੂਚਕ) {
        std::cerr << "ggml_vulkan: Error: Missing op: " << ggml_op_name(op) << " for " << ggml_type_name(src0->type);
        ਜੇ (src1 != ਖਾਲੀ_ਸੂਚਕ) {
            std::cerr << " and " << ggml_type_name(src1->type);
        }
        std::cerr << " to " << ggml_type_name(dst->type) << std::ਸਤਰ_ਅੰਤ;
        GGML_ABORT("fatal error");
    }

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer src0_buf = ggml_vk_tensor_subbuffer(ctx, src0, ਸੱਚ);
    vk_subbuffer src1_buf = use_src1 ? ggml_vk_tensor_subbuffer(ctx, src1, ਸੱਚ) : vk_subbuffer{};
    vk_subbuffer src2_buf = use_src2 ? ggml_vk_tensor_subbuffer(ctx, src2, ਸੱਚ) : vk_subbuffer{};
    vk_subbuffer src3_buf = use_src3 ? ggml_vk_tensor_subbuffer(ctx, src3, ਸੱਚ) : vk_subbuffer{};
    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst, ਸੱਚ);

    // Compute misalignment offset for descriptors and store it in in push constants.
    init_pushconst_tensor_offsets(ctx, pc, src0, src1, src2, src3, dst);

    std::array<uint32_t, 3> elements;

    ਚੋਣ (op) {
    ਹਾਲਤ GGML_OP_NORM:
    ਹਾਲਤ GGML_OP_RMS_NORM_ਪਿੱਠ:
    ਹਾਲਤ GGML_OP_L2_NORM:
    ਹਾਲਤ GGML_OP_SOFT_MAX:
    ਹਾਲਤ GGML_OP_SOFT_MAX_ਪਿੱਠ:
    ਹਾਲਤ GGML_OP_ਜੋੜ_ਪੰਕਤੀਆਂ:
    ਹਾਲਤ GGML_OP_CUMSUM:
    ਹਾਲਤ GGML_OP_ਔਸਤ:
    ਹਾਲਤ GGML_OP_ARGMAX:
        {
            ਅਚਲ uint32_t nr = ggml_nrows(src0);
            ਜੇ (nr > 262144) {
                elements = { 512, 512, CEIL_DIV(nr, 262144) };
            } ਨਹੀਂਤਾਂ ਜੇ (nr > 512) {
                elements = { 512, CEIL_DIV(nr, 512), 1 };
            } ਨਹੀਂਤਾਂ {
                elements = { nr, 1, 1 };
            }
        } ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_SOLVE_TRI:
        {
            uint32_t nr = (uint32_t)(ne02 * ne03);
            ਜੇ (nr > 262144) {
                elements = { 512, 512, CEIL_DIV(nr, 262144) };
            } ਨਹੀਂਤਾਂ ਜੇ (nr > 512) {
                elements = { 512, CEIL_DIV(nr, 512), 1 };
            } ਨਹੀਂਤਾਂ {
                elements = { nr, 1, 1 };
            }
        }
        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_RMS_NORM:
        ਜੇ (ctx->do_ਜੋੜੋ_rms_partials) {
            // Run one element per thread, 128 threads per workgroup
            elements = { (uint32_t)CEIL_DIV(ne00, 128), 1, 1 };
        } ਨਹੀਂਤਾਂ {
            elements = { (uint32_t)ne01, (uint32_t)ne02, (uint32_t)ne03 };
            elements[1] = std::min(elements[1], ctx->device->properties.limits.max_Compute_Work_ਸਮੂਹ_ਗਿਣਤੀ[1]);
            elements[2] = std::min(elements[2], ctx->device->properties.limits.max_Compute_Work_ਸਮੂਹ_ਗਿਣਤੀ[2]);
        }
        ਰੋਕੋ;

    ਹਾਲਤ GGML_OP_ਜੋੜ:
        // We use GGML_OP_SUM_ROWS with 1 row.
        elements = { 1, 1, 1 };
        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_ਸਮੂਹ_NORM:
        {
            ਅਚਲ uint32_t ਅੰਕ_groups = dst->op_params[0];
            elements = { ਅੰਕ_groups * (uint32_t)src0->ne[3], 1, 1 };
        } ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_DIAG_ਨਕਾਬ_INF:
        elements = { (uint32_t)ggml_nrows(src0), (uint32_t)ne00, 1 };
        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_ROPE:
    ਹਾਲਤ GGML_OP_ROPE_ਪਿੱਠ:
        {
            uint32_t nrows = (uint32_t)ggml_nrows(src0);
            uint32_t z = 1;
            ਜੇ (nrows > ctx->device->properties.limits.max_Compute_Work_ਸਮੂਹ_ਗਿਣਤੀ[0]) {
                z = CEIL_DIV(nrows, 32768);
                nrows = 32768;
            }
            elements = { nrows, (uint32_t)ne00, z };

        } ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_GET_ਪੰਕਤੀਆਂ:
        elements = { (uint32_t)ne00, (uint32_t)ne10, (uint32_t)(ne11 * ne12) };
        elements[1] = std::min(elements[1], ctx->device->properties.limits.max_Compute_Work_ਸਮੂਹ_ਗਿਣਤੀ[1]);
        elements[2] = std::min(elements[2], ctx->device->properties.limits.max_Compute_Work_ਸਮੂਹ_ਗਿਣਤੀ[2]);
        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_GET_ਪੰਕਤੀਆਂ_ਪਿੱਠ:
        elements = { (uint32_t)dst->ne[0], (uint32_t)dst->ne[1], 1 };
        elements[1] = std::min(elements[1], ctx->device->properties.limits.max_Compute_Work_ਸਮੂਹ_ਗਿਣਤੀ[1]);
        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_ARGSORT:
        GGML_ASSERT(0);
        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_IM2_ਖਾਨਾ:
        {
            ਅਚਲ ਤਾਰਕਿਕ is_2D = dst->op_params[6] == 1;

            ਅਚਲ uint32_t IC = src1->ne[is_2D ? 2 : 1];

            ਅਚਲ uint32_t KH = is_2D ? src0->ne[1] : 1;
            ਅਚਲ uint32_t KW =         src0->ne[0];

            ਅਚਲ uint32_t OH = is_2D ? dst->ne[2] : 1;
            ਅਚਲ uint32_t OW =         dst->ne[1];

            ਅਚਲ uint32_t batch = src1->ne[is_2D ? 3 : 2];

            ਅਚਲ uint32_t CHW = IC * KH * KW;
            // Cap X workgroups to limit concurrent IC channel reads.
            // The shader loops over X to cover the full CHW dimension.
            // AMD prefers a lower limit
            ਅਚਲ uint32_t min_cap = ctx->device->vendor_id == VK_VENDOR_ID_AMD ? 512u : 4096u;
            ਅਚਲ uint32_t x_elements = std::min(CHW, std::max(min_cap, OW * KH * KW));
            elements = { x_elements, OW, OH * batch };
            elements[1] = std::min(elements[1], ctx->device->properties.limits.max_Compute_Work_ਸਮੂਹ_ਗਿਣਤੀ[1]);
            elements[2] = std::min(elements[2], ctx->device->properties.limits.max_Compute_Work_ਸਮੂਹ_ਗਿਣਤੀ[2]);
        } ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_IM2_ਖਾਨਾ_3_D:
        {
            ਅਚਲ uint32_t IC = ((ਅਚਲ uint32_t *)(dst->op_params))[9];

            ਅਚਲ uint32_t N  = ne13 / IC;

            ਅਚਲ uint32_t KD = ne02;
            ਅਚਲ uint32_t KH = ne01;
            ਅਚਲ uint32_t KW = ne00;

            ਅਚਲ uint32_t OD = dst->ne[3] / N;
            ਅਚਲ uint32_t OH = dst->ne[2];
            ਅਚਲ uint32_t OW = dst->ne[1];

            ਅਚਲ uint32_t IC_KD_KH_KW = IC*KD*KH*KW;
            ਅਚਲ uint32_t N_OD_OH = N*OD*OH;

            elements = { IC_KD_KH_KW, OW, N_OD_OH };
            elements[2] = std::min(elements[2], ctx->device->properties.limits.max_Compute_Work_ਸਮੂਹ_ਗਿਣਤੀ[2]);
        } ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_TIMESTEP_EMBEDDING:
        {
            ਅਚਲ uint32_t dim = dst->op_params[0];
            uint32_t half_ceil = (dim + 1) / 2;
            elements = { half_ceil, (uint32_t)src0->ne[0], 1 };
        } ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_CONV_TRANSPOSE_1D:
        {
            elements = {uint32_t(src0->ne[1]), 1, 1}; // parallelize in {Cout, 1, 1}
        } ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_COL2IM_1D:
        {
            elements = { uint32_t(dst->ne[0]), uint32_t(dst->ne[1]), 1 };
        } ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_POOL_1D:
        {
            ਅਚਲ uint32_t N = dst->ne[3] * dst->ne[2];
            ਅਚਲ uint32_t OC = dst->ne[1];
            ਅਚਲ uint32_t OL = dst->ne[0];
            elements = { N * OC * OL, 1, 1};
        } ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_POOL_2D:
        {
            ਅਚਲ uint32_t outputs = ggml_nelements(dst);
            ਅਚਲ uint32_t outputs_ਪ੍ਰਤੀ_workgroup = pipeline == ctx->device->pipeline_pool2d_subgroup_f32 ? 128 / ctx->device->subgroup_ਮਾਪ : 512;
            ਅਚਲ uint32_t workgroups = CEIL_DIV(outputs, outputs_ਪ੍ਰਤੀ_workgroup);
            ਅਚਲ uint32_t groups_y = CEIL_DIV(workgroups, ctx->device->properties.limits.max_Compute_Work_ਸਮੂਹ_ਗਿਣਤੀ[0]);
            ਅਚਲ uint32_t groups_x = CEIL_DIV(workgroups, groups_y);
            elements = { groups_x, groups_y, 1 };
        } ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_CONV_2D:
    ਹਾਲਤ GGML_OP_CONV_TRANSPOSE_2D:
        ਜੇ constexpr (std::is_same_v<PC, vk_op_conv2d_ਧੱਕੋ_constants>) {
            ਅਚਲ uint32_t NPQ = pc.N * pc.OH * pc.OW;
            ਅਚਲ vk_conv_shapes shape = ggml_vk_conv_select_shape(ctx, pc.Cout, NPQ);
            ਅਚਲ uint32_t NPQ_ਖੰਡਾਂ = CEIL_DIV(NPQ, vk_conv_ਖੰਡ_sizes[shape].NPQ);

            elements = { pc.Cout, NPQ_ਖੰਡਾਂ, 1 };
            ਜੇ (elements[1] > 512) {
                elements[2] = CEIL_DIV(elements[1], 512);
                elements[1] = 512;
            }
        } ਨਹੀਂਤਾਂ {
            GGML_ABORT("invalid push constant type for CONV_2D");
        }
        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_CONV_3D:
        ਜੇ constexpr (std::is_same_v<PC, vk_op_conv3d_ਧੱਕੋ_constants>) {
            ਅਚਲ uint32_t NPQ = pc.N * pc.OD * pc.OH * pc.OW;
            ਅਚਲ vk_conv_shapes shape = ggml_vk_conv_select_shape(ctx, pc.OC, NPQ);
            ਅਚਲ uint32_t NPQ_ਖੰਡਾਂ = CEIL_DIV(NPQ, vk_conv_ਖੰਡ_sizes[shape].NPQ);

            elements = { pc.OC, NPQ_ਖੰਡਾਂ, 1 };
            ਜੇ (elements[1] > 512) {
                elements[2] = CEIL_DIV(elements[1], 512);
                elements[1] = 512;
            }
        } ਨਹੀਂਤਾਂ {
            GGML_ABORT("invalid push constant type for CONV_3D");
        }
        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_ਜੋੜੋ:
    ਹਾਲਤ GGML_OP_SUB:
    ਹਾਲਤ GGML_OP_DIV:
    ਹਾਲਤ GGML_OP_MUL:
    ਹਾਲਤ GGML_OP_ADD1:
    ਹਾਲਤ GGML_OP_ਬਾਹਰਲਾ_PROD:
    ਹਾਲਤ GGML_OP_ARANGE:
    ਹਾਲਤ GGML_OP_FILL:
    ਹਾਲਤ GGML_OP_SCALE:
    ਹਾਲਤ GGML_OP_SQR:
    ਹਾਲਤ GGML_OP_SQRT:
    ਹਾਲਤ GGML_OP_SIN:
    ਹਾਲਤ GGML_OP_COS:
    ਹਾਲਤ GGML_OP_LOG:
    ਹਾਲਤ GGML_OP_TRI:
    ਹਾਲਤ GGML_OP_DIAG:
    ਹਾਲਤ GGML_OP_CLAMP:
    ਹਾਲਤ GGML_OP_LEAKY_RELU:
    ਹਾਲਤ GGML_OP_PAD:
    ਹਾਲਤ GGML_OP_PAD_REFLECT_1D:
    ਹਾਲਤ GGML_OP_ROLL:
    ਹਾਲਤ GGML_OP_REPEAT:
    ਹਾਲਤ GGML_OP_REPEAT_ਪਿੱਠ:
    ਹਾਲਤ GGML_OP_CPY:
    ਹਾਲਤ GGML_OP_CONCAT:
    ਹਾਲਤ GGML_OP_UPSCALE:
    ਹਾਲਤ GGML_OP_UNARY:
    ਹਾਲਤ GGML_OP_GLU:
    ਹਾਲਤ GGML_OP_CONV_2D_DW:
        {
            uint32_t ne = ggml_nelements(dst);
            ਜੇ (op == GGML_OP_CPY && ggml_is_quantized(src0->type) && ggml_is_quantized(dst->type)) {
                // Convert from number of logical elements to 2- or 4-byte units.
                ne /= ggml_blck_ਮਾਪ(src0->type);
                ਜੇ ((ggml_type_ਮਾਪ(src0->type) % 4) == 0) {
                    ne *= ggml_type_ਮਾਪ(src0->type) / 4;
                } ਨਹੀਂਤਾਂ {
                    ne *= ggml_type_ਮਾਪ(src0->type) / 2;
                }
            }
            ਜੇ (op == GGML_OP_CONCAT && ggml_is_quantized(dst->type)) {
                ne = ne / ggml_blck_ਮਾਪ(dst->type) * ggml_type_ਮਾਪ(dst->type) / ggml_vk_concat_unit_ਮਾਪ(dst->type);
            }
            // copy_to_quant has block size of 32, and each thread does QUANT_K elements.
            // Splitting into 512x512xZ wouldn't work well since each workgroup does 1024 elements.
            // So divide by block size here before splitting into 512x512 groups.
            ਜੇ (op == GGML_OP_CPY && !ggml_is_quantized(src0->type) && ggml_is_quantized(dst->type)) {
                ne = CEIL_DIV(ne, ggml_blck_ਮਾਪ(dst->type));
            }
            ਜੇ (ne > 262144) {
                elements = { 512, 512, CEIL_DIV(ne, 262144) };
            } ਨਹੀਂਤਾਂ ਜੇ (ne > 512) {
                elements = { 512, CEIL_DIV(ne, 512), 1 };
            } ਨਹੀਂਤਾਂ {
                elements = { ne, 1, 1 };
            }

            ਜੇ (pipeline == ctx->device->pipeline_cpy_transpose_02_32 ||
                pipeline == ctx->device->pipeline_cpy_transpose_02_16) {
                // 32x32 tiles over dims 0 and 2; dim1 and dim3 are the batch
                elements[0] = (uint32_t)CEIL_DIV(dst->ne[0], 32);
                elements[1] = (uint32_t)CEIL_DIV(dst->ne[2], 32);
                elements[2] = (uint32_t)(dst->ne[1]*dst->ne[3]);
                elements[0] = std::min(elements[0], ctx->device->properties.limits.max_Compute_Work_ਸਮੂਹ_ਗਿਣਤੀ[0]);
                elements[1] = std::min(elements[1], ctx->device->properties.limits.max_Compute_Work_ਸਮੂਹ_ਗਿਣਤੀ[1]);
                elements[2] = std::min(elements[2], ctx->device->properties.limits.max_Compute_Work_ਸਮੂਹ_ਗਿਣਤੀ[2]);
            } ਨਹੀਂਤਾਂ ਜੇ (pipeline == ctx->device->pipeline_cpy_transpose_32 ||
                pipeline == ctx->device->pipeline_cpy_transpose_16) {
                // 32x32 tiles
                elements[0] = (uint32_t)CEIL_DIV(dst->ne[0], 32);
                elements[1] = (uint32_t)CEIL_DIV(dst->ne[1], 32);
                elements[2] = (uint32_t)(dst->ne[2]*dst->ne[3]);
                elements[0] = std::min(elements[0], ctx->device->properties.limits.max_Compute_Work_ਸਮੂਹ_ਗਿਣਤੀ[0]);
                elements[1] = std::min(elements[1], ctx->device->properties.limits.max_Compute_Work_ਸਮੂਹ_ਗਿਣਤੀ[1]);
                elements[2] = std::min(elements[2], ctx->device->properties.limits.max_Compute_Work_ਸਮੂਹ_ਗਿਣਤੀ[2]);
            }
        } ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_ਜੋੜੋ_ID:
        {
            elements = { (uint32_t)ne01, (uint32_t)ne02, 1 };
        } ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_ਸੈੱਟ_ਪੰਕਤੀਆਂ:
        {
            uint32_t ne = ggml_nelements(src0);
            ਜੇ (ggml_is_quantized(dst->type)) {
                // quants run 32 threads each doing QUANT_K elements
                ne = CEIL_DIV(ne, 32 * ggml_blck_ਮਾਪ(dst->type));
            } ਨਹੀਂਤਾਂ {
                // scalar types do one element per thread, running 512 threads
                ne = CEIL_DIV(ne, 512);
            }
            ਜੇ (ne > 262144) {
                elements = { 512, 512, CEIL_DIV(ne, 262144) };
            } ਨਹੀਂਤਾਂ ਜੇ (ne > 512) {
                elements = { 512, CEIL_DIV(ne, 512), 1 };
            } ਨਹੀਂਤਾਂ {
                elements = { ne, 1, 1 };
            }
        }
        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_SSM_CONV:
        {
            ਅਚਲ uint32_t nr  = src0->ne[1];
            ਅਚਲ uint32_t n_t = dst->ne[1];
            ਅਚਲ uint32_t n_s = dst->ne[2];
            elements = { nr, n_t, n_s };
        }
        ਰੋਕੋ;
    ਮੂਲ:
        elements = { (uint32_t)ggml_nelements(src0), 1, 1 };
        ਰੋਕੋ;
    }

    ਜੇ (op == GGML_OP_ਜੋੜੋ || op == GGML_OP_RMS_NORM) {
        vk_subbuffer a_buf = src0_buf;
        ਜੇ (ctx->do_ਜੋੜੋ_rms_partials) {
            a_buf = ggml_vk_subbuffer(ctx, ctx->prealloc_ਜੋੜੋ_rms_partials, ctx->prealloc_ਮਾਪ_ਜੋੜੋ_rms_partials_ਵਿੱਥ);
        }
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
            { src0_buf, src1_buf, dst_buf, a_buf }, pc, elements);
    } ਨਹੀਂਤਾਂ ਜੇ (op == GGML_OP_GLU) {
        // Empty src1 is possible in glu, but the shader needs a buffer
        vk_subbuffer subbuf1 = use_src1 ? src1_buf : src0_buf;
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, subbuf1, dst_buf }, pc, elements);
    } ਨਹੀਂਤਾਂ ਜੇ (op == GGML_OP_SOFT_MAX) {
        // Empty src1 and src2 is possible in soft_max, but the shader needs a buffer
        vk_subbuffer subbuf1 = use_src1 ? src1_buf : src0_buf;
        vk_subbuffer subbuf2 = use_src2 ? src2_buf : src0_buf;
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, subbuf1, subbuf2, dst_buf }, pc, elements);
    } ਨਹੀਂਤਾਂ ਜੇ (op == GGML_OP_ROPE || op == GGML_OP_ROPE_ਪਿੱਠ) {
        // Empty src2 and src3 is possible in rope, but the shader needs a buffer
        vk_subbuffer subbuf2 = use_src2 ? src2_buf : src0_buf;
        vk_subbuffer subbuf3 = use_src3 ? src3_buf : src0_buf;
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, subbuf2, dst_buf, subbuf3 }, pc, elements);
    } ਨਹੀਂਤਾਂ ਜੇ (op == GGML_OP_IM2_ਖਾਨਾ || op == GGML_OP_IM2_ਖਾਨਾ_3_D) {
        ਜੇ (ctx->device->shader_int64 && ctx->device->ਬਫਰ_device_ਪਤਾ) {
            // buffer device address path doesn't use dst buffer
            dst_buf.size = 1;
        }
        // im2col uses only src1 and dst buffers
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src1_buf, dst_buf }, pc, elements);
    } ਨਹੀਂਤਾਂ ਜੇ (op == GGML_OP_ਗਿਣਤੀ_EQUAL) {
        // count_equal assumes that destination buffer is initialized with zeroes
        ggml_vk_ਬਫਰ_memset_async(subctx, dst_buf.ਬਫਰ, dst_buf.ਵਿੱਥ, 0, dst_buf.size);
        ggml_vk_sync_buffers(ctx, subctx);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, dst_buf }, pc, elements);
    } ਨਹੀਂਤਾਂ ਜੇ (op == GGML_OP_OPT_ਕਦਮ_SGD) {
        // OPT_STEP_SGD works on src0, it does not need dst
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, src2_buf }, pc, elements);
    } ਨਹੀਂਤਾਂ ਜੇ (use_src3) {
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, src2_buf, src3_buf, dst_buf }, pc, elements);
    } ਨਹੀਂਤਾਂ ਜੇ (use_src2) {
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, src2_buf, dst_buf }, pc, elements);
    } ਨਹੀਂਤਾਂ ਜੇ (use_src1) {
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, dst_buf }, pc, elements);
    } ਨਹੀਂਤਾਂ {
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, dst_buf }, pc, elements);
    }
}

ਖਾਲੀ ggml_vk_get_ਪੰਕਤੀਆਂ(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ggml_tensor * dst) {
    ਅਚਲ uint32_t src0_type_ਮਾਪ = ggml_type_ਮਾਪ(src0->type);
    ਅਚਲ uint32_t src1_type_ਮਾਪ = ggml_type_ਮਾਪ(src1->type);
    ਅਚਲ uint32_t dst_type_ਮਾਪ = ggml_type_ਮਾਪ(dst->type);

    ggml_vk_op_f32<vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants>(ctx, subctx, src0, src1, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, dst, GGML_OP_GET_ਪੰਕਤੀਆਂ, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_ਮਾਪ, (uint32_t)src0->nb[1] / src0_type_ਮਾਪ, (uint32_t)src0->nb[2] / src0_type_ਮਾਪ, (uint32_t)src0->nb[3] / src0_type_ਮਾਪ,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_ਮਾਪ, (uint32_t)src1->nb[1] / src1_type_ਮਾਪ, (uint32_t)src1->nb[2] / src1_type_ਮਾਪ, (uint32_t)src1->nb[3] / src1_type_ਮਾਪ,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_ਮਾਪ, (uint32_t) dst->nb[1] /  dst_type_ਮਾਪ, (uint32_t) dst->nb[2] /  dst_type_ਮਾਪ, (uint32_t) dst->nb[3] /  dst_type_ਮਾਪ,
        0,
        0.0f, 0.0f, 0,
    });
}

ਖਾਲੀ ggml_vk_get_ਪੰਕਤੀਆਂ_ਪਿੱਠ(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ggml_tensor * dst) {
    ਅਚਲ uint32_t src0_type_ਮਾਪ = ggml_type_ਮਾਪ(src0->type);
    ਅਚਲ uint32_t src1_type_ਮਾਪ = ggml_type_ਮਾਪ(src1->type);
    ਅਚਲ uint32_t dst_type_ਮਾਪ = ggml_type_ਮਾਪ(dst->type);

    ggml_vk_op_f32<vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants>(ctx, subctx, src0, src1, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, dst, GGML_OP_GET_ਪੰਕਤੀਆਂ_ਪਿੱਠ, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2], (uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_ਮਾਪ, (uint32_t)src0->nb[1] / src0_type_ਮਾਪ, (uint32_t)src0->nb[2] / src0_type_ਮਾਪ, (uint32_t)src0->nb[3] / src0_type_ਮਾਪ,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2], (uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_ਮਾਪ, (uint32_t)src1->nb[1] / src1_type_ਮਾਪ, (uint32_t)src1->nb[2] / src1_type_ਮਾਪ, (uint32_t)src1->nb[3] / src1_type_ਮਾਪ,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2], (uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_ਮਾਪ, (uint32_t) dst->nb[1] /  dst_type_ਮਾਪ, (uint32_t) dst->nb[2] /  dst_type_ਮਾਪ, (uint32_t) dst->nb[3] /  dst_type_ਮਾਪ,
        0,
        0.0f, 0.0f, 0,
    });
}

ਖਾਲੀ ggml_vk_acc(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ggml_tensor * dst) {
    ਅਚਲ uint32_t src0_type_ਮਾਪ = ggml_type_ਮਾਪ(src0->type);
    ਅਚਲ uint32_t src1_type_ਮਾਪ = ggml_type_ਮਾਪ(src1->type);
    ਅਚਲ uint32_t dst_type_ਮਾਪ = ggml_type_ਮਾਪ(dst->type);

    ਪੂਰਨ nb1 = dst->op_params[0] / src0_type_ਮਾਪ; // 4 bytes of float32
    ਪੂਰਨ nb2 = dst->op_params[1] / src0_type_ਮਾਪ; // 4 bytes of float32
    ਪੂਰਨ nb3 = dst->op_params[2] / src0_type_ਮਾਪ; // 4 bytes of float32
    ਪੂਰਨ ਵਿੱਥ = dst->op_params[3] / src0_type_ਮਾਪ; // offset in bytes

    ggml_vk_op_f32<vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants>(ctx, subctx, src0, src1, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, dst, dst->op, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_ਮਾਪ, (uint32_t)nb1, (uint32_t)nb2, (uint32_t)nb3,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_ਮਾਪ, (uint32_t)src1->nb[1] / src1_type_ਮਾਪ, (uint32_t)src1->nb[2] / src1_type_ਮਾਪ, (uint32_t)src1->nb[3] / src1_type_ਮਾਪ,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_ਮਾਪ, (uint32_t)nb1, (uint32_t)nb2, (uint32_t)nb3,
        0,
        0.0f, 0.0f, ਵਿੱਥ,
    });
}

ਖਾਲੀ ggml_vk_multi_ਜੋੜੋ(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_cgraph * cgraph, ਪੂਰਨ ਗੰਢ_idx) {
    ਅਚਲ ggml_tensor *ਪਹਿਲਾ_ਗੰਢ = cgraph->nodes[ਗੰਢ_idx];
    ਅਚਲ ggml_tensor *dst = cgraph->nodes[ਗੰਢ_idx + ctx->ਅੰਕ_additional_fused_ops];

    // Make a list of all the tensors used by the op.
    // Last element of the list is the dest tensor.
    ਅਚਲ ggml_tensor *tensors[MAX_PARAMETER_ਗਿਣਤੀ];
    uint32_t ਅੰਕ_srcs = ctx->ਅੰਕ_additional_fused_ops + 2;
    uint32_t ਅੰਕ_tensors = ਅੰਕ_srcs + 1;
    GGML_ASSERT(ਅੰਕ_tensors + ctx->do_ਜੋੜੋ_rms_partials <= MAX_PARAMETER_ਗਿਣਤੀ);

    tensors[0] = ਪਹਿਲਾ_ਗੰਢ->src[0];
    tensors[1] = ਪਹਿਲਾ_ਗੰਢ->src[1];
    ਲਈ (int32_t i = 0; i < ctx->ਅੰਕ_additional_fused_ops; ++i) {
        // check whether the previous result is src[0] or src[1]
        ਜੇ (cgraph->nodes[ਗੰਢ_idx + i] == cgraph->nodes[ਗੰਢ_idx + i + 1]->src[0]) {
            tensors[i+2] = cgraph->nodes[ਗੰਢ_idx + i + 1]->src[1];
        } ਨਹੀਂਤਾਂ {
            tensors[i+2] = cgraph->nodes[ਗੰਢ_idx + i + 1]->src[0];
        }
    }
    tensors[ਅੰਕ_srcs] = dst;

    vk_op_multi_ਜੋੜੋ_ਧੱਕੋ_constants pc;
    pc.ne20 = (uint32_t)dst->ne[0];
    pc.ne21 = (uint32_t)dst->ne[1];
    pc.ne22 = (uint32_t)dst->ne[2];
    pc.ne23 = (uint32_t)dst->ne[3];

    ਲਈ (uint32_t i = 0; i < ਅੰਕ_tensors; ++i) {
        ਅਚਲ ggml_tensor *t = tensors[i];
        pc.nb[i][0] = (uint32_t)t->nb[0] / ਆਕਾਰ(ਦਸ਼ਮਲਵ);
        pc.nb[i][1] = (uint32_t)t->nb[1] / ਆਕਾਰ(ਦਸ਼ਮਲਵ);
        pc.nb[i][2] = (uint32_t)t->nb[2] / ਆਕਾਰ(ਦਸ਼ਮਲਵ);
        pc.nb[i][3] = (uint32_t)t->nb[3] / ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    }
    pc.rms_partials = ctx->do_ਜੋੜੋ_rms_partials;

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, tensors[0], tensors[1], ਖਾਲੀ_ਸੂਚਕ, dst, dst->op);

    ਜੇ (pipeline == ਖਾਲੀ_ਸੂਚਕ) {
        std::cerr << "ggml_vulkan: Error: Missing multi_add";
        GGML_ABORT("fatal error");
    }

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    ggml_backend_vk_ਬਫਰ_context * buf_ctx[MAX_PARAMETER_ਗਿਣਤੀ];
    vk_ਬਫਰ buf[MAX_PARAMETER_ਗਿਣਤੀ];
    size_t ਵਿੱਥ[MAX_PARAMETER_ਗਿਣਤੀ];
    ਤਾਰਕਿਕ uma[MAX_PARAMETER_ਗਿਣਤੀ];

    ਲਈ (uint32_t i = 0; i < ਅੰਕ_tensors; ++i) {
        buf_ctx[i] = (ggml_backend_vk_ਬਫਰ_context *)tensors[i]->ਬਫਰ->context;
        buf[i] = ਖਾਲੀ_ਸੂਚਕ;
        ਵਿੱਥ[i] = 0;
        uma[i] = ਝੂਠ;

        ਜੇ (ctx->device->uma) {
            ggml_vk_host_get(ctx->device, tensors[i]->ਅੰਕੜੇ, buf[i], ਵਿੱਥ[i]);
            uma[i] = buf[i] != ਖਾਲੀ_ਸੂਚਕ;
        }
        ਜੇ (!uma[i]) {
            buf[i] = buf_ctx[i]->dev_ਬਫਰ;
            ਵਿੱਥ[i] = vk_tensor_ਵਿੱਥ(tensors[i]) + tensors[i]->view_offs;
        }
        GGML_ASSERT(buf[i] != ਖਾਲੀ_ਸੂਚਕ);
    }
    // If any remaining descriptors are unused, just point them at src[0]
    ਲਈ (uint32_t i = ਅੰਕ_tensors; i < MAX_PARAMETER_ਗਿਣਤੀ; ++i) {
        buf[i] = buf[0];
        ਵਿੱਥ[i] = 0;
    }
    ਜੇ (ctx->do_ਜੋੜੋ_rms_partials) {
        buf[ਅੰਕ_tensors] = ctx->prealloc_ਜੋੜੋ_rms_partials;
        ਵਿੱਥ[ਅੰਕ_tensors] = ctx->prealloc_ਮਾਪ_ਜੋੜੋ_rms_partials_ਵਿੱਥ;
    }

    std::array<uint32_t, 3> elements;

    uint32_t ne = ggml_nelements(dst);
    ਜੇ (ne > 262144) {
        elements = { 512, 512, CEIL_DIV(ne, 262144) };
    } ਨਹੀਂਤਾਂ ਜੇ (ne > 512) {
        elements = { 512, CEIL_DIV(ne, 512), 1 };
    } ਨਹੀਂਤਾਂ {
        elements = { ne, 1, 1 };
    }

    static_assert(MAX_PARAMETER_ਗਿਣਤੀ == 12);
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        {
            ggml_vk_subbuffer(ctx, buf[0], ਵਿੱਥ[0]),
            ggml_vk_subbuffer(ctx, buf[1], ਵਿੱਥ[1]),
            ggml_vk_subbuffer(ctx, buf[2], ਵਿੱਥ[2]),
            ggml_vk_subbuffer(ctx, buf[3], ਵਿੱਥ[3]),
            ggml_vk_subbuffer(ctx, buf[4], ਵਿੱਥ[4]),
            ggml_vk_subbuffer(ctx, buf[5], ਵਿੱਥ[5]),
            ggml_vk_subbuffer(ctx, buf[6], ਵਿੱਥ[6]),
            ggml_vk_subbuffer(ctx, buf[7], ਵਿੱਥ[7]),
            ggml_vk_subbuffer(ctx, buf[8], ਵਿੱਥ[8]),
            ggml_vk_subbuffer(ctx, buf[9], ਵਿੱਥ[9]),
            ggml_vk_subbuffer(ctx, buf[10], ਵਿੱਥ[10]),
            ggml_vk_subbuffer(ctx, buf[11], ਵਿੱਥ[11]),
        }, pc, elements);
}

ਖਾਲੀ ggml_vk_ਜੋੜੋ(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ggml_tensor * dst) {
    ਅਚਲ uint32_t src0_type_ਮਾਪ = ggml_type_ਮਾਪ(src0->type);
    ਅਚਲ uint32_t src1_type_ਮਾਪ = ggml_type_ਮਾਪ(src1->type);
    ਅਚਲ uint32_t dst_type_ਮਾਪ = ggml_type_ਮਾਪ(dst->type);

    ggml_vk_op_f32<vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants>(ctx, subctx, src0, src1, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, dst, GGML_OP_ਜੋੜੋ, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_ਮਾਪ, (uint32_t)src0->nb[1] / src0_type_ਮਾਪ, (uint32_t)src0->nb[2] / src0_type_ਮਾਪ, (uint32_t)src0->nb[3] / src0_type_ਮਾਪ,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_ਮਾਪ, (uint32_t)src1->nb[1] / src1_type_ਮਾਪ, (uint32_t)src1->nb[2] / src1_type_ਮਾਪ, (uint32_t)src1->nb[3] / src1_type_ਮਾਪ,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_ਮਾਪ, (uint32_t) dst->nb[1] /  dst_type_ਮਾਪ, (uint32_t) dst->nb[2] /  dst_type_ਮਾਪ, (uint32_t) dst->nb[3] /  dst_type_ਮਾਪ,
        0,
        0.0f, 0.0f, ctx->do_ਜੋੜੋ_rms_partials,
    });
}

ਖਾਲੀ ggml_vk_ਬਾਹਰਲਾ_prod(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ggml_tensor * dst) {
    ਅਚਲ uint32_t src0_type_ਮਾਪ = ggml_type_ਮਾਪ(src0->type);
    ਅਚਲ uint32_t src1_type_ਮਾਪ = ggml_type_ਮਾਪ(src1->type);
    ਅਚਲ uint32_t dst_type_ਮਾਪ = ggml_type_ਮਾਪ(dst->type);

    ggml_vk_op_f32<vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants>(ctx, subctx, src0, src1, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, dst, GGML_OP_ਬਾਹਰਲਾ_PROD, {
        (uint32_t)ggml_nelements(dst),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3],
        (uint32_t)src0->nb[0] / src0_type_ਮਾਪ, (uint32_t)src0->nb[1] / src0_type_ਮਾਪ, (uint32_t)src0->nb[2] / src0_type_ਮਾਪ, (uint32_t)src0->nb[3] / src0_type_ਮਾਪ,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3],
        (uint32_t)src1->nb[0] / src1_type_ਮਾਪ, (uint32_t)src1->nb[1] / src1_type_ਮਾਪ, (uint32_t)src1->nb[2] / src1_type_ਮਾਪ, (uint32_t)src1->nb[3] / src1_type_ਮਾਪ,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3],
        (uint32_t) dst->nb[0] /  dst_type_ਮਾਪ, (uint32_t) dst->nb[1] /  dst_type_ਮਾਪ, (uint32_t) dst->nb[2] /  dst_type_ਮਾਪ, (uint32_t) dst->nb[3] /  dst_type_ਮਾਪ,
        0,
        0.0f, 0.0f, 0,
    });
}

ਖਾਲੀ ggml_vk_sub(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ggml_tensor * dst) {
    ਅਚਲ uint32_t src0_type_ਮਾਪ = ggml_type_ਮਾਪ(src0->type);
    ਅਚਲ uint32_t src1_type_ਮਾਪ = ggml_type_ਮਾਪ(src1->type);
    ਅਚਲ uint32_t dst_type_ਮਾਪ = ggml_type_ਮਾਪ(dst->type);

    ggml_vk_op_f32<vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants>(ctx, subctx, src0, src1, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, dst, GGML_OP_SUB, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_ਮਾਪ, (uint32_t)src0->nb[1] / src0_type_ਮਾਪ, (uint32_t)src0->nb[2] / src0_type_ਮਾਪ, (uint32_t)src0->nb[3] / src0_type_ਮਾਪ,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_ਮਾਪ, (uint32_t)src1->nb[1] / src1_type_ਮਾਪ, (uint32_t)src1->nb[2] / src1_type_ਮਾਪ, (uint32_t)src1->nb[3] / src1_type_ਮਾਪ,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_ਮਾਪ, (uint32_t) dst->nb[1] /  dst_type_ਮਾਪ, (uint32_t) dst->nb[2] /  dst_type_ਮਾਪ, (uint32_t) dst->nb[3] /  dst_type_ਮਾਪ,
        0,
        0.0f, 0.0f, 0,
    });
}

ਖਾਲੀ ggml_vk_mul(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ggml_tensor * dst) {
    ਅਚਲ uint32_t src0_type_ਮਾਪ = ggml_type_ਮਾਪ(src0->type);
    ਅਚਲ uint32_t src1_type_ਮਾਪ = ggml_type_ਮਾਪ(src1->type);
    ਅਚਲ uint32_t dst_type_ਮਾਪ = ggml_type_ਮਾਪ(dst->type);

    ggml_vk_op_f32<vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants>(ctx, subctx, src0, src1, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, dst, GGML_OP_MUL, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_ਮਾਪ, (uint32_t)src0->nb[1] / src0_type_ਮਾਪ, (uint32_t)src0->nb[2] / src0_type_ਮਾਪ, (uint32_t)src0->nb[3] / src0_type_ਮਾਪ,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_ਮਾਪ, (uint32_t)src1->nb[1] / src1_type_ਮਾਪ, (uint32_t)src1->nb[2] / src1_type_ਮਾਪ, (uint32_t)src1->nb[3] / src1_type_ਮਾਪ,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_ਮਾਪ, (uint32_t) dst->nb[1] /  dst_type_ਮਾਪ, (uint32_t) dst->nb[2] /  dst_type_ਮਾਪ, (uint32_t) dst->nb[3] /  dst_type_ਮਾਪ,
        0,
        0.0f, 0.0f, 0,
    });
}

ਪੂਰਨ ggml_vk_unary_mul_op_ਸੂਚਕਾਂਕ(ggml_unary_op op) {
    ਚੋਣ (op) {
        ਹਾਲਤ GGML_UNARY_OP_GELU:     ਵਾਪਸ 0;
        ਹਾਲਤ GGML_UNARY_OP_SIGMOID:  ਵਾਪਸ 1;
        ਹਾਲਤ GGML_UNARY_OP_SILU:     ਵਾਪਸ 2;
        ਹਾਲਤ GGML_UNARY_OP_SOFTPLUS: ਵਾਪਸ 3;
        ਮੂਲ:                     ਵਾਪਸ -1;
    }
}

ਖਾਲੀ ggml_vk_unary_mul(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ਬਣਤਰ ggml_cgraph * cgraph, ਪੂਰਨ ਗੰਢ_idx) {
    ਅਚਲ ggml_tensor * unary = cgraph->nodes[ਗੰਢ_idx];
    ggml_tensor * mul = cgraph->nodes[ਗੰਢ_idx + 1];

    // unary on src1 that tiles into src0
    ਅਚਲ ਤਾਰਕਿਕ op_on_b = mul->src[1] == unary &&
                         !ggml_are_same_shape(unary->src[0], mul->src[0]) &&
                         ggml_can_repeat(unary, mul->src[0]);

    ਅਚਲ ggml_tensor * src0 = op_on_b ? mul->src[0] : unary->src[0];
    ਅਚਲ ggml_tensor * src1 = op_on_b ? unary->src[0] :
        ((mul->src[0] == unary) ? mul->src[1] : mul->src[0]);

    ਅਚਲ ਤਾਰਕਿਕ f16 = src0->type == GGML_TYPE_F16;
    ਅਚਲ ਤਾਰਕਿਕ norepeat = ggml_are_same_shape(src0, src1);
    ਅਚਲ ਪੂਰਨ oi = ggml_vk_unary_mul_op_ਸੂਚਕਾਂਕ(ggml_get_unary_op(unary));
    ਜੇ (oi < 0) {
        GGML_ABORT("fatal error");
    }
    vk_pipeline pipeline = ctx->device->pipeline_unary_mul[oi][f16][norepeat][op_on_b];

    ਅਚਲ uint32_t src0_type_ਮਾਪ = ggml_type_ਮਾਪ(src0->type);
    ਅਚਲ uint32_t src1_type_ਮਾਪ = ggml_type_ਮਾਪ(src1->type);
    ਅਚਲ uint32_t dst_type_ਮਾਪ = ggml_type_ਮਾਪ(mul->type);

    ggml_vk_op_f32<vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants>(ctx, subctx, src0, src1, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, mul, GGML_OP_UNARY, {
        (uint32_t)ggml_nelements(op_on_b ? mul : src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_ਮਾਪ, (uint32_t)src0->nb[1] / src0_type_ਮਾਪ, (uint32_t)src0->nb[2] / src0_type_ਮਾਪ, (uint32_t)src0->nb[3] / src0_type_ਮਾਪ,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_ਮਾਪ, (uint32_t)src1->nb[1] / src1_type_ਮਾਪ, (uint32_t)src1->nb[2] / src1_type_ਮਾਪ, (uint32_t)src1->nb[3] / src1_type_ਮਾਪ,
        (uint32_t) mul->ne[0], (uint32_t) mul->ne[1], (uint32_t) mul->ne[2],(uint32_t) mul->ne[3], (uint32_t) mul->nb[0] /  dst_type_ਮਾਪ, (uint32_t) mul->nb[1] /  dst_type_ਮਾਪ, (uint32_t) mul->nb[2] /  dst_type_ਮਾਪ, (uint32_t) mul->nb[3] /  dst_type_ਮਾਪ,
        0,
        0.0f, 0.0f, 0,
    }, pipeline);
}

ਖਾਲੀ ggml_vk_div(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ggml_tensor * dst) {
    ਅਚਲ uint32_t src0_type_ਮਾਪ = ggml_type_ਮਾਪ(src0->type);
    ਅਚਲ uint32_t src1_type_ਮਾਪ = ggml_type_ਮਾਪ(src1->type);
    ਅਚਲ uint32_t dst_type_ਮਾਪ = ggml_type_ਮਾਪ(dst->type);

    ggml_vk_op_f32<vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants>(ctx, subctx, src0, src1, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, dst, GGML_OP_DIV, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_ਮਾਪ, (uint32_t)src0->nb[1] / src0_type_ਮਾਪ, (uint32_t)src0->nb[2] / src0_type_ਮਾਪ, (uint32_t)src0->nb[3] / src0_type_ਮਾਪ,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_ਮਾਪ, (uint32_t)src1->nb[1] / src1_type_ਮਾਪ, (uint32_t)src1->nb[2] / src1_type_ਮਾਪ, (uint32_t)src1->nb[3] / src1_type_ਮਾਪ,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_ਮਾਪ, (uint32_t) dst->nb[1] /  dst_type_ਮਾਪ, (uint32_t) dst->nb[2] /  dst_type_ਮਾਪ, (uint32_t) dst->nb[3] /  dst_type_ਮਾਪ,
        0,
        0.0f, 0.0f, 0,
    });
}

ਖਾਲੀ ggml_vk_ਜੋੜੋ_id(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ਅਚਲ ggml_tensor * src2, ggml_tensor * dst) {
    ਅਚਲ uint32_t src0_type_ਮਾਪ = ggml_type_ਮਾਪ(src0->type);
    ਅਚਲ uint32_t src1_type_ਮਾਪ = ggml_type_ਮਾਪ(src1->type);
    ਅਚਲ uint32_t src2_type_ਮਾਪ = ggml_type_ਮਾਪ(src2->type);

    ggml_vk_op_f32<vk_op_ਜੋੜੋ_id_ਧੱਕੋ_constants>(ctx, subctx, src0, src1, src2, ਖਾਲੀ_ਸੂਚਕ, dst, GGML_OP_ਜੋੜੋ_ID, {
        (uint32_t)dst->ne[0],
        (uint32_t)dst->ne[1],
        (uint32_t)src0->nb[1] / src0_type_ਮਾਪ,
        (uint32_t)src0->nb[2] / src0_type_ਮਾਪ,
        (uint32_t)src1->nb[1] / src1_type_ਮਾਪ,
        (uint32_t)src2->nb[1] / src2_type_ਮਾਪ,
    });
}

ਸਥਿਰ ਖਾਲੀ ggml_vk_op_f32_wkv(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst, ਅਚਲ vk_op_rwkv_wkv6_ਧੱਕੋ_constants&& pc, ਪੂਰਨ ਸੰਸਕਰਣ_2) {
    GGML_ASSERT(ਸੰਸਕਰਣ_2 == 6 || ਸੰਸਕਰਣ_2 == 7);
    ਪੂਰਨ ਅੰਕ_srcs = ਸੰਸਕਰਣ_2 == 6 ? 6 : 7;

    ਲਈ (ਪੂਰਨ i = 0; i < ਅੰਕ_srcs; i++) {
        GGML_ASSERT(!ggml_is_quantized(dst->src[i]->type));
    }

    GGML_ASSERT(dst->ਬਫਰ != ਖਾਲੀ_ਸੂਚਕ);

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, dst->src[0], dst->src[1], dst->src[2], dst, dst->op);
    GGML_ASSERT(pipeline != ਖਾਲੀ_ਸੂਚਕ);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
    vk_subbuffer src_buf[7] = {};
    ਲਈ (ਪੂਰਨ i = 0; i < ਅੰਕ_srcs; i++) {
        src_buf[i] = ggml_vk_tensor_subbuffer(ctx, dst->src[i]);
    }

    std::array<uint32_t, 3> elements = {
        (uint32_t)(pc.B * pc.H),
        1,
        1
    };

    ਜੇ (ਸੰਸਕਰਣ_2 == 6) {
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
            {src_buf[0], src_buf[1], src_buf[2], src_buf[3], src_buf[4], src_buf[5], dst_buf},
            pc, elements);
    } ਨਹੀਂਤਾਂ ਜੇ (ਸੰਸਕਰਣ_2 == 7) {
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
            {src_buf[0], src_buf[1], src_buf[2], src_buf[3], src_buf[4], src_buf[5], src_buf[6], dst_buf},
            pc, elements);
    } ਨਹੀਂਤਾਂ {
        // shouldn't happen
        GGML_ASSERT(ਝੂਠ);
    }
}

ਖਾਲੀ ggml_vk_rwkv_wkv6(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    ਅਚਲ size_t seq_ਲੰਮਾਈ = dst->src[0]->ne[2];
    ਅਚਲ size_t n_embed = dst->ne[0];
    ਅਚਲ size_t n_heads = dst->src[0]->ne[1];
    ਅਚਲ size_t n_seqs = dst->src[5]->ne[1];

    ggml_vk_op_f32_wkv(
        ctx, subctx, dst,
        {
            (uint32_t)n_seqs,
            (uint32_t)seq_ਲੰਮਾਈ,
            (uint32_t)n_embed,
            (uint32_t)n_heads,
        },
        6
    );
}

ਖਾਲੀ ggml_vk_rwkv_wkv7(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    ਅਚਲ size_t seq_ਲੰਮਾਈ = dst->src[0]->ne[2];
    ਅਚਲ size_t n_embed = dst->ne[0];
    ਅਚਲ size_t n_heads = dst->src[0]->ne[1];
    ਅਚਲ size_t n_seqs = dst->src[6]->ne[1];

    ggml_vk_op_f32_wkv(
        ctx, subctx, dst,
        {
            (uint32_t)n_seqs,
            (uint32_t)seq_ਲੰਮਾਈ,
            (uint32_t)n_embed,
            (uint32_t)n_heads,
        },
        7
    );
}

ਖਾਲੀ ggml_vk_gated_ਰੇਖਿਕ_attn(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    ਅਚਲ size_t seq_ਲੰਮਾਈ = dst->src[0]->ne[2];
    ਅਚਲ size_t n_embed    = dst->ne[0];
    ਅਚਲ size_t n_heads    = dst->src[0]->ne[1];
    ਅਚਲ size_t n_seqs     = dst->src[4]->ne[1];

    ਦਸ਼ਮਲਵ scale;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&scale, dst->op_params, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    GGML_ASSERT(dst->ਬਫਰ != ਖਾਲੀ_ਸੂਚਕ);

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, dst->src[0], dst->src[1], dst->src[2], dst, dst->op);
    GGML_ASSERT(pipeline != ਖਾਲੀ_ਸੂਚਕ);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
    vk_subbuffer src_buf[5] = {};
    ਲਈ (ਪੂਰਨ i = 0; i < 5; i++) {
        src_buf[i] = ggml_vk_tensor_subbuffer(ctx, dst->src[i]);
    }

    ਅਚਲ vk_op_gated_ਰੇਖਿਕ_attn_ਧੱਕੋ_constants pc = {
        (uint32_t)n_seqs,
        (uint32_t)seq_ਲੰਮਾਈ,
        (uint32_t)n_embed,
        (uint32_t)n_heads,
        scale,
    };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        {src_buf[0], src_buf[1], src_buf[2], src_buf[3], src_buf[4], dst_buf},
        pc, { (uint32_t)(n_seqs * n_heads), 1, 1 });
}

ਖਾਲੀ ggml_vk_lightning_indexer(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    ਅਚਲ ggml_tensor * q = dst->src[0];
    ਅਚਲ ggml_tensor * k = dst->src[1];
    ਅਚਲ ggml_tensor * w = dst->src[2];
    ਅਚਲ ggml_tensor * m = dst->src[3];

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, q, k, w, dst, dst->op);
    GGML_ASSERT(pipeline != ਖਾਲੀ_ਸੂਚਕ);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    ਅਚਲ uint32_t n_kv      = k->ne[2];
    ਅਚਲ uint32_t n_heads   = q->ne[1];
    ਅਚਲ uint32_t n_ਟੋਕਨ  = q->ne[2];
    ਅਚਲ uint32_t n_streams = q->ne[3];
    ਅਚਲ uint32_t n_masks   = m->ne[3];

    // one workgroup per tile of 64 keys and 8 tokens, see lightning_indexer.comp
    ਅਚਲ uint32_t n_tiles_kv = CEIL_DIV(n_kv, 64);
    ਅਚਲ uint32_t n_tiles_t  = CEIL_DIV(n_ਟੋਕਨ, 8);

    // q, w and dst are f32 and m is f16, so their strides are passed in elements;
    // k may be quantized, so its strides stay in bytes
    ਅਚਲ uint32_t q_nb1 = q->nb[1] / ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ uint32_t q_nb2 = q->nb[2] / ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ uint32_t q_nb3 = q->nb[3] / ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ uint32_t k_nb2 = k->nb[2];
    ਅਚਲ uint32_t k_nb3 = k->nb[3];
    ਅਚਲ uint32_t w_nb1 = w->nb[1] / ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ uint32_t w_nb3 = w->nb[3] / ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ uint32_t m_nb1 = m->nb[1] / ਆਕਾਰ(ggml_fp16_t);
    ਅਚਲ uint32_t m_nb3 = m->nb[3] / ਆਕਾਰ(ggml_fp16_t);
    ਅਚਲ uint32_t d_nb1 = dst->nb[1] / ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ uint32_t d_nb3 = dst->nb[3] / ਆਕਾਰ(ਦਸ਼ਮਲਵ);

    ਅਚਲ vk_op_lightning_indexer_ਧੱਕੋ_constants pc = {
        n_kv, n_heads, n_ਟੋਕਨ, n_masks,
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

ਖਾਲੀ ggml_vk_gated_delta_net(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    ਅਚਲ ggml_tensor * src_q     = dst->src[0];
    ਅਚਲ ggml_tensor * src_v     = dst->src[2];
    ਅਚਲ ggml_tensor * src_beta  = dst->src[4];

    GGML_ASSERT(dst->ਬਫਰ != ਖਾਲੀ_ਸੂਚਕ);

    ਅਚਲ uint32_t S_v      = (uint32_t)src_v->ne[0];
    ਅਚਲ uint32_t H        = (uint32_t)src_v->ne[1];
    ਅਚਲ uint32_t n_ਟੋਕਨ = (uint32_t)src_v->ne[2];
    ਅਚਲ uint32_t n_seqs   = (uint32_t)src_v->ne[3];

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    ਅਚਲ uint32_t K = (uint32_t)ggml_get_op_params_i32(dst, 0);

    ਅਚਲ uint32_t s_off = S_v * H * n_ਟੋਕਨ * n_seqs;

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, dst->src[0], dst->src[1], dst->src[2], dst, dst->op);
    GGML_ASSERT(pipeline != ਖਾਲੀ_ਸੂਚਕ);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
    vk_subbuffer src_buf[6] = {};
    ਲਈ (ਪੂਰਨ i = 0; i < 6; i++) {
        src_buf[i] = ggml_vk_tensor_subbuffer(ctx, dst->src[i]);
    }

    ਅਚਲ uint32_t sq1 = (uint32_t)(src_q->nb[1] / ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਅਚਲ uint32_t sq2 = (uint32_t)(src_q->nb[2] / ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਅਚਲ uint32_t sq3 = (uint32_t)(src_q->nb[3] / ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਅਚਲ uint32_t sv1 = (uint32_t)(src_v->nb[1] / ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਅਚਲ uint32_t sv2 = (uint32_t)(src_v->nb[2] / ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਅਚਲ uint32_t sv3 = (uint32_t)(src_v->nb[3] / ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਅਚਲ uint32_t sb1 = (uint32_t)(src_beta->nb[1] / ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਅਚਲ uint32_t sb2 = (uint32_t)(src_beta->nb[2] / ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਅਚਲ uint32_t sb3 = (uint32_t)(src_beta->nb[3] / ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਅਚਲ uint32_t neq1 = (uint32_t)src_q->ne[1];
    ਅਚਲ uint32_t rq3  = (uint32_t)(src_v->ne[3] / src_q->ne[3]);

    ਅਚਲ ਦਸ਼ਮਲਵ scale = 1.0f / sqrtf((ਦਸ਼ਮਲਵ)S_v);
    ਅਚਲ vk_op_gated_delta_net_ਧੱਕੋ_constants pc = {
        H, n_ਟੋਕਨ, n_seqs, s_off,
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

ਖਾਲੀ ggml_vk_ssm_ਜਾਂਚ(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];
    ਅਚਲ ggml_tensor * src2 = dst->src[2];
    ਅਚਲ ggml_tensor * src3 = dst->src[3];
    ਅਚਲ ggml_tensor * src4 = dst->src[4];
    ਅਚਲ ggml_tensor * src5 = dst->src[5];

    GGML_ASSERT(dst->ਬਫਰ != ਖਾਲੀ_ਸੂਚਕ);

    ਅਚਲ uint32_t ਸਿਰ_dim = src0->ne[1];
    ਅਚਲ uint32_t n_ਸਿਰ = src1->ne[1];
    ਅਚਲ uint32_t n_ਸਮੂਹ = src4->ne[1];
    ਅਚਲ uint32_t n_tok = src1->ne[2];
    ਅਚਲ uint32_t n_seq = src1->ne[3];

    ਤਾਰਕਿਕ is_mamba2 = (src3->nb[1] == ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    GGML_ASSERT(is_mamba2);

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, src0, src1, src2, dst, dst->op);
    GGML_ASSERT(pipeline != ਖਾਲੀ_ਸੂਚਕ);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    ਅਚਲ int64_t s_off = ggml_nelements(src1) * ਆਕਾਰ(ਦਸ਼ਮਲਵ);

    ਅਚਲ vk_op_ssm_ਜਾਂਚ_ਧੱਕੋ_constants pc = {
        (uint32_t)src0->nb[2], (uint32_t)src0->nb[3],
        (uint32_t)src1->nb[2], (uint32_t)src1->nb[3],
        (uint32_t)src2->nb[1], (uint32_t)src2->nb[2],
        (uint32_t)src3->nb[1],
        (uint32_t)src4->nb[2], (uint32_t)src4->nb[3],
        (uint32_t)src5->nb[2], (uint32_t)src5->nb[3],
        (uint32_t)s_off,
        n_ਸਿਰ, ਸਿਰ_dim, n_ਸਮੂਹ, n_tok,
        n_seq, (uint32_t) ggml_get_op_params_i32(dst, 0)
    };

    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
    vk_subbuffer src_buf[7] = {};
    ਲਈ (ਪੂਰਨ i = 0; i < 7 && dst->src[i] != ਖਾਲੀ_ਸੂਚਕ; i++) {
        src_buf[i] = ggml_vk_tensor_subbuffer(ctx, dst->src[i]);
    }

    std::array<uint32_t, 3> elements;

    ਅਚਲ uint32_t d_state = src0->ne[0];
    uint32_t ਅੰਕ_subgroups = d_state / ctx->device->subgroup_ਮਾਪ;
    ਅਚਲ uint32_t ਅੰਕ_workgroups_x = CEIL_DIV(n_ਸਿਰ * ਸਿਰ_dim, ਅੰਕ_subgroups);
    ਅਚਲ uint32_t ਅੰਕ_workgroups_y = n_seq;
    elements = { ਅੰਕ_workgroups_x, ਅੰਕ_workgroups_y, 1 };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        {src_buf[0], src_buf[1], src_buf[2], src_buf[3], src_buf[4], src_buf[5], src_buf[6], dst_buf},
        pc, elements);
}

ਖਾਲੀ ggml_vk_ssm_conv(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ਬਣਤਰ ggml_cgraph * cgraph, ਪੂਰਨ ਗੰਢ_idx) {
    ggml_tensor * conv = cgraph->nodes[ਗੰਢ_idx];
    ਅਚਲ ggml_tensor * src0 = conv->src[0];
    ਅਚਲ ggml_tensor * src1 = conv->src[1];

    // Pick the destination tensor (last node in the fused chain) and the optional bias.
    // Fusion modes: 0 = ssm_conv, 1 = ssm_conv+silu, 2 = ssm_conv+add(bias)+silu.
    ggml_tensor * dst = conv;
    ਅਚਲ ggml_tensor * bias = ਖਾਲੀ_ਸੂਚਕ;

    ਜੇ (ctx->ਅੰਕ_additional_fused_ops == 1) {
        dst = cgraph->nodes[ਗੰਢ_idx + 1]; // silu
    } ਨਹੀਂਤਾਂ ਜੇ (ctx->ਅੰਕ_additional_fused_ops == 2) {
        ggml_tensor * ਜੋੜੋ_2 = cgraph->nodes[ਗੰਢ_idx + 1];
        bias = (ਜੋੜੋ_2->src[0] == conv) ? ਜੋੜੋ_2->src[1] : ਜੋੜੋ_2->src[0];
        dst = cgraph->nodes[ਗੰਢ_idx + 2]; // silu
    }

    // The shader always declares 4 bindings; bind src0 as a dummy when bias isn't fused.
    ਅਚਲ ggml_tensor * src2 = bias ? bias : src0;

    ggml_vk_op_f32<vk_op_ssm_conv_ਧੱਕੋ_constants>(ctx, subctx, src0, src1, src2, ਖਾਲੀ_ਸੂਚਕ, dst, GGML_OP_SSM_CONV, {
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

ਸਥਿਰ ਖਾਲੀ ggml_vk_op_f32_opt_ਕਦਮ_adamw(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst, ਅਚਲ vk_op_ਧੱਕੋ_constants&& pc) {
    ਅਚਲ ggml_tensor * x = dst->src[0];
    ਅਚਲ ggml_tensor * g = dst->src[1];
    ਅਚਲ ggml_tensor * gm = dst->src[2];
    ਅਚਲ ggml_tensor * gv = dst->src[3];
    ਅਚਲ ggml_tensor * p = dst->src[4];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(g->type == GGML_TYPE_F32);
    GGML_ASSERT(gm->type == GGML_TYPE_F32);
    GGML_ASSERT(gv->type == GGML_TYPE_F32);
    GGML_ASSERT(p->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->ਬਫਰ != ਖਾਲੀ_ਸੂਚਕ);
    GGML_ASSERT(ggml_is_contiguous(x));
    GGML_ASSERT(ggml_is_contiguous(g));
    GGML_ASSERT(ggml_is_contiguous(gm));
    GGML_ASSERT(ggml_is_contiguous(gv));
    GGML_ASSERT(ggml_is_contiguous(p));
    GGML_ASSERT(ggml_are_same_shape(x, g));
    GGML_ASSERT(ggml_are_same_shape(x, gm));
    GGML_ASSERT(ggml_are_same_shape(x, gv));
    GGML_ASSERT(ggml_nelements(p) == 7);

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, g, gm, gv, dst, GGML_OP_OPT_ਕਦਮ_ADAMW);
    GGML_ASSERT(pipeline != ਖਾਲੀ_ਸੂਚਕ);

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

ਖਾਲੀ ggml_vk_opt_ਕਦਮ_adamw(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    ਅਚਲ size_t n = ggml_nelements(dst->src[0]);

    ggml_vk_op_f32_opt_ਕਦਮ_adamw(
        ctx, subctx, dst,
        { (uint32_t)n, 0, 0.0f, 0.0f, 0.0f, 0.0f }
    );
}

ਖਾਲੀ ggml_vk_opt_ਕਦਮ_sgd(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ਅਚਲ ggml_tensor * src2, ggml_tensor * dst) {
    ਅਚਲ size_t n = ggml_nelements(dst->src[0]);

    ggml_vk_op_f32<vk_op_ਧੱਕੋ_constants>(ctx, subctx, src0, src1, src2, ਖਾਲੀ_ਸੂਚਕ, dst, GGML_OP_OPT_ਕਦਮ_SGD, { (uint32_t)n, 0, 0.0f, 0.0f, 0.0f, 0.0f });
}

ਖਾਲੀ ggml_vk_concat(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ggml_tensor * dst) {
    ਪੂਰਨ * op_params = (ਪੂਰਨ *)dst->op_params;

    ਅਚਲ uint32_t unit_ਮਾਪ = ggml_vk_concat_unit_ਮਾਪ(dst->type);
    ਅਚਲ uint32_t units_ਪ੍ਰਤੀ_ਖੰਡ = ggml_type_ਮਾਪ(dst->type) / unit_ਮਾਪ;
    ਅਚਲ uint32_t ਖੰਡ_ਮਾਪ_3 = ggml_blck_ਮਾਪ(dst->type);
    ਅਚਲ ਤਾਰਕਿਕ quantized = ggml_is_quantized(dst->type);

    // Address dimension 0 in packed storage units; higher strides may be noncontiguous.
    ਅਚਲ uint32_t ne00 = src0->ne[0] / ਖੰਡ_ਮਾਪ_3 * units_ਪ੍ਰਤੀ_ਖੰਡ;
    ਅਚਲ uint32_t ne10 = src1->ne[0] / ਖੰਡ_ਮਾਪ_3 * units_ਪ੍ਰਤੀ_ਖੰਡ;
    ਅਚਲ uint32_t ne20 =  dst->ne[0] / ਖੰਡ_ਮਾਪ_3 * units_ਪ੍ਰਤੀ_ਖੰਡ;
    ਅਚਲ uint32_t nb00 = quantized ? 1 : src0->nb[0] / unit_ਮਾਪ;
    ਅਚਲ uint32_t nb10 = quantized ? 1 : src1->nb[0] / unit_ਮਾਪ;
    ਅਚਲ uint32_t nb20 = quantized ? 1 :  dst->nb[0] / unit_ਮਾਪ;

    vk_op_concat_ਧੱਕੋ_constants pc {{
        ne20 * (uint32_t)dst->ne[1] * (uint32_t)dst->ne[2] * (uint32_t)dst->ne[3],
        ne00, (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], nb00, (uint32_t)src0->nb[1] / unit_ਮਾਪ, (uint32_t)src0->nb[2] / unit_ਮਾਪ, (uint32_t)src0->nb[3] / unit_ਮਾਪ,
        ne10, (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], nb10, (uint32_t)src1->nb[1] / unit_ਮਾਪ, (uint32_t)src1->nb[2] / unit_ਮਾਪ, (uint32_t)src1->nb[3] / unit_ਮਾਪ,
        ne20, (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], nb20, (uint32_t) dst->nb[1] / unit_ਮਾਪ, (uint32_t) dst->nb[2] / unit_ਮਾਪ, (uint32_t) dst->nb[3] / unit_ਮਾਪ,
        0,
        0.0f, 0.0f, op_params[0],
    }};
    ggml_vk_op_f32<vk_op_concat_ਧੱਕੋ_constants>(ctx, subctx, src0, src1, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, dst, GGML_OP_CONCAT, std::move(pc));
}

ਖਾਲੀ ggml_vk_upscale(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ggml_tensor * dst) {
    ਅਚਲ uint32_t src0_type_ਮਾਪ = ggml_type_ਮਾਪ(src0->type);
    ਅਚਲ uint32_t mode = (uint32_t)ggml_get_op_params_i32(dst, 0);

    GGML_TENSOR_UNARY_OP_LOCALS

    ਦਸ਼ਮਲਵ sf0 = (ਦਸ਼ਮਲਵ)ne0 / ne00;
    ਦਸ਼ਮਲਵ sf1 = (ਦਸ਼ਮਲਵ)ne1 / ne01;
    ਦਸ਼ਮਲਵ sf2 = (ਦਸ਼ਮਲਵ)ne2 / ne02;
    ਦਸ਼ਮਲਵ sf3 = (ਦਸ਼ਮਲਵ)ne3 / ne03;
    ਦਸ਼ਮਲਵ pixel_ਵਿੱਥ = 0.5f;

    ਜੇ (mode & GGML_SCALE_FLAG_ALIGN_CORNERS) {
        sf0 = ne0 > 1 && ne00 > 1 ? (ਦਸ਼ਮਲਵ)(ne0 - 1) / (ne00 - 1) : sf0;
        sf1 = ne1 > 1 && ne01 > 1 ? (ਦਸ਼ਮਲਵ)(ne1 - 1) / (ne01 - 1) : sf1;
        pixel_ਵਿੱਥ = 0.0f;
    }

    ggml_vk_op_f32<vk_op_upscale_ਧੱਕੋ_constants>(ctx, subctx, src0, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, dst, GGML_OP_UPSCALE, {
        (uint32_t)ggml_nelements(dst), 0, 0,
        (uint32_t)ne00, (uint32_t)ne01,
        (uint32_t)nb00 / src0_type_ਮਾਪ, (uint32_t)nb01 / src0_type_ਮਾਪ, (uint32_t)nb02 / src0_type_ਮਾਪ, (uint32_t)nb03 / src0_type_ਮਾਪ,
        (uint32_t)ne0, (uint32_t)ne1, (uint32_t)ne2, (uint32_t)ne3,
        sf0, sf1, sf2, sf3, pixel_ਵਿੱਥ
    });
}

ਖਾਲੀ ggml_vk_scale(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_unary_ਧੱਕੋ_constants p = vk_op_unary_ਧੱਕੋ_constants_init(src0, dst);
    p.param1 = ggml_get_op_params_f32(dst, 0);
    p.param2 = ggml_get_op_params_f32(dst, 1);

    ggml_vk_op_f32(ctx, subctx, src0, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, dst, GGML_OP_SCALE, std::move(p));
}

ਖਾਲੀ ggml_vk_sqr(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ggml_tensor * dst) {
    ggml_vk_op_f32(ctx, subctx, src0, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, dst, GGML_OP_SQR, vk_op_unary_ਧੱਕੋ_constants_init(src0, dst));
}

ਖਾਲੀ ggml_vk_sqrt(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ggml_tensor * dst) {
    ggml_vk_op_f32(ctx, subctx, src0, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, dst, GGML_OP_SQRT, vk_op_unary_ਧੱਕੋ_constants_init(src0, dst));
}

ਖਾਲੀ ggml_vk_add1(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ggml_tensor * dst) {
    ਅਚਲ uint32_t src0_type_ਮਾਪ = ggml_type_ਮਾਪ(src0->type);
    ਅਚਲ uint32_t src1_type_ਮਾਪ = ggml_type_ਮਾਪ(src1->type);
    ਅਚਲ uint32_t dst_type_ਮਾਪ = ggml_type_ਮਾਪ(dst->type);

    ggml_vk_op_f32<vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants>(ctx, subctx, src0, src1, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, dst, GGML_OP_ADD1, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_ਮਾਪ, (uint32_t)src0->nb[1] / src0_type_ਮਾਪ, (uint32_t)src0->nb[2] / src0_type_ਮਾਪ, (uint32_t)src0->nb[3] / src0_type_ਮਾਪ,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_ਮਾਪ, (uint32_t)src1->nb[1] / src1_type_ਮਾਪ, (uint32_t)src1->nb[2] / src1_type_ਮਾਪ, (uint32_t)src1->nb[3] / src1_type_ਮਾਪ,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_ਮਾਪ, (uint32_t) dst->nb[1] /  dst_type_ਮਾਪ, (uint32_t) dst->nb[2] /  dst_type_ਮਾਪ, (uint32_t) dst->nb[3] /  dst_type_ਮਾਪ,
        0,
        0.0f, 0.0f, 0,
    });
}

ਖਾਲੀ ggml_vk_arange(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_vk_arange(dst=" << dst << ", ne=" << ggml_nelements(dst) << ")");

    vk_op_ਧੱਕੋ_constants pc = {
        (uint32_t)ggml_nelements(dst),
        1,
        ggml_get_op_params_f32(dst, 0),
        ggml_get_op_params_f32(dst, 2),
        0.0f, 0.0f,
    };

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, dst, GGML_OP_ARANGE);
    GGML_ASSERT(pipeline != ਖਾਲੀ_ਸੂਚਕ);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst, ਝੂਠ);

    std::array<uint32_t, 3> elements = { (uint32_t)ggml_nelements(dst), 1, 1 };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { dst_buf }, pc, elements);
}

ਖਾਲੀ ggml_vk_fill(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_vk_fill(dst=" << dst << ", ne=" << ggml_nelements(dst) << ")");
    ਅਚਲ uint64_t n = ggml_nelements(dst);
    GGML_ASSERT(n > 0);

    vk_op_ਧੱਕੋ_constants pc = {
        (uint32_t)n,
        1,
        ggml_get_op_params_f32(dst, 0),
        0.0f,
        0.0f, 0.0f,
    };

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, dst, GGML_OP_FILL);
    GGML_ASSERT(pipeline != ਖਾਲੀ_ਸੂਚਕ);

    // Split the task distribution to 2D to avoid exceeding maxComputeWorkGroupCount
    ਅਚਲ uint32_t ਕੁੱਲ_wg = CEIL_DIV(n, pipeline->wg_denoms[0]);
    ਅਚਲ uint32_t wg_x = std::min(ਕੁੱਲ_wg, ctx->device->properties.limits.max_Compute_Work_ਸਮੂਹ_ਗਿਣਤੀ[0]);
    ਅਚਲ uint32_t wg_y = CEIL_DIV(ਕੁੱਲ_wg, wg_x);
    GGML_ASSERT(wg_y <= ctx->device->properties.limits.max_Compute_Work_ਸਮੂਹ_ਗਿਣਤੀ[1]);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst, ਝੂਠ);

    std::array<uint32_t, 3> elements = { wg_x * pipeline->wg_denoms[0], wg_y * pipeline->wg_denoms[1], 1 };
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { dst_buf }, pc, elements);
}

ਖਾਲੀ ggml_vk_sin(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ggml_tensor * dst) {
    ggml_vk_op_f32(ctx, subctx, src0, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, dst, GGML_OP_SIN, vk_op_unary_ਧੱਕੋ_constants_init(src0, dst));
}

ਖਾਲੀ ggml_vk_cos(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ggml_tensor * dst) {
    ggml_vk_op_f32(ctx, subctx, src0, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, dst, GGML_OP_COS, vk_op_unary_ਧੱਕੋ_constants_init(src0, dst));
}

ਖਾਲੀ ggml_vk_log(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ggml_tensor * dst) {
    ggml_vk_op_f32(ctx, subctx, src0, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, dst, GGML_OP_LOG, vk_op_unary_ਧੱਕੋ_constants_init(src0, dst));
}

ਖਾਲੀ ggml_vk_tri(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_unary_ਧੱਕੋ_constants p = vk_op_unary_ਧੱਕੋ_constants_init(src0, dst);
    p.param1 = ggml_get_op_params_f32(dst, 0);

    ggml_vk_op_f32(ctx, subctx, src0, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, dst, GGML_OP_TRI, std::move(p));
}

ਖਾਲੀ ggml_vk_diag(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_unary_ਧੱਕੋ_constants p = vk_op_unary_ਧੱਕੋ_constants_init(src0, dst, ggml_nelements(dst));

    ggml_vk_op_f32(ctx, subctx, src0, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, dst, GGML_OP_DIAG, std::move(p));
}

ਖਾਲੀ ggml_vk_clamp(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_unary_ਧੱਕੋ_constants p = vk_op_unary_ਧੱਕੋ_constants_init(src0, dst);
    p.param1 = ggml_get_op_params_f32(dst, 0);
    p.param2 = ggml_get_op_params_f32(dst, 1);

    ggml_vk_op_f32(ctx, subctx, src0, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, dst, GGML_OP_CLAMP, std::move(p));
}

ਖਾਲੀ ggml_vk_pad(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_pad_ਧੱਕੋ_constants p = vk_op_pad_ਧੱਕੋ_constants_init(src0, dst);
    ggml_vk_op_f32(ctx, subctx, src0, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, dst, GGML_OP_PAD, std::move(p));
}

ਖਾਲੀ ggml_vk_pad_reflect_1d(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ggml_tensor * dst) {
    ਅਚਲ uint32_t p0 = (uint32_t)dst->op_params[0];
    ਅਚਲ uint32_t p1 = (uint32_t)dst->op_params[1];

    vk_op_unary_ਧੱਕੋ_constants p = vk_op_unary_ਧੱਕੋ_constants_init(src0, dst, ggml_nelements(dst));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&p.param1, &p0, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&p.param2, &p1, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ggml_vk_op_f32(ctx, subctx, src0, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, dst, GGML_OP_PAD_REFLECT_1D, std::move(p));
}

ਖਾਲੀ ggml_vk_roll(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ggml_tensor * dst) {
    ਅਚਲ int32_t s0 = ggml_get_op_params_i32(dst, 0);
    ਅਚਲ int32_t s1 = ggml_get_op_params_i32(dst, 1);
    ਅਚਲ int32_t s2 = ggml_get_op_params_i32(dst, 2);
    ਅਚਲ int32_t s3 = ggml_get_op_params_i32(dst, 3);
    ਅਚਲ uint32_t s01_packed = ((s0 + 0x8000) << 16) | (s1 + 0x8000);
    ਅਚਲ uint32_t s23_packed = ((s2 + 0x8000) << 16) | (s3 + 0x8000);

    vk_op_unary_ਧੱਕੋ_constants p = vk_op_unary_ਧੱਕੋ_constants_init(src0, dst);
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&p.param1, &s01_packed, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&p.param2, &s23_packed, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ggml_vk_op_f32(ctx, subctx, src0, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, dst, GGML_OP_ROLL, std::move(p));
}

ਖਾਲੀ ggml_vk_repeat(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_unary_ਧੱਕੋ_constants p = vk_op_unary_ਧੱਕੋ_constants_init(src0, dst, ggml_nelements(dst));
    ggml_vk_op_f32(ctx, subctx, src0, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, dst, GGML_OP_REPEAT, std::move(p));
}

ਖਾਲੀ ggml_vk_repeat_ਪਿੱਠ(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_unary_ਧੱਕੋ_constants p = vk_op_unary_ਧੱਕੋ_constants_init(src0, dst, ggml_nelements(dst));
    ggml_vk_op_f32(ctx, subctx, src0, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, dst, GGML_OP_REPEAT_ਪਿੱਠ, std::move(p));
}

ਖਾਲੀ ggml_vk_cpy(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ggml_tensor * dst) {
    uint32_t ne = (uint32_t)ggml_nelements(src0);
    ਜੇ (ggml_is_quantized(src0->type) && ggml_is_quantized(dst->type)) {
        // Convert from number of logical elements to 2- or 4-byte units.
        ne /= ggml_blck_ਮਾਪ(src0->type);
        ਜੇ ((ggml_type_ਮਾਪ(src0->type) % 4) == 0) {
            ne *= ggml_type_ਮਾਪ(src0->type) / 4;
        } ਨਹੀਂਤਾਂ {
            ne *= ggml_type_ਮਾਪ(src0->type) / 2;
        }
    }

    vk_op_unary_ਧੱਕੋ_constants p = vk_op_unary_ਧੱਕੋ_constants_init(src0, dst, ne);
    ggml_vk_op_f32(ctx, subctx, src0, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, dst, GGML_OP_CPY, std::move(p));
}

ਖਾਲੀ ggml_vk_ਸੈੱਟ_ਪੰਕਤੀਆਂ(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ggml_tensor * dst) {
    ਅਚਲ uint32_t src0_type_ਮਾਪ = ggml_type_ਮਾਪ(src0->type);
    ਅਚਲ uint32_t src1_type_ਮਾਪ = ggml_type_ਮਾਪ(src1->type);
    ਅਚਲ uint32_t dst_type_ਮਾਪ = ggml_type_ਮਾਪ(dst->type);

    // Skip empty skip_rows operations. For most ops the empty check at the start
    // of ggml_vk_build_graph is sufficient, but set_rows can have a nonempty dst
    // with empty srcs.
    ਜੇ (ggml_is_ਖਾਲੀ_ਥਾਵਾਂ(src0) || ggml_is_ਖਾਲੀ_ਥਾਵਾਂ(src1)) {
        ਵਾਪਸ;
    }

    ggml_vk_op_f32<vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants>(ctx, subctx, src0, src1, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, dst, GGML_OP_ਸੈੱਟ_ਪੰਕਤੀਆਂ, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_ਮਾਪ, (uint32_t)src0->nb[1] / src0_type_ਮਾਪ, (uint32_t)src0->nb[2] / src0_type_ਮਾਪ, (uint32_t)src0->nb[3] / src0_type_ਮਾਪ,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_ਮਾਪ, (uint32_t)src1->nb[1] / src1_type_ਮਾਪ, (uint32_t)src1->nb[2] / src1_type_ਮਾਪ, (uint32_t)src1->nb[3] / src1_type_ਮਾਪ,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_ਮਾਪ, (uint32_t) dst->nb[1] /  dst_type_ਮਾਪ, (uint32_t) dst->nb[2] /  dst_type_ਮਾਪ, (uint32_t) dst->nb[3] /  dst_type_ਮਾਪ,
        0,
        0.0f, 0.0f, 0,
    });
}

ਖਾਲੀ ggml_vk_silu_ਪਿੱਠ(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ggml_tensor * dst) {
    ggml_vk_op_f32<vk_op_ਧੱਕੋ_constants>(ctx, subctx, src0, src1, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, dst, GGML_OP_SILU_ਪਿੱਠ, { (uint32_t)ggml_nelements(src0), 0, 0.0f, 0.0f, 0.0f, 0.0f });
}

ਖਾਲੀ ggml_vk_norm(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ggml_tensor * dst) {
    ਦਸ਼ਮਲਵ * op_params = (ਦਸ਼ਮਲਵ *)dst->op_params;
    vk_op_unary_ਧੱਕੋ_constants p = vk_op_unary_ਧੱਕੋ_constants_init(src0, dst);
    p.param1 = op_params[0];

    ggml_vk_op_f32(ctx, subctx, src0, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, dst, GGML_OP_NORM, std::move(p));
}

ਖਾਲੀ ggml_vk_ਸਮੂਹ_norm(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ggml_tensor * dst) {
    ਅਚਲ ਪੂਰਨ * int_op_params = (ਅਚਲ ਪੂਰਨ *)dst->op_params;
    ਅਚਲ ਦਸ਼ਮਲਵ * float_op_params = (ਅਚਲ ਦਸ਼ਮਲਵ *)dst->op_params;

    ਅਚਲ uint32_t ਅੰਕ_groups = int_op_params[0];
    ਅਚਲ ਦਸ਼ਮਲਵ eps = float_op_params[1];
    ਅਚਲ uint32_t ਸਮੂਹ_ਮਾਪ = src0->ne[0] * src0->ne[1] * ((src0->ne[2] + ਅੰਕ_groups - 1) / ਅੰਕ_groups);

    ggml_vk_op_f32<vk_op_ਧੱਕੋ_constants>(ctx, subctx, src0, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, dst, GGML_OP_ਸਮੂਹ_NORM, { ਸਮੂਹ_ਮਾਪ, 0, eps, 0.0f, 0.0f, 0.0f });
}

ਸਥਿਰ uint32_t ggml_vk_rms_ਅੰਕ_partials(ggml_backend_vk_context * ctx, ਅਚਲ ggml_tensor *ਗੰਢ) {
    ਅਚਲ uint32_t ne = (uint32_t)ਗੰਢ->ne[0];
    ਅਚਲ uint32_t denom = ctx->device->pipeline_ਜੋੜੋ_rms[0][0][0]->wg_denoms[0];
    ਅਚਲ uint32_t ਅੰਕ_partials = CEIL_DIV(ne, denom);
    ਵਾਪਸ ਅੰਕ_partials;
}

uint32_t ggml_vk_rms_partials_ਮਾਪ(ggml_backend_vk_context * ctx, ਅਚਲ ggml_tensor *ਗੰਢ) {
    ਅਚਲ uint32_t ਅੰਕ_partials = ggml_vk_rms_ਅੰਕ_partials(ctx, ਗੰਢ);
    ਅਚਲ uint32_t ਅੰਕ_ਬਾਈਟਾਂ = ROUNDUP_POW2(ਅੰਕ_partials * ਆਕਾਰ(uint32_t), ctx->device->partials_binding_alignment);
    ਵਾਪਸ ਅੰਕ_ਬਾਈਟਾਂ;
}

ਸਥਿਰ vk_op_rope_ਧੱਕੋ_constants ggml_vk_make_rope_constants(ਅਚਲ ggml_tensor *dst, ਅਚਲ ggml_tensor *src0, ਅਚਲ ਤਾਰਕਿਕ has_ff, ਤਾਰਕਿਕ backprop, ਅਚਲ uint32_t ਸੈੱਟ_ਪੰਕਤੀਆਂ_stride) {
    ਅਚਲ ਪੂਰਨ n_dims        = ((ਅਚਲ int32_t *) dst->op_params)[1];
    ਅਚਲ ਪੂਰਨ mode          = ((ਅਚਲ int32_t *) dst->op_params)[2];
    ਅਚਲ ਪੂਰਨ n_offs        = ((ਅਚਲ int32_t *) dst->op_params)[15];
    // const int n_ctx         = ((const int32_t *) dst->op_params)[3];
    ਅਚਲ ਪੂਰਨ n_ctx_orig    = ((ਅਚਲ int32_t *) dst->op_params)[4];
    ਅਚਲ ਦਸ਼ਮਲਵ freq_ਅਧਾਰ   = ((ਅਚਲ ਦਸ਼ਮਲਵ *)   dst->op_params)[5];
    ਅਚਲ ਦਸ਼ਮਲਵ freq_scale  = ((ਅਚਲ ਦਸ਼ਮਲਵ *)   dst->op_params)[6];
    ਅਚਲ ਦਸ਼ਮਲਵ ext_ਗੁਣਕ  = ((ਅਚਲ ਦਸ਼ਮਲਵ *)   dst->op_params)[7];
    ਅਚਲ ਦਸ਼ਮਲਵ attn_ਗੁਣਕ = ((ਅਚਲ ਦਸ਼ਮਲਵ *)   dst->op_params)[8];
    ਅਚਲ ਦਸ਼ਮਲਵ beta_fast   = ((ਅਚਲ ਦਸ਼ਮਲਵ *)   dst->op_params)[9];
    ਅਚਲ ਦਸ਼ਮਲਵ beta_slow   = ((ਅਚਲ ਦਸ਼ਮਲਵ *)   dst->op_params)[10];
    ਪੂਰਨ sections[4] {};
    ਜੇ (mode & GGML_ROPE_TYPE_MROPE) {
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(sections, (ਅਚਲ int32_t *) dst->op_params + 11, ਆਕਾਰ(ਪੂਰਨ)*4);
    }

    ਅਚਲ ਤਾਰਕਿਕ is_imrope = mode == GGML_ROPE_TYPE_IMROPE;

    ਦਸ਼ਮਲਵ corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_ਅਧਾਰ, beta_fast, beta_slow, corr_dims);

    ਅਚਲ ਦਸ਼ਮਲਵ theta_scale = powf(freq_ਅਧਾਰ, -2.0f/n_dims);

    uint32_t nb01 = src0->nb[1] / ggml_type_ਮਾਪ(src0->type);
    uint32_t nb02 = src0->nb[2] / ggml_type_ਮਾਪ(src0->type);
    uint32_t nb03 = src0->nb[3] / ggml_type_ਮਾਪ(src0->type);

    uint32_t nb11 = dst->nb[1] / ggml_type_ਮਾਪ(dst->type);
    uint32_t nb12 = dst->nb[2] / ggml_type_ਮਾਪ(dst->type);
    uint32_t nb13 = dst->nb[3] / ggml_type_ਮਾਪ(dst->type);

    vk_op_rope_ਧੱਕੋ_constants rope {
        (uint32_t)mode, (uint32_t)ggml_nrows(src0), (uint32_t)n_dims, (uint32_t)n_offs, freq_scale,
        freq_ਅਧਾਰ, ext_ਗੁਣਕ, attn_ਗੁਣਕ, {corr_dims[0], corr_dims[1]}, theta_scale, has_ff,
        { sections[0], sections[1], sections[2], sections[3] }, is_imrope, backprop, ਸੈੱਟ_ਪੰਕਤੀਆਂ_stride,

        (uint32_t)src0->ne[0],
        (uint32_t)src0->ne[1],
        (uint32_t)src0->ne[2],
        nb01, nb02, nb03,
        nb11, nb12, nb13,
        0, 0, // a_offset, d_offset filled in by init_pushconst_tensor_offsets
    };

    ਵਾਪਸ rope;
}

ਸਥਿਰ ਖਾਲੀ ggml_vk_rms_norm_ਅੰਤ(ggml_backend_vk_context * ctx, ਅਚਲ ggml_tensor * src0) {
    ਜੇ (ctx->do_ਜੋੜੋ_rms_partials_ਵਿੱਥ_calculation) {
        ctx->prealloc_ਮਾਪ_ਜੋੜੋ_rms_partials_ਵਿੱਥ += ggml_vk_rms_partials_ਮਾਪ(ctx, src0);
        ctx->do_ਜੋੜੋ_rms_partials = ਝੂਠ;
        ctx->do_ਜੋੜੋ_rms_partials_ਵਿੱਥ_calculation = ਝੂਠ;
    }
}

ਖਾਲੀ ggml_vk_rms_norm(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ਬਣਤਰ ggml_cgraph * cgraph, ਪੂਰਨ ਗੰਢ_idx, ਦਸ਼ਮਲਵ * op_params) {
    ggml_tensor * rms = cgraph->nodes[ਗੰਢ_idx];
    ਅਚਲ ggml_tensor * src0 = rms->src[0];

    ਜੇ (ctx->fused_rms_norm_ਬਹੁਲਕ == RMS_NORM_VIEW_ਸੈੱਟ_ਪੰਕਤੀਆਂ) {
        GGML_ASSERT(ctx->ਅੰਕ_additional_fused_ops == 2);
        ggml_tensor * ਸੈੱਟ_ਪੰਕਤੀਆਂ = cgraph->nodes[ਗੰਢ_idx + 2];
        ਅਚਲ ggml_tensor * indices = ਸੈੱਟ_ਪੰਕਤੀਆਂ->src[1];
        vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants pc = ggml_vk_rms_norm_ਧੱਕੋ_constants(src0, src0, ਸੈੱਟ_ਪੰਕਤੀਆਂ, op_params[0], 0);
        init_pushconst_tensor_offsets(ctx, pc, src0, src0, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, ਸੈੱਟ_ਪੰਕਤੀਆਂ);

        vk_pipeline pipeline = ਸੈੱਟ_ਪੰਕਤੀਆਂ->type == GGML_TYPE_F16 ?
            ctx->device->pipeline_rms_norm_ਸੈੱਟ_ਪੰਕਤੀਆਂ_f32_f16 : ctx->device->pipeline_rms_norm_ਸੈੱਟ_ਪੰਕਤੀਆਂ_f32_f32;
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
            {
                ggml_vk_tensor_subbuffer(ctx, src0, ਸੱਚ),
                ggml_vk_tensor_subbuffer(ctx, src0, ਸੱਚ),
                ggml_vk_tensor_subbuffer(ctx, ਸੈੱਟ_ਪੰਕਤੀਆਂ, ਸੱਚ),
                ggml_vk_tensor_subbuffer(ctx, indices),
            }, pc, {
                (uint32_t)src0->ne[1],
                std::min((uint32_t)src0->ne[2], ctx->device->properties.limits.max_Compute_Work_ਸਮੂਹ_ਗਿਣਤੀ[1]),
                std::min((uint32_t)src0->ne[3], ctx->device->properties.limits.max_Compute_Work_ਸਮੂਹ_ਗਿਣਤੀ[2]),
            });
        ggml_vk_rms_norm_ਅੰਤ(ctx, src0);
        ਵਾਪਸ;
    }

    ਜੇ (ctx->fused_rms_norm_ਬਹੁਲਕ == RMS_NORM_MUL_ਜੋੜੋ || ctx->fused_rms_norm_ਬਹੁਲਕ == RMS_NORM_MUL_ਜੋੜੋ_MUL) {
        ggml_tensor * mul = cgraph->nodes[ਗੰਢ_idx + 1];
        ggml_tensor * ਜੋੜੋ_2 = cgraph->nodes[ਗੰਢ_idx + 2];
        ਅਚਲ ggml_tensor * ਭਾਰ = mul->src[0] == rms ? mul->src[1] : mul->src[0];
        ਅਚਲ ggml_tensor * residual = ਜੋੜੋ_2->src[0] == mul ? ਜੋੜੋ_2->src[1] : ਜੋੜੋ_2->src[0];
        ਅਚਲ ਤਾਰਕਿਕ do_post_multiply = ctx->fused_rms_norm_ਬਹੁਲਕ == RMS_NORM_MUL_ਜੋੜੋ_MUL;
        GGML_ASSERT(ctx->ਅੰਕ_additional_fused_ops == (do_post_multiply ? 3 : 2));
        ggml_tensor * dst = do_post_multiply ? cgraph->nodes[ਗੰਢ_idx + 3] : ਜੋੜੋ_2;
        ਅਚਲ ggml_tensor * post_scale = do_post_multiply ?
            (dst->src[0] == ਜੋੜੋ_2 ? dst->src[1] : dst->src[0]) : src0;

        ਅਚਲ uint32_t ਅੰਕ_partials = ctx->do_ਜੋੜੋ_rms_partials ? ggml_vk_rms_ਅੰਕ_partials(ctx, dst) : 0;
        vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants pc = ggml_vk_rms_norm_ਧੱਕੋ_constants(src0, ਭਾਰ, dst, op_params[0], ਅੰਕ_partials);
        init_pushconst_tensor_offsets(ctx, pc, src0, ਭਾਰ, residual, post_scale, dst);

        vk_pipeline pipeline;
        ਜੇ (ctx->do_ਜੋੜੋ_rms_partials) {
            pipeline = do_post_multiply ?
                ctx->device->pipeline_rms_norm_mul_ਜੋੜੋ_mul_partials_f32 : ctx->device->pipeline_rms_norm_mul_ਜੋੜੋ_partials_f32;
        } ਨਹੀਂਤਾਂ {
            pipeline = do_post_multiply ?
                ctx->device->pipeline_rms_norm_mul_ਜੋੜੋ_mul_f32 : ctx->device->pipeline_rms_norm_mul_ਜੋੜੋ_f32;
        }
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
        ਜੇ (ctx->do_ਜੋੜੋ_rms_partials) {
            ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
                {
                    ggml_vk_tensor_subbuffer(ctx, src0, ਸੱਚ),
                    ggml_vk_tensor_subbuffer(ctx, ਭਾਰ, ਸੱਚ),
                    ggml_vk_tensor_subbuffer(ctx, dst, ਸੱਚ),
                    ggml_vk_subbuffer(ctx, ctx->prealloc_ਜੋੜੋ_rms_partials, ctx->prealloc_ਮਾਪ_ਜੋੜੋ_rms_partials_ਵਿੱਥ),
                    ggml_vk_tensor_subbuffer(ctx, residual),
                    ggml_vk_tensor_subbuffer(ctx, post_scale),
                }, pc, { (uint32_t)CEIL_DIV(src0->ne[0], 128), 1, 1 });
        } ਨਹੀਂਤਾਂ {
            ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
                {
                    ggml_vk_tensor_subbuffer(ctx, src0, ਸੱਚ),
                    ggml_vk_tensor_subbuffer(ctx, ਭਾਰ, ਸੱਚ),
                    ggml_vk_tensor_subbuffer(ctx, dst, ਸੱਚ),
                    ggml_vk_tensor_subbuffer(ctx, residual),
                    ggml_vk_tensor_subbuffer(ctx, post_scale),
                }, pc, {
                    (uint32_t)src0->ne[1],
                    std::min((uint32_t)src0->ne[2], ctx->device->properties.limits.max_Compute_Work_ਸਮੂਹ_ਗਿਣਤੀ[1]),
                    std::min((uint32_t)src0->ne[3], ctx->device->properties.limits.max_Compute_Work_ਸਮੂਹ_ਗਿਣਤੀ[2]),
                });
        }
        ggml_vk_rms_norm_ਅੰਤ(ctx, src0);
        ਵਾਪਸ;
    }

    ggml_tensor * dst;
    ਅਚਲ ggml_tensor * src1;

    ਜੇ (ctx->fused_rms_norm_ਬਹੁਲਕ != RMS_NORM_ਗਿਣਤੀ) {
        ggml_tensor * mul = cgraph->nodes[ਗੰਢ_idx + 1];
        dst = mul;
        src1 = mul->src[0] == rms ? mul->src[1] : mul->src[0];
    } ਨਹੀਂਤਾਂ {
        dst = rms;
        src1 = src0;
    }

    ਅਚਲ uint32_t ਅੰਕ_partials = ctx->do_ਜੋੜੋ_rms_partials ? ggml_vk_rms_ਅੰਕ_partials(ctx, dst) : 0;
    vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants bin = ggml_vk_rms_norm_ਧੱਕੋ_constants(src0, src1, dst, op_params[0], ਅੰਕ_partials);

    ਜੇ (ctx->fused_rms_norm_ਬਹੁਲਕ == RMS_NORM_MUL_ROPE ||
        ctx->fused_rms_norm_ਬਹੁਲਕ == RMS_NORM_MUL_ROPE_VIEW_ਸੈੱਟ_ਪੰਕਤੀਆਂ) {
        ਸਥਿਰ constexpr uint32_t max_tensors = 7;
        ਅਚਲ ggml_tensor *tensors[max_tensors] {};

        ggml_tensor *rms = cgraph->nodes[ਗੰਢ_idx + 0];
        ggml_tensor *mul = cgraph->nodes[ਗੰਢ_idx + 1];
        ggml_tensor *rope = cgraph->nodes[ਗੰਢ_idx + 2];

        ggml_tensor *other_src = mul->src[0] == rms ? mul->src[1] : mul->src[0];

        ਤਾਰਕਿਕ do_ਸੈੱਟ_ਪੰਕਤੀਆਂ = ctx->fused_rms_norm_ਬਹੁਲਕ == RMS_NORM_MUL_ROPE_VIEW_ਸੈੱਟ_ਪੰਕਤੀਆਂ;
        GGML_ASSERT(ctx->ਅੰਕ_additional_fused_ops == (do_ਸੈੱਟ_ਪੰਕਤੀਆਂ ? 4 : 2));

        tensors[0] = rms->src[0];
        tensors[1] = other_src;
        tensors[2] = mul;
        tensors[3] = rope->src[1]; // pos
        tensors[4] = rope->src[2]; // ff
        tensors[5] = cgraph->nodes[ਗੰਢ_idx + ctx->ਅੰਕ_additional_fused_ops]; // dst
        tensors[6] = do_ਸੈੱਟ_ਪੰਕਤੀਆਂ ? tensors[5]->src[1] : ਖਾਲੀ_ਸੂਚਕ;
        ਅਚਲ uint32_t ਸੈੱਟ_ਪੰਕਤੀਆਂ_stride = do_ਸੈੱਟ_ਪੰਕਤੀਆਂ ? tensors[5]->nb[1] / ggml_type_ਮਾਪ(tensors[5]->type) : 0;

        vk_op_rms_norm_mul_rope_ਧੱਕੋ_constants pc;
        pc.bin = bin;
        pc.rope = ggml_vk_make_rope_constants(rope, rope->src[0], tensors[4] != ਖਾਲੀ_ਸੂਚਕ, ਝੂਠ, ਸੈੱਟ_ਪੰਕਤੀਆਂ_stride);

        vk_pipeline pipeline = tensors[5]->type == GGML_TYPE_F16 ? ctx->device->pipeline_rms_norm_mul_rope_f32_f16 : ctx->device->pipeline_rms_norm_mul_rope_f32_f32;

        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

        ggml_backend_vk_ਬਫਰ_context * buf_ctx[max_tensors];
        vk_ਬਫਰ buf[max_tensors];
        size_t ਵਿੱਥ[max_tensors];
        ਤਾਰਕਿਕ uma[max_tensors];

        ਲਈ (uint32_t i = 0; i < max_tensors; ++i) {
            ਜੇ (!tensors[i]) {
                // If any remaining descriptors are unused, just point them at src[0]
                buf[i] = buf[0];
                ਵਿੱਥ[i] = 0;
                ਜਾਰੀ;
            }
            buf_ctx[i] = (ggml_backend_vk_ਬਫਰ_context *)tensors[i]->ਬਫਰ->context;
            buf[i] = ਖਾਲੀ_ਸੂਚਕ;
            ਵਿੱਥ[i] = 0;
            uma[i] = ਝੂਠ;

            ਜੇ (ctx->device->uma) {
                ggml_vk_host_get(ctx->device, tensors[i]->ਅੰਕੜੇ, buf[i], ਵਿੱਥ[i]);
                uma[i] = buf[i] != ਖਾਲੀ_ਸੂਚਕ;
            }
            ਜੇ (!uma[i]) {
                buf[i] = buf_ctx[i]->dev_ਬਫਰ;
                ਵਿੱਥ[i] = vk_tensor_ਵਿੱਥ(tensors[i]) + tensors[i]->view_offs;
            }
            GGML_ASSERT(buf[i] != ਖਾਲੀ_ਸੂਚਕ);
        }

        // a_offset is unused (the fused path reads from shared memory), but the rope/set_rows dst can be misaligned.
        // Round the binding offset down to the storage buffer alignment; the in-element shift goes in pc.rope.d_offset.
        pc.rope.d_ਵਿੱਥ = get_misalign_ਬਾਈਟਾਂ(ctx, tensors[5]) / ggml_type_ਮਾਪ(tensors[5]->type);
        ਵਿੱਥ[5] &= ~(size_t(ctx->device->properties.limits.min_Storage_ਬਫਰ_ਵਿੱਥ_Alignment) - 1);

        std::array<uint32_t, 3> elements;
        elements = { (uint32_t)rms->src[0]->ne[1], (uint32_t)rms->src[0]->ne[2], (uint32_t)rms->src[0]->ne[3] };
        elements[1] = std::min(elements[1], ctx->device->properties.limits.max_Compute_Work_ਸਮੂਹ_ਗਿਣਤੀ[1]);
        elements[2] = std::min(elements[2], ctx->device->properties.limits.max_Compute_Work_ਸਮੂਹ_ਗਿਣਤੀ[2]);

        static_assert(max_tensors == 7);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
            {
                ggml_vk_subbuffer(ctx, buf[0], ਵਿੱਥ[0]),
                ggml_vk_subbuffer(ctx, buf[1], ਵਿੱਥ[1]),
                ggml_vk_subbuffer(ctx, buf[2], ਵਿੱਥ[2]),
                ggml_vk_subbuffer(ctx, buf[3], ਵਿੱਥ[3]),
                ggml_vk_subbuffer(ctx, buf[4], ਵਿੱਥ[4]),
                ggml_vk_subbuffer(ctx, buf[5], ਵਿੱਥ[5]),
                ggml_vk_subbuffer(ctx, buf[6], ਵਿੱਥ[6]),
            }, pc, elements);
    } ਨਹੀਂਤਾਂ {
        GGML_ASSERT(ctx->fused_rms_norm_ਬਹੁਲਕ == RMS_NORM_MUL || ctx->fused_rms_norm_ਬਹੁਲਕ == RMS_NORM_ਗਿਣਤੀ);
        ggml_vk_op_f32<vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants>(ctx, subctx, src0, src1, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, dst, GGML_OP_RMS_NORM, std::move(bin));
    }

    ggml_vk_rms_norm_ਅੰਤ(ctx, src0);
}

ਖਾਲੀ ggml_vk_rms_norm_ਪਿੱਠ(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ggml_tensor * dst) {
    ਦਸ਼ਮਲਵ * op_params = (ਦਸ਼ਮਲਵ *)dst->op_params;
    ggml_vk_op_f32<vk_op_ਧੱਕੋ_constants>(ctx, subctx, src0, src1, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, dst, GGML_OP_RMS_NORM_ਪਿੱਠ, { (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], op_params[0], 0.0f, 0.0f, 0.0f });
}

ਖਾਲੀ ggml_vk_l2_norm(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ggml_tensor * dst) {
    ਅਚਲ ਦਸ਼ਮਲਵ * op_params = (ਅਚਲ ਦਸ਼ਮਲਵ *)dst->op_params;
    vk_op_unary_ਧੱਕੋ_constants p = vk_op_unary_ਧੱਕੋ_constants_init(src0, dst);
    p.param1 = op_params[0];
    ggml_vk_op_f32<vk_op_unary_ਧੱਕੋ_constants>(ctx, subctx, src0, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, dst, GGML_OP_L2_NORM, std::move(p));
}

ਖਾਲੀ ggml_vk_unary(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ggml_tensor * dst) {
    ggml_vk_op_f32(ctx, subctx, src0, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, dst, GGML_OP_UNARY, vk_op_unary_ਧੱਕੋ_constants_init(src0, dst));
}

ਖਾਲੀ ggml_vk_xielu(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ggml_tensor * dst) {
    ਦਸ਼ਮਲਵ * op_params = (ਦਸ਼ਮਲਵ *)dst->op_params;
    vk_op_unary_ਧੱਕੋ_constants p = vk_op_unary_ਧੱਕੋ_constants_init(src0, dst);
    p.param1 = op_params[1];
    p.param2 = op_params[2];
    p.param3 = op_params[3];
    p.param4 = op_params[4];
    ggml_vk_op_f32(ctx, subctx, src0, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, dst, GGML_OP_UNARY, std::move(p));
}

ਖਾਲੀ ggml_vk_glu(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ggml_tensor * dst) {
    ਅਚਲ ਦਸ਼ਮਲਵ * op_params_f = (ਅਚਲ ਦਸ਼ਮਲਵ *)dst->op_params;

    ਅਚਲ ਤਾਰਕਿਕ swapped = (ਤਾਰਕਿਕ)dst->op_params[1];
    ਅਚਲ ਤਾਰਕਿਕ ਵੰਡੋ = src1 != ਖਾਲੀ_ਸੂਚਕ;
    ਅਚਲ ਦਸ਼ਮਲਵ alpha = op_params_f[2];
    ਅਚਲ ਦਸ਼ਮਲਵ ਹੱਦਬੰਦੀ = op_params_f[3];

    ਜੇ (!ਵੰਡੋ) {
        GGML_ASSERT(src0->ne[0] / 2 == dst->ne[0]);
    } ਨਹੀਂਤਾਂ {
        GGML_ASSERT(src0->ne[0] == src1->ne[0]);
        GGML_ASSERT(src0->ne[0] == dst->ne[0]);
        GGML_ASSERT(src0->type == src1->type);
    }

    ਅਚਲ uint32_t mode = ਵੰਡੋ ? 2 : (swapped ? 1 : 0);
    ਅਚਲ uint32_t src0_type_ਮਾਪ = ggml_type_ਮਾਪ(src0->type);
    ਅਚਲ uint32_t src1_type_ਮਾਪ = ਵੰਡੋ ? ggml_type_ਮਾਪ(src1->type) : src0_type_ਮਾਪ;
    ਅਚਲ uint32_t dst_type_ਮਾਪ  = ggml_type_ਮਾਪ(dst->type);

    ggml_vk_op_f32<vk_op_glu_ਧੱਕੋ_constants>(ctx, subctx, src0, src1, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, dst, GGML_OP_GLU,
        {
            (uint32_t)ggml_nelements(dst),
            (uint32_t)src0->ne[0],
            (uint32_t)dst->ne[0],
            mode,
            alpha,
            ਹੱਦਬੰਦੀ,
            (uint32_t)(src0->nb[0] / src0_type_ਮਾਪ),
            (uint32_t)(src0->nb[1] / src0_type_ਮਾਪ),
            (uint32_t)(src0->nb[2] / src0_type_ਮਾਪ),
            (uint32_t)(src0->nb[3] / src0_type_ਮਾਪ),
            (uint32_t)((ਵੰਡੋ ? src1->nb[0] : src0->nb[0]) / src1_type_ਮਾਪ),
            (uint32_t)((ਵੰਡੋ ? src1->nb[1] : src0->nb[1]) / src1_type_ਮਾਪ),
            (uint32_t)((ਵੰਡੋ ? src1->nb[2] : src0->nb[2]) / src1_type_ਮਾਪ),
            (uint32_t)((ਵੰਡੋ ? src1->nb[3] : src0->nb[3]) / src1_type_ਮਾਪ),
            (uint32_t)(dst->nb[0] / dst_type_ਮਾਪ),
            (uint32_t)(dst->nb[1] / dst_type_ਮਾਪ),
            (uint32_t)(dst->nb[2] / dst_type_ਮਾਪ),
            (uint32_t)(dst->nb[3] / dst_type_ਮਾਪ),
            (uint32_t)dst->ne[1],
            (uint32_t)dst->ne[2],
            0,
            0, 0, 0, 0, 0, 0,
        });
}

ਖਾਲੀ ggml_vk_diag_ਨਕਾਬ_inf(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ggml_tensor * dst) {
    int32_t * op_params = (int32_t *)dst->op_params;
    ggml_vk_op_f32<vk_op_diag_ਨਕਾਬ_ਧੱਕੋ_constants>(ctx, subctx, src0, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, dst, GGML_OP_DIAG_ਨਕਾਬ_INF, { (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], op_params[0] });
}

ਖਾਲੀ ggml_vk_soft_max(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ਅਚਲ ggml_tensor * src2, ggml_tensor * dst) {
    ਦਸ਼ਮਲਵ * op_params = (ਦਸ਼ਮਲਵ *)dst->op_params;

    ਦਸ਼ਮਲਵ scale = op_params[0];
    ਦਸ਼ਮਲਵ max_bias = op_params[1];

    ਅਚਲ uint32_t ncols =   (uint32_t)src0->ne[0];
    ਅਚਲ uint32_t nrows_x = (uint32_t)ggml_nrows(src0);
    ਅਚਲ uint32_t nrows_y = (uint32_t)src0->ne[1];

    ਅਚਲ uint32_t ne12 = src1 ? (uint32_t)(src1->ne[2]) : 0u;
    ਅਚਲ uint32_t ne13 = src1 ? (uint32_t)(src1->ne[3]) : 0u;
    ਅਚਲ uint32_t nb11 = src1 ? (uint32_t)(src1->nb[1] / src1->nb[0]) : 0u;
    ਅਚਲ uint32_t nb12 = src1 ? (uint32_t)(src1->nb[2] / src1->nb[0]) : 0u;
    ਅਚਲ uint32_t nb13 = src1 ? (uint32_t)(src1->nb[3] / src1->nb[0]) : 0u;

    ਅਚਲ uint32_t n_ਸਿਰ_kv   = src0->ne[2];
    ਅਚਲ uint32_t n_ਸਿਰ_log2 = 1u << (uint32_t) floorf(log2f((ਦਸ਼ਮਲਵ) n_ਸਿਰ_kv));

    ਅਚਲ ਦਸ਼ਮਲਵ m0 = powf(2.0f, -(max_bias       ) / n_ਸਿਰ_log2);
    ਅਚਲ ਦਸ਼ਮਲਵ m1 = powf(2.0f, -(max_bias / 2.0f) / n_ਸਿਰ_log2);

    vk_op_soft_max_ਧੱਕੋ_constants pc {
        ncols,
        src1 != ਖਾਲੀ_ਸੂਚਕ ? nrows_y : (uint32_t)0,
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],
        ne12, ne13,
        nb11, nb12, nb13,
        scale, max_bias,
        m0, m1,
        n_ਸਿਰ_log2,
        nrows_x,
        src2 != ਖਾਲੀ_ਸੂਚਕ
    };

    ਜੇ (ncols <= 16384) {
        ggml_vk_op_f32<vk_op_soft_max_ਧੱਕੋ_constants>(ctx, subctx, src0, src1, src2, ਖਾਲੀ_ਸੂਚਕ, dst, GGML_OP_SOFT_MAX, std::move(pc));
    } ਨਹੀਂਤਾਂ {

        vk_subbuffer buf_a = ggml_vk_tensor_subbuffer(ctx, src0);
        vk_subbuffer buf_b = src1 ? ggml_vk_tensor_subbuffer(ctx, src1) : buf_a;
        vk_subbuffer buf_c = src2 ? ggml_vk_tensor_subbuffer(ctx, src2) : buf_a;
        vk_subbuffer buf_d = ggml_vk_tensor_subbuffer(ctx, dst);

        uint32_t elems_ਪ੍ਰਤੀ_wg = 128 * 4;
        uint32_t ਅੰਕ_wgs = CEIL_DIV(ncols, elems_ਪ੍ਰਤੀ_wg);
        size_t tmp_ਮਾਪ = ਅੰਕ_wgs * nrows_x * ਆਕਾਰ(ਦਸ਼ਮਲਵ);

        ਜੇ (ctx->prealloc_ਮਾਪ_x < tmp_ਮਾਪ) {
            ctx->prealloc_ਮਾਪ_x = tmp_ਮਾਪ;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        ਜੇ (ctx->prealloc_ਮਾਪ_y < tmp_ਮਾਪ) {
            ctx->prealloc_ਮਾਪ_y = tmp_ਮਾਪ;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        ਜੇ (ctx->prealloc_x_need_sync || ctx->prealloc_y_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }

        vk_subbuffer buf_x = { ctx->prealloc_x, 0, tmp_ਮਾਪ };
        vk_subbuffer buf_y = { ctx->prealloc_y, 0, tmp_ਮਾਪ };

        // the partial results overwrite a matmul input converted into prealloc_y
        ctx->prealloc_y_ਆਖਰੀ_pipeline_ਵਰਤਿਆ = ਖਾਲੀ_ਸੂਚਕ;
        ctx->prealloc_y_ਆਖਰੀ_tensor_ਵਰਤਿਆ = ਖਾਲੀ_ਸੂਚਕ;

        std::array<uint32_t, 3> elements = { ਅੰਕ_wgs, nrows_x, 1 };

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

        ctx->prealloc_x_need_sync = ਸੱਚ;
        ctx->prealloc_y_need_sync = ਸੱਚ;
    }
}

ਖਾਲੀ ggml_vk_soft_max_ਪਿੱਠ(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ggml_tensor * dst) {
    ਦਸ਼ਮਲਵ * op_params = (ਦਸ਼ਮਲਵ *)dst->op_params;
    ggml_vk_op_f32<vk_op_ਧੱਕੋ_constants>(ctx, subctx, src0, src1, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, dst, GGML_OP_SOFT_MAX_ਪਿੱਠ, { (uint32_t)src0->ne[0], (uint32_t)ggml_nrows(src0), op_params[0], op_params[1], 0.0f, 0.0f });
}

ਖਾਲੀ ggml_vk_topk_moe(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_cgraph * cgraph, ਪੂਰਨ ਗੰਢ_idx) {
    topk_moe_ਬਹੁਲਕ mode = ctx->fused_topk_moe_ਬਹੁਲਕ;
    ਅਚਲ ਤਾਰਕਿਕ has_bias = mode == TOPK_MOE_SIGMOID_NORM_BIAS || mode == TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS;
    ggml_tensor * logits = cgraph->nodes[ਗੰਢ_idx + 0]->src[0];
    ggml_tensor * bias = mode == TOPK_MOE_SIGMOID_NORM_BIAS       ? cgraph->nodes[ਗੰਢ_idx + 2]->src[1] :
                         mode == TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS ? cgraph->nodes[ਗੰਢ_idx + 3]->src[1] :
                                                                   logits;
    ggml_tensor * weights = cgraph->nodes[ਗੰਢ_idx + ctx->ਅੰਕ_additional_fused_ops];
    ggml_tensor * ids = mode == TOPK_MOE_SIGMOID_NORM_BIAS       ? cgraph->nodes[ਗੰਢ_idx + 4] :
                        mode == TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS ? cgraph->nodes[ਗੰਢ_idx + 5] :
                        mode == TOPK_MOE_LATE_SOFTMAX             ? cgraph->nodes[ਗੰਢ_idx + 1] :
                                                                   cgraph->nodes[ਗੰਢ_idx + 3];

    GGML_ASSERT(logits->type == GGML_TYPE_F32);
    GGML_ASSERT(bias->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    ਅਚਲ ਪੂਰਨ n_experts = logits->ne[0];
    ਅਚਲ ਪੂਰਨ n_ਪੰਕਤੀਆਂ    = logits->ne[1];
    ਅਚਲ ਪੂਰਨ n_expert_ਵਰਤਿਆ = weights->ne[1];

    GGML_ASSERT(ids->nb[1] / ggml_type_ਮਾਪ(ids->type) == (size_t) n_experts);

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, cgraph->nodes[ਗੰਢ_idx], GGML_OP_SOFT_MAX);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer logits_buf = ggml_vk_tensor_subbuffer(ctx, logits);
    vk_subbuffer bias_buf = ggml_vk_tensor_subbuffer(ctx, bias);
    vk_subbuffer weights_buf = ggml_vk_tensor_subbuffer(ctx, weights);
    vk_subbuffer ids_buf = ggml_vk_tensor_subbuffer(ctx, ids);

    vk_op_topk_moe_ਧੱਕੋ_constants pc {};
    pc.n_ਪੰਕਤੀਆਂ = n_ਪੰਕਤੀਆਂ;
    pc.n_experts_ਧੱਕੋ = n_experts;
    pc.n_expert_ਵਰਤਿਆ = n_expert_ਵਰਤਿਆ;
    pc.clamp_min = -std::numeric_limits<ਦਸ਼ਮਲਵ>::infinity();
    pc.clamp_max = std::numeric_limits<ਦਸ਼ਮਲਵ>::infinity();
    ਜੇ (mode == TOPK_MOE_EARLY_SOFTMAX_NORM) {
        ggml_tensor * clamp = cgraph->nodes[ਗੰਢ_idx + 7];
        GGML_ASSERT(clamp->op == GGML_OP_CLAMP);
        pc.clamp_min = ggml_get_op_params_f32(clamp, 0);
        pc.clamp_max = ggml_get_op_params_f32(clamp, 1);
    }
    ਜੇ (mode == TOPK_MOE_SIGMOID_NORM_BIAS) {
        ggml_tensor * clamp = cgraph->nodes[ਗੰਢ_idx + 8];
        GGML_ASSERT(clamp->op == GGML_OP_CLAMP);
        pc.clamp_min = ggml_get_op_params_f32(clamp, 0);
        pc.clamp_max = ggml_get_op_params_f32(clamp, 1);
    }
    ਜੇ (mode == TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS) {
        ggml_tensor * clamp = cgraph->nodes[ਗੰਢ_idx + 9];
        GGML_ASSERT(clamp->op == GGML_OP_CLAMP);
        pc.clamp_min = ggml_get_op_params_f32(clamp, 0);
        pc.clamp_max = ggml_get_op_params_f32(clamp, 1);
    }

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GATING_FUNC_SOFTMAX 0
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GATING_FUNC_SIGMOID 1
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GATING_FUNC_SOFTMAX_ਭਾਰ 2
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GATING_FUNC_SQRT_SOFTPLUS 3

    pc.gating_func = mode == TOPK_MOE_SIGMOID_NORM_BIAS       ? GATING_FUNC_SIGMOID :
                     mode == TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS ? GATING_FUNC_SQRT_SOFTPLUS :
                     mode == TOPK_MOE_LATE_SOFTMAX             ? GATING_FUNC_SOFTMAX_ਭਾਰ :
                                                                 GATING_FUNC_SOFTMAX;
    pc.has_bias = has_bias;
    pc.with_norm = mode == TOPK_MOE_EARLY_SOFTMAX_NORM || has_bias;
    ਜੇ (ctx->fused_topk_moe_scale) {
        GGML_ASSERT(weights->op == GGML_OP_SCALE);
        pc.output_scale = ggml_get_op_params_f32(weights, 0);
        pc.output_bias = ggml_get_op_params_f32(weights, 1);
    } ਨਹੀਂਤਾਂ {
        pc.output_scale = 1.0f;
        pc.output_bias = 0.0f;
    }

    GGML_ASSERT(n_expert_ਵਰਤਿਆ <= n_experts);

    ਅਚਲ uint32_t ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_ਖੰਡ = 4;
    std::array<uint32_t, 3> elements = { CEIL_DIV(n_ਪੰਕਤੀਆਂ, ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_ਖੰਡ), 1, 1 };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, {logits_buf, bias_buf, weights_buf, ids_buf}, pc, elements);
}

ਖਾਲੀ ggml_vk_rope(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_cgraph * cgraph, ਪੂਰਨ ਗੰਢ_idx, ਤਾਰਕਿਕ backprop) {
    ggml_tensor * dst = cgraph->nodes[ਗੰਢ_idx];
    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];
    ਅਚਲ ggml_tensor * src2 = dst->src[2];
    ਅਚਲ ggml_tensor * src3 = ਖਾਲੀ_ਸੂਚਕ;
    ਅਚਲ ਪੂਰਨ n_dims        = ((int32_t *) dst->op_params)[1];
    ਅਚਲ ਪੂਰਨ mode          = ((int32_t *) dst->op_params)[2];
    // const int n_ctx         = ((int32_t *) dst->op_params)[3];
    ਅਚਲ ਪੂਰਨ n_ctx_orig    = ((int32_t *) dst->op_params)[4];
    ਅਚਲ ਦਸ਼ਮਲਵ freq_ਅਧਾਰ   = ((ਦਸ਼ਮਲਵ *)   dst->op_params)[5];
    ਅਚਲ ਦਸ਼ਮਲਵ beta_fast   = ((ਦਸ਼ਮਲਵ *)   dst->op_params)[9];
    ਅਚਲ ਦਸ਼ਮਲਵ beta_slow   = ((ਦਸ਼ਮਲਵ *)   dst->op_params)[10];
    ਪੂਰਨ sections[4] {};
    ਜੇ (mode & GGML_ROPE_TYPE_MROPE) {
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(sections, (int32_t *) dst->op_params + 11, ਆਕਾਰ(ਪੂਰਨ)*4);
    }

    ਦਸ਼ਮਲਵ corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_ਅਧਾਰ, beta_fast, beta_slow, corr_dims);

    uint32_t ਸੈੱਟ_ਪੰਕਤੀਆਂ_stride = 0;
    // Fused rope + view + set_rows passes the set_rows destination stride in set_rows_stride
    // and overrides the dst and sets src3=row_indices
    ਜੇ (ctx->ਅੰਕ_additional_fused_ops > 0) {
        ਸੈੱਟ_ਪੰਕਤੀਆਂ_stride = cgraph->nodes[ਗੰਢ_idx + 2]->nb[1] / ggml_type_ਮਾਪ(cgraph->nodes[ਗੰਢ_idx + 2]->type);
        src3 = cgraph->nodes[ਗੰਢ_idx + 2]->src[1];
        dst = cgraph->nodes[ਗੰਢ_idx + 2];
    }

    ggml_vk_op_f32<vk_op_rope_ਧੱਕੋ_constants>(ctx, subctx, src0, src1, src2, src3, dst, GGML_OP_ROPE,
        ggml_vk_make_rope_constants(cgraph->nodes[ਗੰਢ_idx], src0, src2 != ਖਾਲੀ_ਸੂਚਕ, backprop, ਸੈੱਟ_ਪੰਕਤੀਆਂ_stride));
}

ਖਾਲੀ ggml_vk_argsort(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ggml_tensor * dst) {
    ਅਚਲ uint32_t * op_params = (ਅਚਲ uint32_t *)dst->op_params;

    uint32_t ncols = src0->ne[0];
    uint32_t nrows = ggml_nrows(src0);

    uint32_t ncols_pad_log2 = (uint32_t)ceilf(log2f(ਦਸ਼ਮਲਵ(ncols)));
    uint32_t ncolsp2 = 1 << ncols_pad_log2;

    vk_op_argsort_ਧੱਕੋ_constants pc { ncols, ncolsp2, ncols_pad_log2, nrows, op_params[0], 0, 0, 0, 0, };

    // Pick the largest workgroup size <= ncolsp2
    uint32_t pipeline_idx = std::min(ncols_pad_log2, ਅੰਕ_argsort_pipelines - 1);

    uint32_t max_wg_log2 = std::min(ctx->device->max_workgroup_ਮਾਪ_log2, ਅੰਕ_argsort_pipelines - 1);

    // Use the "small" argsort shader if the whole sort can be done by a single workgroup.
    ਤਾਰਕਿਕ use_small = ncols_pad_log2 <= max_wg_log2 &&
                     ctx->device->pipeline_argsort_f32[pipeline_idx] != ਖਾਲੀ_ਸੂਚਕ;

    vk_pipeline pipeline = use_small ? ctx->device->pipeline_argsort_f32[pipeline_idx]
                                     : ctx->device->pipeline_argsort_large_f32[pipeline_idx];

    vk_subbuffer src0_buf = ggml_vk_tensor_subbuffer(ctx, src0);
    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
    vk_subbuffer subbuf1 = dst_buf;

    // Reserve space for ivec2 per element, with rows padded to a power of two
    ਜੇ (!use_small) {
        ਅਚਲ size_t x_sz = size_t{ncolsp2} * nrows * 2 * ਆਕਾਰ(ਪੂਰਨ);

        ਜੇ (ctx->prealloc_ਮਾਪ_x < x_sz) {
            ctx->prealloc_ਮਾਪ_x = x_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        ਜੇ (ctx->prealloc_x_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }
        subbuf1 = { ctx->prealloc_x, 0, ctx->prealloc_x->size };
    }

    std::array<uint32_t, 3> elements;

    elements[0] = ncolsp2;
    elements[1] = std::min((uint32_t)ggml_nrows(src0), ctx->device->properties.limits.max_Compute_Work_ਸਮੂਹ_ਗਿਣਤੀ[1]);
    elements[2] = 1;

    // First dispatch initializes tmp_idx and does the first N passes where
    // there is only communication between threads in the same workgroup.
    {
        vk_op_argsort_ਧੱਕੋ_constants pc2 = pc;
        pc2.outer_ਸ਼ੁਰੂ = 0;
        pc2.outer_ਸਿਰਾ = std::min(ncols_pad_log2, max_wg_log2);
        pc2.inner_ਸ਼ੁਰੂ = 0;
        pc2.inner_ਸਿਰਾ = 100;
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, subbuf1, dst_buf }, pc2, elements);
    }
    ਜੇ (!use_small) {
        ggml_vk_sync_buffers(ctx, subctx);
        // Loop over outer/inner passes, synchronizing between each pass.
        ਲਈ (uint32_t outer = max_wg_log2; outer < ncols_pad_log2; ++outer) {
            ਲਈ (uint32_t inner = 0; inner < outer + 1; ++inner) {
                vk_op_argsort_ਧੱਕੋ_constants pc2 = pc;
                pc2.outer_ਸ਼ੁਰੂ = outer;
                pc2.outer_ਸਿਰਾ = outer + 1;
                pc2.inner_ਸ਼ੁਰੂ = inner;
                pc2.inner_ਸਿਰਾ = inner + 1;
                // When the inner idx is large enough, there's only communication
                // within a workgroup. So the remaining inner iterations can all
                // run in the same dispatch.
                ਜੇ (outer - inner < pipeline_idx) {
                    pc2.inner_ਸਿਰਾ = 100;
                    inner = outer;
                    pipeline = ctx->device->pipeline_argsort_large_f32[pipeline_idx];
                } ਨਹੀਂਤਾਂ {
                    // Smaller workgroup empirically seems to perform better
                    pipeline = ctx->device->pipeline_argsort_large_f32[pipeline_idx - 2];
                }
                ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
                ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, subbuf1, dst_buf }, pc2, elements);
                ggml_vk_sync_buffers(ctx, subctx);
            }
        }
        ctx->prealloc_x_need_sync = ਸੱਚ;
    }
}

ਖਾਲੀ ggml_vk_topk(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ggml_tensor * dst) {
    uint32_t ncols = src0->ne[0];
    uint32_t nrows = ggml_nrows(src0);
    uint32_t k = dst->ne[0];

    // tournament path is faster where it fits; use radix-select only past its k limit
    ਅਚਲ uint32_t k_min_pipeline = std::max((uint32_t) log2f(ਦਸ਼ਮਲਵ(k)) + 1, ctx->device->subgroup_ਮਾਪ_log2);
    ਜੇ (k_min_pipeline >= ਅੰਕ_topk_pipelines || ctx->device->pipeline_topk_f32[k_min_pipeline] == ਖਾਲੀ_ਸੂਚਕ) {
        vk_pipeline pipeline = ctx->device->pipeline_topk_radix_f32;
        GGML_ASSERT(pipeline != ਖਾਲੀ_ਸੂਚਕ);

        ਜੇ (ctx->prealloc_x_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }

        vk_op_topk_radix_ਧੱਕੋ_constants pc { ncols, k, nrows, 0, 0, 0 };
        std::array<uint32_t, 3> elements {
            pipeline->wg_denoms[0],
            std::min(nrows, ctx->device->properties.limits.max_Compute_Work_ਸਮੂਹ_ਗਿਣਤੀ[1]),
            1,
        };
        // the non-QSA path only uses bindings 0/1; bind valid buffers for the unused QSA slots
        vk_subbuffer src0_buf = ggml_vk_tensor_subbuffer(ctx, src0);
        vk_subbuffer dst_buf  = ggml_vk_tensor_subbuffer(ctx, dst);
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
            { src0_buf, dst_buf, src0_buf, src0_buf, src0_buf }, pc, elements);
        ਵਾਪਸ;
    }

    vk_op_topk_ਧੱਕੋ_constants pc { ncols, ncols, ncols, k, nrows, 0, 0 };

    ਜੇ (ctx->prealloc_x_need_sync) {
        ggml_vk_sync_buffers(ctx, subctx);
    }

    std::array<uint32_t, 3> elements;
    elements[1] = std::min(nrows, ctx->device->properties.limits.max_Compute_Work_ਸਮੂਹ_ਗਿਣਤੀ[1]);
    elements[2] = 1;

    uint32_t ਅੰਕ_elements = ncols;

    // Each iteration reduces a workgroup's worth of elements down to the K
    // largest elements. Repeat until we have the top K elements.
    // Need to do at least one iteration to write out the results.
    ਤਾਰਕਿਕ ਪੂਰਾ_one_iter = ਝੂਠ;
    uint32_t dbl_buf_ਸੂਚਕਾਂਕ = 0;
    size_t dbl_buf_ਮਾਪ;
    ਜਦੋਂ (ਅੰਕ_elements > k || !ਪੂਰਾ_one_iter) {

        // Prefer going as small as num_topk_pipelines - 3 for perf reasons.
        // But if K is larger, then we need a larger workgroup
        uint32_t max_pipeline = ਅੰਕ_topk_pipelines - 1;
        uint32_t preferred_pipeline = std::max(ਅੰਕ_topk_pipelines - 3, (uint32_t)log2f(ਦਸ਼ਮਲਵ(k)) + 2);
        max_pipeline = std::min(preferred_pipeline, max_pipeline);
        uint32_t min_pipeline = (uint32_t)log2f(ਦਸ਼ਮਲਵ(k)) + 1;
        // require full subgroup
        min_pipeline = std::max(min_pipeline, ctx->device->subgroup_ਮਾਪ_log2);

        uint32_t pipeline_idx = (uint32_t)ceilf(log2f(ਦਸ਼ਮਲਵ(ਅੰਕ_elements)));
        pipeline_idx = std::min(pipeline_idx, max_pipeline);
        pipeline_idx = std::max(pipeline_idx, min_pipeline);

        ਜੇ (ਅੰਕ_elements > (1u << pipeline_idx)) {
            // If we could finish on this loop iteration (i.e. a single workgroup)
            // then do so. It's better than the overhead of another pass.
            ਲਈ (uint32_t i = pipeline_idx; i < ਅੰਕ_topk_pipelines; ++i) {
                ਜੇ (ਅੰਕ_elements <= (1u << i)) {
                    pipeline_idx = i;
                    ਰੋਕੋ;
                }
            }
        }

        vk_pipeline pipeline = ctx->device->pipeline_topk_f32[pipeline_idx];
        // If the device doesn't support a pipeline this large, use smaller
        ਜਦੋਂ (!pipeline) {
            pipeline_idx--;
            GGML_ASSERT(pipeline_idx >= min_pipeline);
            pipeline = ctx->device->pipeline_topk_f32[pipeline_idx];
        }

        vk_op_topk_ਧੱਕੋ_constants pc2 = pc;
        pc2.ncols_ਨਿਵੇਸ਼ = ਅੰਕ_elements;

        // Number of elements remaining after this pass
        uint32_t ਅੰਕ_dst_elements = (ਅੰਕ_elements / pipeline->wg_denoms[0]) * k + std::min(k, ਅੰਕ_elements % pipeline->wg_denoms[0]);

        pc2.ncols_output = ਅੰਕ_dst_elements;

        ਜੇ (!ਪੂਰਾ_one_iter) {
            // Reserve space for ivec2 per element, double buffered
            // K per workgroup per row
            dbl_buf_ਮਾਪ = ਅੰਕ_dst_elements * nrows * 2 * ਆਕਾਰ(ਪੂਰਨ);
            dbl_buf_ਮਾਪ = ROUNDUP_POW2(dbl_buf_ਮਾਪ, ctx->device->properties.limits.min_Storage_ਬਫਰ_ਵਿੱਥ_Alignment);
            ਅਚਲ size_t x_sz = dbl_buf_ਮਾਪ * 2;

            ਜੇ (ctx->prealloc_ਮਾਪ_x < x_sz) {
                ctx->prealloc_ਮਾਪ_x = x_sz;
                ggml_vk_preallocate_buffers(ctx, subctx);
            }
        }

        vk_subbuffer src_buf;
        vk_subbuffer dst_buf;

        ਜੇ (ਅੰਕ_elements == ncols) {
            pc2.ਪਹਿਲਾ_pass = 1;
            src_buf = ggml_vk_tensor_subbuffer(ctx, src0);
        } ਨਹੀਂਤਾਂ {
            src_buf = { ctx->prealloc_x, dbl_buf_ਸੂਚਕਾਂਕ * dbl_buf_ਮਾਪ, dbl_buf_ਮਾਪ };
        }
        ਜੇ (ਅੰਕ_dst_elements == k) {
            pc2.ਆਖਰੀ_pass = 1;
            dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
        } ਨਹੀਂਤਾਂ {
            dst_buf = { ctx->prealloc_x, (dbl_buf_ਸੂਚਕਾਂਕ ^ 1) * dbl_buf_ਮਾਪ, dbl_buf_ਮਾਪ };
        }

        elements[0] = ਅੰਕ_elements;

        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src_buf, dst_buf }, pc2, elements);
        ਅੰਕ_elements = ਅੰਕ_dst_elements;
        dbl_buf_ਸੂਚਕਾਂਕ ^= 1;
        ਜੇ (ਅੰਕ_elements > k) {
            ggml_vk_sync_buffers(ctx, subctx);
        }
        ਪੂਰਾ_one_iter = ਸੱਚ;
    }
    ctx->prealloc_x_need_sync = ਸੱਚ;
}

ਖਾਲੀ ggml_vk_topk_qsa(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_cgraph * cgraph, ਪੂਰਨ ਗੰਢ_idx) {
    ਅਚਲ ggml_tensor * get_ਪੰਕਤੀਆਂ = cgraph->nodes[ਗੰਢ_idx + 0];
    ਅਚਲ ggml_tensor * ਜੋੜੋ_2      = cgraph->nodes[ਗੰਢ_idx + ctx->ਅੰਕ_additional_fused_ops - 1];
    ggml_tensor *       ਸਿਖਰ_k    = cgraph->nodes[ਗੰਢ_idx + ctx->ਅੰਕ_additional_fused_ops];

    ਅਚਲ ggml_tensor * scores   = get_ਪੰਕਤੀਆਂ->src[0]; // [n_tps, n_blocks, n_stream]
    ਅਚਲ ggml_tensor * cell_blk = get_ਪੰਕਤੀਆਂ->src[1]; // [n_kv, n_stream]

    // raw f16 mask: follow the reshape/cpy chain back to the materialized input
    ਅਚਲ ggml_tensor * ਨਕਾਬ = ਜੋੜੋ_2->src[1];
    ਜਦੋਂ (ਨਕਾਬ->op == GGML_OP_RESHAPE || ਨਕਾਬ->op == GGML_OP_CPY) {
        ਨਕਾਬ = ਨਕਾਬ->src[0];
    }

    ਅਚਲ uint32_t n_tps    = scores->ne[0];
    ਅਚਲ uint32_t n_ਖੰਡਾਂ = scores->ne[1];
    ਅਚਲ uint32_t n_ਧਾਰਾ = scores->ne[2];
    ਅਚਲ uint32_t n_kv     = cell_blk->ne[0];
    ਅਚਲ uint32_t ਚੌੜਾਈ    = ਸਿਖਰ_k->ne[0];
    ਅਚਲ uint32_t nrows    = n_tps * n_ਧਾਰਾ;

    vk_pipeline pipeline = ctx->device->pipeline_topk_radix_qsa;
    GGML_ASSERT(pipeline != ਖਾਲੀ_ਸੂਚਕ);

    // scratch holds the gathered+masked input, materialized once and reused across passes
    ਅਚਲ size_t scratch_ਮਾਪ = size_t{ n_kv } * nrows * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਜੇ (ctx->prealloc_ਮਾਪ_x < scratch_ਮਾਪ) {
        ctx->prealloc_ਮਾਪ_x = scratch_ਮਾਪ;
        ggml_vk_preallocate_buffers(ctx, subctx);
    }
    ਜੇ (ctx->prealloc_x_need_sync) {
        ggml_vk_sync_buffers(ctx, subctx);
    }

    vk_op_topk_radix_ਧੱਕੋ_constants pc { n_kv, ਚੌੜਾਈ, nrows, n_tps, n_ਖੰਡਾਂ, n_ਧਾਰਾ };
    std::array<uint32_t, 3> elements {
        pipeline->wg_denoms[0],
        std::min(nrows, ctx->device->properties.limits.max_Compute_Work_ਸਮੂਹ_ਗਿਣਤੀ[1]),
        1,
    };
    vk_subbuffer scratch_buf { ctx->prealloc_x, 0, ctx->prealloc_x->size };
    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        { ggml_vk_tensor_subbuffer(ctx, scores), ggml_vk_tensor_subbuffer(ctx, ਸਿਖਰ_k),
          ggml_vk_tensor_subbuffer(ctx, cell_blk), ggml_vk_tensor_subbuffer(ctx, ਨਕਾਬ),
          scratch_buf }, pc, elements);
    ctx->prealloc_x_need_sync = ਸੱਚ;
}

ਖਾਲੀ ggml_vk_ਜੋੜ(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_ਜੋੜ_ਪੰਕਤੀਆਂ_ਧੱਕੋ_constants p = vk_op_ਜੋੜ_ਪੰਕਤੀਆਂ_ਧੱਕੋ_constants_init(src0, dst, ggml_nelements(src0));
    ggml_vk_op_f32(ctx, subctx, src0, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, dst, GGML_OP_ਜੋੜ, p);
}

ਖਾਲੀ ggml_vk_ਜੋੜ_ਪੰਕਤੀਆਂ(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_ਜੋੜ_ਪੰਕਤੀਆਂ_ਧੱਕੋ_constants p = vk_op_ਜੋੜ_ਪੰਕਤੀਆਂ_ਧੱਕੋ_constants_init(src0, dst, src0->ne[0]);
    ggml_vk_op_f32(ctx, subctx, src0, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, dst, GGML_OP_ਜੋੜ_ਪੰਕਤੀਆਂ, p);
}

ਖਾਲੀ ggml_vk_ਔਸਤ(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_ਜੋੜ_ਪੰਕਤੀਆਂ_ਧੱਕੋ_constants p = vk_op_ਜੋੜ_ਪੰਕਤੀਆਂ_ਧੱਕੋ_constants_init(src0, dst, src0->ne[0]);
    p.ਭਾਰ = 1.0f / (ਦਸ਼ਮਲਵ)src0->ne[0];
    ggml_vk_op_f32(ctx, subctx, src0, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, dst, GGML_OP_ਔਸਤ, p);
}

ਖਾਲੀ ggml_vk_cumsum(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_ਜੋੜ_ਪੰਕਤੀਆਂ_ਧੱਕੋ_constants pc = vk_op_ਜੋੜ_ਪੰਕਤੀਆਂ_ਧੱਕੋ_constants_init(src0, dst, src0->ne[0]);
    // Use the single pass shader when the rows are small or there are enough rows to fill the GPU.
    // For fewer, larger rows, use the multipass shader to spread each row across SMs.
    ਜੇ (dst->ne[0] <= 4096 || ggml_nrows(dst) >= ctx->device->shader_core_ਗਿਣਤੀ) {
        ggml_vk_op_f32(ctx, subctx, src0, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, dst, GGML_OP_CUMSUM, pc);
        ਵਾਪਸ;
    }

    // First pass computes partial sums within a block, and stores the last partial
    // to the temp buffer. Second pass sums the block partials from the temp buffer
    // and adds that to the result of the first pass.
    vk_pipeline pipeline1 = ctx->device->pipeline_cumsum_multipass1_f32;
    vk_pipeline pipeline2 = ctx->device->pipeline_cumsum_multipass2_f32;
    GGML_ASSERT(pipeline1 != ਖਾਲੀ_ਸੂਚਕ && pipeline2 != ਖਾਲੀ_ਸੂਚਕ);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline1, 1);
    ggml_pipeline_request_descriptor_sets(ctx, pipeline2, 1);

    std::array<uint32_t, 3> elements;

    elements[0] = dst->ne[0];
    elements[1] = (uint32_t)ggml_nrows(dst);
    elements[2] = 1;

    size_t ਆਰਜ਼ੀ_ਮਾਪ = ਆਕਾਰ(ਦਸ਼ਮਲਵ) * elements[0] * ggml_nrows(dst);

    ਜੇ (ctx->prealloc_ਮਾਪ_ਵੰਡੋ_k < ਆਰਜ਼ੀ_ਮਾਪ) {
        ctx->prealloc_ਮਾਪ_ਵੰਡੋ_k = ਆਰਜ਼ੀ_ਮਾਪ;
        ggml_vk_preallocate_buffers(ctx, subctx);
    }

    vk_subbuffer src_buf = ggml_vk_tensor_subbuffer(ctx, src0);
    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
    vk_subbuffer ਆਰਜ਼ੀ_buf = ggml_vk_subbuffer(ctx, ctx->prealloc_ਵੰਡੋ_k, 0);

    ਜੇ (ctx->prealloc_ਵੰਡੋ_k_need_sync) {
        ggml_vk_sync_buffers(ctx, subctx);
    }

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline1, {src_buf, dst_buf, ਆਰਜ਼ੀ_buf}, pc, elements);
    ggml_vk_sync_buffers(ctx, subctx);
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline2, {src_buf, dst_buf, ਆਰਜ਼ੀ_buf}, pc, elements);

    ctx->prealloc_ਵੰਡੋ_k_need_sync = ਸੱਚ;
}

ਸਥਿਰ std::array<uint32_t, 3> ggml_vk_nrows_elements(uint32_t nr) {
    ਜੇ (nr > 262144) {
        ਵਾਪਸ { 512, 512, CEIL_DIV(nr, 262144) };
    }
    ਜੇ (nr > 512) {
        ਵਾਪਸ { 512, CEIL_DIV(nr, 512), 1 };
    }
    ਵਾਪਸ { nr, 1, 1 };
}

ਖਾਲੀ ggml_vk_cross_entropy_loss(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_scalar(dst));

    ਅਚਲ uint32_t nclasses = (uint32_t)src0->ne[0];
    ਅਚਲ uint32_t nrows    = (uint32_t)ggml_nrows(src0);

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, src0, src1, ਖਾਲੀ_ਸੂਚਕ, dst, GGML_OP_CROSS_ENTROPY_LOSS);
    GGML_ASSERT(pipeline != ਖਾਲੀ_ਸੂਚਕ);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    ggml_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_ਜੋੜ_ਪੰਕਤੀਆਂ_f32, 1);

    vk_subbuffer src0_buf = ggml_vk_tensor_subbuffer(ctx, src0);
    vk_subbuffer src1_buf = ggml_vk_tensor_subbuffer(ctx, src1);
    vk_subbuffer dst_buf  = ggml_vk_tensor_subbuffer(ctx, dst, ਸੱਚ);

    ਅਚਲ vk_op_ਧੱਕੋ_constants pc = { nclasses, nrows, 0.0f, 0.0f, 0.0f, 0.0f };

    ਅਚਲ size_t tmp_ਮਾਪ = (size_t)nrows * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਜੇ (ctx->prealloc_ਮਾਪ_x < tmp_ਮਾਪ) {
        ctx->prealloc_ਮਾਪ_x = tmp_ਮਾਪ;
        ggml_vk_preallocate_buffers(ctx, subctx);
    }
    ਜੇ (ctx->prealloc_x_need_sync) {
        ggml_vk_sync_buffers(ctx, subctx);
    }

    vk_subbuffer tmp_buf = { ctx->prealloc_x, 0, tmp_ਮਾਪ };
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, tmp_buf }, pc, ggml_vk_nrows_elements(nrows));
    ggml_vk_sync_buffers(ctx, subctx);

    vk_op_ਜੋੜ_ਪੰਕਤੀਆਂ_ਧੱਕੋ_constants sp = {};
    sp.n_ਖਾਨੇ_ਗਿਣਤੀ = nrows;
    sp.ne01 = 1;
    sp.ne02 = 1;
    sp.ਭਾਰ = 1.0f;
    init_pushconst_fastdiv(sp);
    sp.misalign_offsets = get_misalign_ਬਾਈਟਾਂ(ctx, dst) / ggml_type_ਮਾਪ(dst->type);

    ggml_vk_dispatch_pipeline(ctx, subctx, ctx->device->pipeline_ਜੋੜ_ਪੰਕਤੀਆਂ_f32, { tmp_buf, dst_buf }, sp, { 1, 1, 1 });
    ctx->prealloc_x_need_sync = ਸੱਚ;
}

ਖਾਲੀ ggml_vk_cross_entropy_loss_ਪਿੱਠ(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    ਅਚਲ ggml_tensor * grad   = dst->src[0];
    ਅਚਲ ggml_tensor * logits = dst->src[1];
    ਅਚਲ ggml_tensor * labels = dst->src[2];

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

    ਅਚਲ uint32_t nclasses = (uint32_t)logits->ne[0];
    ਅਚਲ uint32_t nrows    = (uint32_t)ggml_nrows(logits);

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, grad, logits, labels, dst, GGML_OP_CROSS_ENTROPY_LOSS_ਪਿੱਠ);
    GGML_ASSERT(pipeline != ਖਾਲੀ_ਸੂਚਕ);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer grad_buf   = ggml_vk_tensor_subbuffer(ctx, grad);
    vk_subbuffer logits_buf = ggml_vk_tensor_subbuffer(ctx, logits);
    vk_subbuffer labels_buf = ggml_vk_tensor_subbuffer(ctx, labels);
    vk_subbuffer dst_buf    = ggml_vk_tensor_subbuffer(ctx, dst);

    ਅਚਲ vk_op_ਧੱਕੋ_constants pc = { nclasses, nrows, 0.0f, 0.0f, 0.0f, 0.0f };
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { grad_buf, logits_buf, labels_buf, dst_buf }, pc, ggml_vk_nrows_elements(nrows));
}

ਖਾਲੀ ggml_vk_argmax(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ggml_tensor * dst) {
    ggml_vk_op_f32<vk_op_ਧੱਕੋ_constants>(ctx, subctx, src0, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, dst, GGML_OP_ARGMAX, { (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], 0.0f, 0.0f, 0.0f, 0.0f });
}

ਖਾਲੀ ggml_vk_ਗਿਣਤੀ_equal(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ggml_tensor * dst) {
    ggml_vk_op_f32<vk_op_ਧੱਕੋ_constants>(ctx, subctx, src0, src1, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, dst, GGML_OP_ਗਿਣਤੀ_EQUAL, { (uint32_t)ggml_nelements(src0), 0, 0.0f, 0.0f, 0.0f, 0.0f });
}

ਖਾਲੀ ggml_vk_solve_tri(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ggml_tensor * dst) {
    ਅਚਲ uint32_t src0_type_ਮਾਪ = ggml_type_ਮਾਪ(src0->type);
    ਅਚਲ uint32_t src1_type_ਮਾਪ = ggml_type_ਮਾਪ(src1->type);
    ਅਚਲ uint32_t dst_type_ਮਾਪ = ggml_type_ਮਾਪ(dst->type);

    ggml_vk_op_f32<vk_op_ਦੁਆਧਾਰੀ_ਧੱਕੋ_constants>(ctx, subctx, src0, src1, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, dst, GGML_OP_SOLVE_TRI, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_ਮਾਪ, (uint32_t)src0->nb[1] / src0_type_ਮਾਪ, (uint32_t)src0->nb[2] / src0_type_ਮਾਪ, (uint32_t)src0->nb[3] / src0_type_ਮਾਪ,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_ਮਾਪ, (uint32_t)src1->nb[1] / src1_type_ਮਾਪ, (uint32_t)src1->nb[2] / src1_type_ਮਾਪ, (uint32_t)src1->nb[3] / src1_type_ਮਾਪ,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_ਮਾਪ, (uint32_t) dst->nb[1] /  dst_type_ਮਾਪ, (uint32_t) dst->nb[2] /  dst_type_ਮਾਪ, (uint32_t) dst->nb[3] /  dst_type_ਮਾਪ,
        0,
        0.0f, 0.0f, 0,
    });
}

ਖਾਲੀ ggml_vk_im2col(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ggml_tensor * dst) {
    ਅਚਲ int32_t s0 = dst->op_params[0];
    ਅਚਲ int32_t s1 = dst->op_params[1];
    ਅਚਲ int32_t p0 = dst->op_params[2];
    ਅਚਲ int32_t p1 = dst->op_params[3];
    ਅਚਲ int32_t d0 = dst->op_params[4];
    ਅਚਲ int32_t d1 = dst->op_params[5];

    ਅਚਲ ਤਾਰਕਿਕ is_2D = dst->op_params[6] == 1;

    ਅਚਲ uint32_t IC = src1->ne[is_2D ? 2 : 1];
    ਅਚਲ uint32_t IH = is_2D ? src1->ne[1] : 1;
    ਅਚਲ uint32_t IW =         src1->ne[0];

    ਅਚਲ uint32_t KH = is_2D ? src0->ne[1] : 1;
    ਅਚਲ uint32_t KW =         src0->ne[0];

    ਅਚਲ uint32_t OH = is_2D ? dst->ne[2] : 1;
    ਅਚਲ uint32_t OW =         dst->ne[1];

    ਅਚਲ uint32_t ਵਿੱਥ_delta = src1->nb[is_2D ? 2 : 1] / 4; // nb is byte offset, src is type float32
    ਅਚਲ uint32_t batch_ਵਿੱਥ = src1->nb[is_2D ? 3 : 2] / 4; // nb is byte offset, src is type float32

    ਅਚਲ uint32_t batch = src1->ne[is_2D ? 3 : 2];

    ਅਚਲ ggml_backend_vk_ਬਫਰ_context * d_buf_ctx = (ggml_backend_vk_ਬਫਰ_context *)dst->ਬਫਰ->context;
    ਅਚਲ vk_ਬਫਰ d_buf = d_buf_ctx->dev_ਬਫਰ;

    ਅਚਲ vk::Device_ਪਤਾ dst_addr = d_buf->bda_addr + vk_tensor_ਵਿੱਥ(dst) + dst->view_offs;

    ggml_vk_op_f32<vk_op_im2col_ਧੱਕੋ_constants>(ctx, subctx, src0, src1, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, dst, GGML_OP_IM2_ਖਾਨਾ, {
        dst_addr,
        batch_ਵਿੱਥ, ਵਿੱਥ_delta,
        IC, IW, IH, OW, OH, KW, KH,
        OH * batch,
        IC * KH * KW,
        s0, s1, p0, p1, d0, d1, batch * IC
    });
}

ਖਾਲੀ ggml_vk_im2col_3d(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ggml_tensor * dst) {
    GGML_TENSOR_ਦੁਆਧਾਰੀ_OP_LOCALS

    ਅਚਲ int32_t s0 = ((ਅਚਲ int32_t *)(dst->op_params))[0];
    ਅਚਲ int32_t s1 = ((ਅਚਲ int32_t *)(dst->op_params))[1];
    ਅਚਲ int32_t s2 = ((ਅਚਲ int32_t *)(dst->op_params))[2];
    ਅਚਲ int32_t p0 = ((ਅਚਲ int32_t *)(dst->op_params))[3];
    ਅਚਲ int32_t p1 = ((ਅਚਲ int32_t *)(dst->op_params))[4];
    ਅਚਲ int32_t p2 = ((ਅਚਲ int32_t *)(dst->op_params))[5];
    ਅਚਲ int32_t d0 = ((ਅਚਲ int32_t *)(dst->op_params))[6];
    ਅਚਲ int32_t d1 = ((ਅਚਲ int32_t *)(dst->op_params))[7];
    ਅਚਲ int32_t d2 = ((ਅਚਲ int32_t *)(dst->op_params))[8];
    ਅਚਲ int32_t IC = ((ਅਚਲ int32_t *)(dst->op_params))[9];

    ਅਚਲ int64_t N  = ne13 / IC;
    ਅਚਲ int64_t ID = ne12;
    ਅਚਲ int64_t IH = ne11;
    ਅਚਲ int64_t IW = ne10;

    ਅਚਲ int64_t KD = ne02;
    ਅਚਲ int64_t KH = ne01;
    ਅਚਲ int64_t KW = ne00;

    ਅਚਲ int64_t OD = ne3 / N;
    ਅਚਲ int64_t OH = ne2;
    ਅਚਲ int64_t OW = ne1;

    ਅਚਲ ggml_backend_vk_ਬਫਰ_context * d_buf_ctx = (ggml_backend_vk_ਬਫਰ_context *)dst->ਬਫਰ->context;
    ਅਚਲ vk_ਬਫਰ d_buf = d_buf_ctx->dev_ਬਫਰ;

    ਅਚਲ vk::Device_ਪਤਾ dst_addr = d_buf->bda_addr + vk_tensor_ਵਿੱਥ(dst) + dst->view_offs;

    vk_op_im2col_3d_ਧੱਕੋ_constants pc {};

    pc.dst_addr = dst_addr;
    pc.nb10 = nb10 / ggml_type_ਮਾਪ(src1->type);
    pc.nb11 = nb11 / ggml_type_ਮਾਪ(src1->type);
    pc.nb12 = nb12 / ggml_type_ਮਾਪ(src1->type);
    pc.nb13 = nb13 / ggml_type_ਮਾਪ(src1->type);
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

    ggml_vk_op_f32<vk_op_im2col_3d_ਧੱਕੋ_constants>(ctx, subctx, src0, src1, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, dst, GGML_OP_IM2_ਖਾਨਾ_3_D, std::move(pc));
}

ਖਾਲੀ ggml_vk_timestep_embedding(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ggml_tensor * dst) {
    ਅਚਲ uint32_t dim = dst->op_params[0];
    ਅਚਲ uint32_t max_period = dst->op_params[1];
    ਅਚਲ uint32_t nb1 = dst->nb[1] / ggml_type_ਮਾਪ(dst->type);

    ggml_vk_op_f32<vk_op_timestep_embedding_ਧੱਕੋ_constants>(ctx, subctx, src0, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, dst, GGML_OP_TIMESTEP_EMBEDDING, {
        nb1, dim, max_period,
    });
}

ਖਾਲੀ ggml_vk_conv_transpose_1d(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ggml_tensor * dst) {
    // src0: (K, Cout, Cin, 1) -- kernel
    // src1: (L, Cin, 1, 1) -- input
    // dst: (*, Cout, 1, 1)

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ਦੁਆਧਾਰੀ_OP_LOCALS

    GGML_ASSERT(nb00 == ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    GGML_ASSERT(nb10 == ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਅਚਲ int32_t s0 = dst->op_params[0];

    vk_op_conv_transpose_1d_ਧੱਕੋ_constants p{};
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

    ggml_vk_op_f32(ctx, subctx, src0, src1, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, dst, GGML_OP_CONV_TRANSPOSE_1D, std::move(p));
}

ਖਾਲੀ ggml_vk_col2im_1d(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ggml_tensor * dst) {
    // src0: [K_OC, T_in] columns from matmul
    // dst:  [T_out, OC]

    ਅਚਲ int32_t stride = dst->op_params[0];
    ਅਚਲ int32_t oc     = dst->op_params[1];
    ਅਚਲ int32_t p0     = dst->op_params[2];

    ਅਚਲ uint32_t K_OC  = static_cast<uint32_t>(src0->ne[0]);
    ਅਚਲ uint32_t T_ਵਿੱਚ  = static_cast<uint32_t>(src0->ne[1]);
    ਅਚਲ uint32_t T_ਬਾਹਰਲਾ = static_cast<uint32_t>(dst->ne[0]);
    ਅਚਲ uint32_t OC    = static_cast<uint32_t>(oc);
    ਅਚਲ uint32_t K     = K_OC / OC;

    vk_op_col2im_1d_ਧੱਕੋ_constants p{};
    p.T_ਬਾਹਰਲਾ  = T_ਬਾਹਰਲਾ;
    p.OC     = OC;
    p.K_OC   = K_OC;
    p.T_ਵਿੱਚ   = T_ਵਿੱਚ;
    p.K      = K;
    p.stride = stride;
    p.p0     = p0;

    ggml_vk_op_f32(ctx, subctx, src0, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, dst, GGML_OP_COL2IM_1D, std::move(p));
}

ਖਾਲੀ ggml_vk_snake_dispatch_fused(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_cgraph * cgraph, ਪੂਰਨ ਗੰਢ_idx) {
    ਅਚਲ ggml_tensor * mul0 = cgraph->nodes[ਗੰਢ_idx + 0];
    ਅਚਲ ggml_tensor * sqr  = cgraph->nodes[ਗੰਢ_idx + 2];
    ਅਚਲ ggml_tensor * mul1 = cgraph->nodes[ਗੰਢ_idx + 3];
    ggml_tensor *       ਜੋੜੋ_2  = cgraph->nodes[ਗੰਢ_idx + 4];

    // x carries the full activation shape, a is the broadcast operand
    ਅਚਲ ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
    ਅਚਲ ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];

    // mul1 reads sqr and inv_b in either operand order
    ਅਚਲ ggml_tensor * inv_b = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];

    vk_pipeline pipeline = ਖਾਲੀ_ਸੂਚਕ;
    ਚੋਣ (x->type) {
        ਹਾਲਤ GGML_TYPE_F32:  pipeline = ctx->device->pipeline_snake_f32;  ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F16:  pipeline = ctx->device->pipeline_snake_f16;  ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_BF16: pipeline = ctx->device->pipeline_snake_bf16; ਰੋਕੋ;
        ਮੂਲ:             GGML_ABORT("unsupported type");
    }
    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer x_buf     = ggml_vk_tensor_subbuffer(ctx, x);
    vk_subbuffer a_buf     = ggml_vk_tensor_subbuffer(ctx, a);
    vk_subbuffer inv_b_buf = ggml_vk_tensor_subbuffer(ctx, inv_b);
    vk_subbuffer dst_buf   = ggml_vk_tensor_subbuffer(ctx, ਜੋੜੋ_2);

    vk_op_snake_ਧੱਕੋ_constants pc{};
    pc.ne0 = static_cast<uint32_t>(x->ne[0]);
    pc.ne1 = static_cast<uint32_t>(x->ne[1]);

    std::array<uint32_t, 3> elements = { pc.ne0, pc.ne1, 1 };
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { x_buf, a_buf, inv_b_buf, dst_buf }, pc, elements);
}

ਖਾਲੀ ggml_vk_pool_1d(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ggml_tensor * dst) {
    uint32_t op = static_cast<uint32_t>(dst->op_params[0]);
    ਅਚਲ int32_t k0 = dst->op_params[1];
    ਅਚਲ int32_t s0 = dst->op_params[2];
    ਅਚਲ int32_t p0 = dst->op_params[3];

    ਅਚਲ uint32_t IL = src0->ne[0];

    ਅਚਲ uint32_t N = dst->ne[3] * dst->ne[2];

    ਅਚਲ uint32_t OC = dst->ne[1];
    ਅਚਲ uint32_t OL = dst->ne[0];

    ਅਚਲ uint32_t parallel_elements = N * OC * OL;

    ggml_vk_op_f32<vk_op_pool1d_ਧੱਕੋ_constants>(ctx, subctx, src0, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, dst, GGML_OP_POOL_1D, {
        IL, OL, OC,
        parallel_elements,
        op,
        k0, s0, p0,
    });
}

ਖਾਲੀ ggml_vk_pool_2d(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ggml_tensor * dst) {
    uint32_t op = static_cast<uint32_t>(dst->op_params[0]);
    ਅਚਲ int32_t k1 = dst->op_params[1];
    ਅਚਲ int32_t k0 = dst->op_params[2];
    ਅਚਲ int32_t s1 = dst->op_params[3];
    ਅਚਲ int32_t s0 = dst->op_params[4];
    ਅਚਲ int32_t p1 = dst->op_params[5];
    ਅਚਲ int32_t p0 = dst->op_params[6];

    ਅਚਲ uint32_t IH = src0->ne[1];
    ਅਚਲ uint32_t IW = src0->ne[0];

    ਅਚਲ uint32_t N = dst->ne[3];

    ਅਚਲ uint32_t OC = dst->ne[2];
    ਅਚਲ uint32_t OH = dst->ne[1];
    ਅਚਲ uint32_t OW = dst->ne[0];

    ਅਚਲ uint32_t parallel_elements = N * OC * OH * OW;

    ggml_vk_op_f32<vk_op_pool2d_ਧੱਕੋ_constants>(ctx, subctx, src0, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, dst, GGML_OP_POOL_2D, {
        IW, IH, OW, OH, OC,
        parallel_elements,
        op,
        k0, k1, s0, s1, p0, p1,
    });
}

ਖਾਲੀ ggml_vk_conv_2d(ggml_backend_vk_context * ctx, vk_context & subctx, ਅਚਲ ggml_tensor * src0,
                            ਅਚਲ ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ਦੁਆਧਾਰੀ_OP_LOCALS
    GGML_ASSERT(nb00 == ਆਕਾਰ(ਦਸ਼ਮਲਵ) || nb00 == ਆਕਾਰ(ggml_fp16_t));
    GGML_ASSERT(nb10 == ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    GGML_ASSERT(nb0 == ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਤਾਰਕਿਕ transpose = dst->op == GGML_OP_CONV_TRANSPOSE_2D;

    vk_op_conv2d_ਧੱਕੋ_constants p{};
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

    ggml_vk_op_f32(ctx, subctx, src0, src1, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, dst, dst->op, std::move(p));
}

ਖਾਲੀ ggml_vk_conv_3d(ggml_backend_vk_context * ctx, vk_context & subctx, ਅਚਲ ggml_tensor * src0,
                            ਅਚਲ ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ਦੁਆਧਾਰੀ_OP_LOCALS
    GGML_ASSERT(nb00 == ਆਕਾਰ(ਦਸ਼ਮਲਵ) || nb00 == ਆਕਾਰ(ggml_fp16_t));
    GGML_ASSERT(nb10 == ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    GGML_ASSERT(nb0 == ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    vk_op_conv3d_ਧੱਕੋ_constants p{};
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

    ggml_vk_op_f32(ctx, subctx, src0, src1, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, dst, GGML_OP_CONV_3D, std::move(p));
}

ਖਾਲੀ ggml_vk_conv_2d_dw(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ggml_tensor * dst) {
    vk_op_conv2d_dw_ਧੱਕੋ_constants p{};
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

    ggml_vk_op_f32(ctx, subctx, src0, src1, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, dst, GGML_OP_CONV_2D_DW, std::move(p));
}

ਖਾਲੀ ggml_vk_leaky_relu(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ggml_tensor * dst) {
    ਅਚਲ ਦਸ਼ਮਲਵ * op_params = (ਅਚਲ ਦਸ਼ਮਲਵ *)dst->op_params;
    vk_op_unary_ਧੱਕੋ_constants p = vk_op_unary_ਧੱਕੋ_constants_init(src0, dst);
    p.param1 = op_params[0];

    ggml_vk_op_f32(ctx, subctx, src0, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, dst, GGML_OP_LEAKY_RELU, std::move(p));
}

ਖਾਲੀ ggml_vk_preallocate_buffers(ggml_backend_vk_context * ctx, vk_context subctx) {
#ਜੇ defined(GGML_VULKAN_ਲੜੀਵਾਰ_TESTS)
    ਅਚਲ std::vector<size_t> vals {
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
    ਅਚਲ size_t ਅੰਕ_it = 100;

    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, ਅੰਕ_it, 1, 0, GGML_TYPE_Q4_0);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, ਅੰਕ_it, 1, 1, GGML_TYPE_Q4_0);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, ਅੰਕ_it, 1, 2, GGML_TYPE_Q4_0);

    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, ਅੰਕ_it, 1, 0, GGML_TYPE_Q4_0, ਸੱਚ);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, ਅੰਕ_it, 1, 1, GGML_TYPE_Q4_0, ਸੱਚ);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, ਅੰਕ_it, 1, 2, GGML_TYPE_Q4_0, ਸੱਚ);

    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, ਅੰਕ_it, 1, 0, GGML_TYPE_Q8_0);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, ਅੰਕ_it, 1, 1, GGML_TYPE_Q8_0);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, ਅੰਕ_it, 1, 2, GGML_TYPE_Q8_0);

    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, ਅੰਕ_it, 1, 0, GGML_TYPE_Q8_0, ਸੱਚ);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, ਅੰਕ_it, 1, 1, GGML_TYPE_Q8_0, ਸੱਚ);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, ਅੰਕ_it, 1, 2, GGML_TYPE_Q8_0, ਸੱਚ);

    abort();

    ਲਈ (size_t i = 0; i < vals.size(); i += 3) {
        ggml_vk_test_matmul<ggml_fp16_t, ਦਸ਼ਮਲਵ>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, ਅੰਕ_it, 1, 0);
        ggml_vk_test_matmul<ggml_fp16_t, ਦਸ਼ਮਲਵ>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, ਅੰਕ_it, 1, 1);
        ggml_vk_test_matmul<ggml_fp16_t, ਦਸ਼ਮਲਵ>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, ਅੰਕ_it, 1, 2);
        std::cerr << '\n';
        ggml_vk_test_matmul<ggml_fp16_t, ਦਸ਼ਮਲਵ>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, ਅੰਕ_it, 2, 0);
        ggml_vk_test_matmul<ggml_fp16_t, ਦਸ਼ਮਲਵ>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, ਅੰਕ_it, 2, 1);
        ggml_vk_test_matmul<ggml_fp16_t, ਦਸ਼ਮਲਵ>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, ਅੰਕ_it, 2, 2);
        std::cerr << '\n';
        ggml_vk_test_matmul<ggml_fp16_t, ਦਸ਼ਮਲਵ>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, ਅੰਕ_it, 4, 0);
        ggml_vk_test_matmul<ggml_fp16_t, ਦਸ਼ਮਲਵ>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, ਅੰਕ_it, 4, 1);
        ggml_vk_test_matmul<ggml_fp16_t, ਦਸ਼ਮਲਵ>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, ਅੰਕ_it, 4, 2);
        std::cerr << '\n' << std::ਸਤਰ_ਅੰਤ;

        ਜੇ (vals[i + 2] % 32 == 0) {
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, ਅੰਕ_it, 1, 0, GGML_TYPE_Q4_0);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, ਅੰਕ_it, 1, 1, GGML_TYPE_Q4_0);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, ਅੰਕ_it, 1, 2, GGML_TYPE_Q4_0);
            std::cerr << '\n';
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, ਅੰਕ_it, 2, 0, GGML_TYPE_Q4_0);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, ਅੰਕ_it, 2, 1, GGML_TYPE_Q4_0);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, ਅੰਕ_it, 2, 2, GGML_TYPE_Q4_0);
            std::cerr << '\n';
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, ਅੰਕ_it, 4, 0, GGML_TYPE_Q4_0);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, ਅੰਕ_it, 4, 1, GGML_TYPE_Q4_0);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, ਅੰਕ_it, 4, 2, GGML_TYPE_Q4_0);
            std::cerr << '\n' << std::ਸਤਰ_ਅੰਤ;
        }

        ਜੇ (vals[i + 2] % 256 == 0) {
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, ਅੰਕ_it, 1, 0, GGML_TYPE_Q4_K);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, ਅੰਕ_it, 1, 1, GGML_TYPE_Q4_K);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, ਅੰਕ_it, 1, 2, GGML_TYPE_Q4_K);
            std::cerr << '\n';
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, ਅੰਕ_it, 2, 0, GGML_TYPE_Q4_K);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, ਅੰਕ_it, 2, 1, GGML_TYPE_Q4_K);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, ਅੰਕ_it, 2, 2, GGML_TYPE_Q4_K);
            std::cerr << '\n';
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, ਅੰਕ_it, 4, 0, GGML_TYPE_Q4_K);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, ਅੰਕ_it, 4, 1, GGML_TYPE_Q4_K);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, ਅੰਕ_it, 4, 2, GGML_TYPE_Q4_K);
            std::cerr << '\n' << std::ਸਤਰ_ਅੰਤ;
        }
    }

    GGML_ABORT("fatal error");
#ਜੇ_ਅੰਤ

    ਜੇ (subctx) {
        // Submit and wait for any pending work before reallocating the buffers
        ggml_vk_ctx_ਸਿਰਾ(subctx);
        ggml_vk_submit(subctx, {});
        ctx->submit_pending = ਸੱਚ;
        ggml_vk_synchronize(ctx);
        GGML_ASSERT(ctx->compute_ctx.expired());
        ggml_vk_ctx_begin(ctx->device, subctx);
        ctx->compute_ctx = subctx;
    }

    ਜੇ (ctx->prealloc_x == ਖਾਲੀ_ਸੂਚਕ || (ctx->prealloc_ਮਾਪ_x > 0 && ctx->prealloc_x->size < ctx->prealloc_ਮਾਪ_x)) {
        VK_LOG_ਸਮ੍ਰਿਤੀ("ggml_vk_preallocate_buffers(x_size: " << ctx->prealloc_ਮਾਪ_x << ")");
        // Resize buffer
        ਜੇ (ctx->prealloc_x != ਖਾਲੀ_ਸੂਚਕ) {
            ggml_vk_destroy_ਬਫਰ(ctx->prealloc_x);
        }
        ctx->prealloc_x = ggml_vk_create_ਬਫਰ_device(ctx->device, ctx->prealloc_ਮਾਪ_x);
    }
    ਜੇ (ctx->prealloc_y == ਖਾਲੀ_ਸੂਚਕ || (ctx->prealloc_ਮਾਪ_y > 0 && ctx->prealloc_y->size < ctx->prealloc_ਮਾਪ_y)) {
        VK_LOG_ਸਮ੍ਰਿਤੀ("ggml_vk_preallocate_buffers(y_size: " << ctx->prealloc_ਮਾਪ_y << ")");
        // Resize buffer
        ਜੇ (ctx->prealloc_y != ਖਾਲੀ_ਸੂਚਕ) {
            ggml_vk_destroy_ਬਫਰ(ctx->prealloc_y);
        }
        ctx->prealloc_y = ggml_vk_create_ਬਫਰ_device(ctx->device, ctx->prealloc_ਮਾਪ_y);
        ctx->prealloc_y_ਆਖਰੀ_pipeline_ਵਰਤਿਆ = ਖਾਲੀ_ਸੂਚਕ;
        ctx->prealloc_y_ਆਖਰੀ_tensor_ਵਰਤਿਆ = ਖਾਲੀ_ਸੂਚਕ;
        ctx->prealloc_y_ਆਖਰੀ_k_padded = ਝੂਠ;
    }
    ਜੇ (ctx->prealloc_ਵੰਡੋ_k == ਖਾਲੀ_ਸੂਚਕ || (ctx->prealloc_ਮਾਪ_ਵੰਡੋ_k > 0 && ctx->prealloc_ਵੰਡੋ_k->size < ctx->prealloc_ਮਾਪ_ਵੰਡੋ_k)) {
        VK_LOG_ਸਮ੍ਰਿਤੀ("ggml_vk_preallocate_buffers(split_k_size: " << ctx->prealloc_ਮਾਪ_ਵੰਡੋ_k << ")");
        // Resize buffer
        ਜੇ (ctx->prealloc_ਵੰਡੋ_k != ਖਾਲੀ_ਸੂਚਕ) {
            ggml_vk_destroy_ਬਫਰ(ctx->prealloc_ਵੰਡੋ_k);
        }
        ctx->prealloc_ਵੰਡੋ_k = ggml_vk_create_ਬਫਰ_device(ctx->device, ctx->prealloc_ਮਾਪ_ਵੰਡੋ_k);
    }
    ਜੇ (ctx->prealloc_ਜੋੜੋ_rms_partials == ਖਾਲੀ_ਸੂਚਕ || (ctx->prealloc_ਮਾਪ_ਜੋੜੋ_rms_partials > 0 && ctx->prealloc_ਜੋੜੋ_rms_partials->size < ctx->prealloc_ਮਾਪ_ਜੋੜੋ_rms_partials)) {
        VK_LOG_ਸਮ੍ਰਿਤੀ("ggml_vk_preallocate_buffers(add_partials_size: " << ctx->prealloc_ਜੋੜੋ_rms_partials << ")");
        // Resize buffer
        ਜੇ (ctx->prealloc_ਜੋੜੋ_rms_partials != ਖਾਲੀ_ਸੂਚਕ) {
            ggml_vk_destroy_ਬਫਰ(ctx->prealloc_ਜੋੜੋ_rms_partials);
        }
        ctx->prealloc_ਜੋੜੋ_rms_partials = ggml_vk_create_ਬਫਰ_device(ctx->device, ctx->prealloc_ਮਾਪ_ਜੋੜੋ_rms_partials);
    }
}

ਤਾਰਕਿਕ ggml_vk_build_ਗ੍ਰਾਫ(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, ਪੂਰਨ ਗੰਢ_idx, ggml_tensor *ਗੰਢ_begin, ਪੂਰਨ ਗੰਢ_idx_begin, ਤਾਰਕਿਕ ਆਖਰੀ_ਗੰਢ, ਤਾਰਕਿਕ almost_ready, ਤਾਰਕਿਕ submit){
    ggml_tensor * ਗੰਢ = cgraph->nodes[ਗੰਢ_idx];
    ਜੇ (ggml_is_ਖਾਲੀ_ਥਾਵਾਂ(ਗੰਢ) || ggml_op_is_ਖਾਲੀ_ਥਾਵਾਂ(ਗੰਢ->op) || !ਗੰਢ->ਬਫਰ) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ ((ਗੰਢ->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
        ਵਾਪਸ ਝੂਠ;
    }

    VK_LOG_DEBUG("ggml_vk_build_graph(" << ਗੰਢ << ", " << ggml_op_name(ਗੰਢ->op) << ")");
    ctx->semaphore_idx = 0;

    ggml_tensor * src0 = ਗੰਢ->src[0];
    ggml_tensor * src1 = ਗੰਢ->src[1];
    ggml_tensor * src2 = ਗੰਢ->src[2];
    ggml_tensor * src3 = ਗੰਢ->src[3];

    ਜੇ (ਗੰਢ->op == GGML_OP_ਜੋੜੋ) {
        ਪੂਰਨ ਅਗਲਾ_ਗੰਢ_idx = ਗੰਢ_idx + 1 + ctx->ਅੰਕ_additional_fused_ops;
        ਜੇ (ਅਗਲਾ_ਗੰਢ_idx < cgraph->n_nodes &&
            cgraph->nodes[ਅਗਲਾ_ਗੰਢ_idx]->op == GGML_OP_RMS_NORM &&
            cgraph->nodes[ਅਗਲਾ_ਗੰਢ_idx]->src[0] == cgraph->nodes[ਅਗਲਾ_ਗੰਢ_idx - 1] &&
            ggml_nrows(cgraph->nodes[ਅਗਲਾ_ਗੰਢ_idx]) == 1 &&
            ctx->device->ਜੋੜੋ_rms_fusion) {
            uint32_t size = ggml_vk_rms_partials_ਮਾਪ(ctx, cgraph->nodes[ਗੰਢ_idx]);
            ctx->do_ਜੋੜੋ_rms_partials_ਵਿੱਥ_calculation = ਸੱਚ;
            ਜੇ (ctx->prealloc_ਮਾਪ_ਜੋੜੋ_rms_partials_ਵਿੱਥ + size <= ctx->prealloc_ਮਾਪ_ਜੋੜੋ_rms_partials) {
                ctx->do_ਜੋੜੋ_rms_partials = ਸੱਚ;
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
        ਤਾਰਕਿਕ need_sync = ਝੂਠ;

        // Check whether "node" requires synchronization. The node requires synchronization if it
        // overlaps in memory with another unsynchronized node and at least one of them is a write.
        // Destination nodes are checked against both the written/read lists. Source nodes are only
        // checked against the written list. Two nodes overlap in memory if they come from the same
        // buffer and the tensor or view ranges overlap.
        ਸਵੈਚਾਲਿਤ ਅਚਲ &overlaps_unsynced = [&](ਅਚਲ ggml_tensor *ਗੰਢ, ਅਚਲ std::vector<ਅਚਲ ggml_tensor *> &unsynced_nodes) -> ਤਾਰਕਿਕ {
            ਜੇ (unsynced_nodes.size() == 0) {
                ਵਾਪਸ ਝੂਠ;
            }
            ਸਵੈਚਾਲਿਤ n_ਅਧਾਰ = vk_tensor_ਵਿੱਥ(ਗੰਢ) + ਗੰਢ->view_offs;
            ਸਵੈਚਾਲਿਤ n_ਮਾਪ = ggml_nbytes(ਗੰਢ);
            ggml_backend_vk_ਬਫਰ_context * a_buf_ctx = (ggml_backend_vk_ਬਫਰ_context *)ਗੰਢ->ਬਫਰ->context;
            vk_ਬਫਰ a_buf = a_buf_ctx->dev_ਬਫਰ;
            ਲਈ (ਸਵੈਚਾਲਿਤ &other : unsynced_nodes) {
                ggml_backend_vk_ਬਫਰ_context * o_buf_ctx = (ggml_backend_vk_ਬਫਰ_context *)other->ਬਫਰ->context;
                vk_ਬਫਰ o_buf = o_buf_ctx->dev_ਬਫਰ;
                ਜੇ (a_buf == o_buf) {
                    ਸਵੈਚਾਲਿਤ o_ਅਧਾਰ = vk_tensor_ਵਿੱਥ(other) + other->view_offs;
                    ਸਵੈਚਾਲਿਤ o_ਮਾਪ = ggml_nbytes(other);

                    ਜੇ ((o_ਅਧਾਰ <= n_ਅਧਾਰ && n_ਅਧਾਰ < o_ਅਧਾਰ + o_ਮਾਪ) ||
                        (n_ਅਧਾਰ <= o_ਅਧਾਰ && o_ਅਧਾਰ < n_ਅਧਾਰ + n_ਮਾਪ)) {
                        ਵਾਪਸ ਸੱਚ;
                    }
                }
            }
            ਵਾਪਸ ਝੂਠ;
        };

        // For all fused ops, check if the destination node or any of the source
        // nodes require synchronization.
        ਲਈ (int32_t i = 0; i < ctx->ਅੰਕ_additional_fused_ops + 1 && !need_sync; ++i) {
            ਅਚਲ ggml_tensor *cur_ਗੰਢ = cgraph->nodes[ਗੰਢ_idx + i];
            // If the node actually writes to memory, then check if it needs to sync
            ਜੇ (ctx->fused_ops_write_ਨਕਾਬ & (1 << i)) {
                ਜੇ (overlaps_unsynced(cur_ਗੰਢ, ctx->unsynced_nodes_read) || overlaps_unsynced(cur_ਗੰਢ, ctx->unsynced_nodes_written)) {
                    need_sync = ਸੱਚ;
                    ਰੋਕੋ;
                }
            }
            ਲਈ (uint32_t j = 0; j < GGML_MAX_SRC; ++j) {
                ਜੇ (!cur_ਗੰਢ->src[j]) {
                    ਜਾਰੀ;
                }
                ਜੇ (overlaps_unsynced(cur_ਗੰਢ->src[j], ctx->unsynced_nodes_written)) {
                    need_sync = ਸੱਚ;
                    ਰੋਕੋ;
                }
            }
        }

        ਜੇ (need_sync) {
            ਜੇ (vk_enable_sync_logger) {
                std::cerr <<  "sync" << std::ਸਤਰ_ਅੰਤ;
            }
            ctx->unsynced_nodes_written.clear();
            ctx->unsynced_nodes_read.clear();
            ggml_vk_sync_buffers(ctx, compute_ctx);

            ਜੇ (vk_perf_logger_enabled && vk_perf_logger_concurrent) {
                ctx->query_ਗੰਢ_idx[ctx->query_idx] = ਗੰਢ_idx;
                compute_ctx->s->ਬਫਰ->buf.writeTimestamp(vk::Pipeline_Stage_Flag_ਬਿੱਟ::eAllCommands, ctx->query_pool, ctx->query_idx++);
                ggml_vk_sync_buffers(ctx, compute_ctx);
            }
        }
        // Add all fused nodes to the unsynchronized lists.
        ਲਈ (int32_t i = 0; i < ctx->ਅੰਕ_additional_fused_ops + 1; ++i) {
            ਅਚਲ ggml_tensor *cur_ਗੰਢ = cgraph->nodes[ਗੰਢ_idx + i];
            // Multiple outputs could be written, e.g. in topk_moe. Add them all to the list.
            ਜੇ (ctx->fused_ops_write_ਨਕਾਬ & (1 << i)) {
                ctx->unsynced_nodes_written.push_back(cur_ਗੰਢ);
            }
            ਲਈ (uint32_t j = 0; j < GGML_MAX_SRC; ++j) {
                ਜੇ (!cur_ਗੰਢ->src[j]) {
                    ਜਾਰੀ;
                }
                ctx->unsynced_nodes_read.push_back(cur_ਗੰਢ->src[j]);
            }
        }
    }
    ਜੇ (vk_enable_sync_logger) {
        ਲਈ (ਪੂਰਨ i = 0; i < ctx->ਅੰਕ_additional_fused_ops + 1; ++i) {
            ਸਵੈਚਾਲਿਤ *n = cgraph->nodes[ਗੰਢ_idx + i];
            std::cerr << ਗੰਢ_idx + i << " " << ggml_op_name(n->op) << " " <<  n->name;
            ਜੇ (n->op == GGML_OP_GLU) {
                std::cerr << " " << ggml_glu_op_name(ggml_get_glu_op(n)) << " " << (n->src[1] ? "split" : "single") << " ";
            }
            ਜੇ (n->op == GGML_OP_ROPE) {
                ਅਚਲ ਪੂਰਨ mode = ((ਅਚਲ int32_t *) n->op_params)[2];
                std::cerr << " rope mode: " << mode;
            }
            std::cerr << std::ਸਤਰ_ਅੰਤ;
        }
    }

    // closed explicitly below, and by the destructor on the early returns
    ggml_vk_debug_label dbg(compute_ctx, cgraph, ਗੰਢ_idx, ctx->ਅੰਕ_additional_fused_ops);

    ਚੋਣ (ਗੰਢ->op) {
    ਹਾਲਤ GGML_OP_REPEAT:
        ggml_vk_repeat(ctx, compute_ctx, src0, ਗੰਢ);

        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_REPEAT_ਪਿੱਠ:
        ggml_vk_repeat_ਪਿੱਠ(ctx, compute_ctx, src0, ਗੰਢ);

        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_ACC:
    ਹਾਲਤ GGML_OP_ਸੈੱਟ:
        ggml_vk_acc(ctx, compute_ctx, src0, src1, ਗੰਢ);

        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_GET_ਪੰਕਤੀਆਂ:
        ਜੇ (ctx->fused_topk_qsa) {
            ggml_vk_topk_qsa(ctx, compute_ctx, cgraph, ਗੰਢ_idx);
        } ਨਹੀਂਤਾਂ {
            ggml_vk_get_ਪੰਕਤੀਆਂ(ctx, compute_ctx, src0, src1, ਗੰਢ);
        }

        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_GET_ਪੰਕਤੀਆਂ_ਪਿੱਠ:
        ggml_vk_get_ਪੰਕਤੀਆਂ_ਪਿੱਠ(ctx, compute_ctx, src0, src1, ਗੰਢ);

        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_ਜੋੜੋ:
        ਜੇ (ctx->ਅੰਕ_additional_fused_ops) {
            ggml_vk_multi_ਜੋੜੋ(ctx, compute_ctx, cgraph, ਗੰਢ_idx);
        } ਨਹੀਂਤਾਂ {
            ggml_vk_ਜੋੜੋ(ctx, compute_ctx, src0, src1, ਗੰਢ);
        }
        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_ਬਾਹਰਲਾ_PROD:
        ggml_vk_ਬਾਹਰਲਾ_prod(ctx, compute_ctx, src0, src1, ਗੰਢ);
        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_SUB:
        ggml_vk_sub(ctx, compute_ctx, src0, src1, ਗੰਢ);

        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_MUL:
        ਜੇ (ctx->ਅੰਕ_additional_fused_ops) {
            ggml_vk_snake_dispatch_fused(ctx, compute_ctx, cgraph, ਗੰਢ_idx);
        } ਨਹੀਂਤਾਂ {
            ggml_vk_mul(ctx, compute_ctx, src0, src1, ਗੰਢ);
        }

        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_DIV:
        ggml_vk_div(ctx, compute_ctx, src0, src1, ਗੰਢ);

        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_ਜੋੜੋ_ID:
        ggml_vk_ਜੋੜੋ_id(ctx, compute_ctx, src0, src1, src2, ਗੰਢ);

        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_CONCAT:
        ggml_vk_concat(ctx, compute_ctx, src0, src1, ਗੰਢ);

        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_UPSCALE:
        ggml_vk_upscale(ctx, compute_ctx, src0, ਗੰਢ);

        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_ADD1:
        ggml_vk_add1(ctx, compute_ctx, src0, src1, ਗੰਢ);

        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_ARANGE:
        ggml_vk_arange(ctx, compute_ctx, ਗੰਢ);

        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_FILL:
        ggml_vk_fill(ctx, compute_ctx, ਗੰਢ);

        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_SCALE:
        ਜੇ (ctx->fused_hc_post_gate) {
            ggml_tensor * hc_post = cgraph->nodes[ਗੰਢ_idx + ctx->ਅੰਕ_additional_fused_ops];
            ggml_vk_dsv4_hc_post(ctx, compute_ctx, hc_post->src[0], hc_post->src[1], hc_post->src[2], hc_post->src[3], hc_post, ਗੰਢ);
        } ਨਹੀਂਤਾਂ {
            ggml_vk_scale(ctx, compute_ctx, src0, ਗੰਢ);
        }

        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_SQR:
        ggml_vk_sqr(ctx, compute_ctx, src0, ਗੰਢ);

        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_SQRT:
        ggml_vk_sqrt(ctx, compute_ctx, src0, ਗੰਢ);

        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_SIN:
        ggml_vk_sin(ctx, compute_ctx, src0, ਗੰਢ);

        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_COS:
        ggml_vk_cos(ctx, compute_ctx, src0, ਗੰਢ);

        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_LOG:
        ggml_vk_log(ctx, compute_ctx, src0, ਗੰਢ);

        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_TRI:
        ggml_vk_tri(ctx, compute_ctx, src0, ਗੰਢ);

        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_DIAG:
        ggml_vk_diag(ctx, compute_ctx, src0, ਗੰਢ);

        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_CLAMP:
        ggml_vk_clamp(ctx, compute_ctx, src0, ਗੰਢ);

        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_PAD:
        ggml_vk_pad(ctx, compute_ctx, src0, ਗੰਢ);

        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_PAD_REFLECT_1D:
        ggml_vk_pad_reflect_1d(ctx, compute_ctx, src0, ਗੰਢ);

        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_ROLL:
        ggml_vk_roll(ctx, compute_ctx, src0, ਗੰਢ);

        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_CPY:
    ਹਾਲਤ GGML_OP_CONT:
    ਹਾਲਤ GGML_OP_DUP:
        ggml_vk_cpy(ctx, compute_ctx, src0, ਗੰਢ);

        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_ਸੈੱਟ_ਪੰਕਤੀਆਂ:
        ggml_vk_ਸੈੱਟ_ਪੰਕਤੀਆਂ(ctx, compute_ctx, src0, src1, ਗੰਢ);

        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_SILU_ਪਿੱਠ:
        ggml_vk_silu_ਪਿੱਠ(ctx, compute_ctx, src0, src1, ਗੰਢ);

        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_NORM:
        ggml_vk_norm(ctx, compute_ctx, src0, ਗੰਢ);

        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_ਸਮੂਹ_NORM:
        ggml_vk_ਸਮੂਹ_norm(ctx, compute_ctx, src0, ਗੰਢ);

        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_RMS_NORM:
        ggml_vk_rms_norm(ctx, compute_ctx, cgraph, ਗੰਢ_idx, (ਦਸ਼ਮਲਵ *)ਗੰਢ->op_params);
        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_RMS_NORM_ਪਿੱਠ:
        ggml_vk_rms_norm_ਪਿੱਠ(ctx, compute_ctx, src0, src1, ਗੰਢ);

        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_L2_NORM:
        ggml_vk_l2_norm(ctx, compute_ctx, src0, ਗੰਢ);

        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_UNARY:
        ਜੇ (ctx->fused_topk_moe_ਬਹੁਲਕ != TOPK_MOE_ਗਿਣਤੀ) {
            ggml_vk_topk_moe(ctx, compute_ctx, cgraph, ਗੰਢ_idx);
            ਰੋਕੋ;
        }
        ਜੇ (ctx->ਅੰਕ_additional_fused_ops) {
            ggml_vk_unary_mul(ctx, compute_ctx, cgraph, ਗੰਢ_idx);
            ਰੋਕੋ;
        }

        ਚੋਣ (ggml_get_unary_op(ਗੰਢ)) {
        ਹਾਲਤ GGML_UNARY_OP_ELU:
        ਹਾਲਤ GGML_UNARY_OP_EXP:
        ਹਾਲਤ GGML_UNARY_OP_EXPM1:
        ਹਾਲਤ GGML_UNARY_OP_SILU:
        ਹਾਲਤ GGML_UNARY_OP_GELU:
        ਹਾਲਤ GGML_UNARY_OP_GELU_ERF:
        ਹਾਲਤ GGML_UNARY_OP_GELU_ਤੇਜ਼:
        ਹਾਲਤ GGML_UNARY_OP_RELU:
        ਹਾਲਤ GGML_UNARY_OP_NEG:
        ਹਾਲਤ GGML_UNARY_OP_TANH:
        ਹਾਲਤ GGML_UNARY_OP_SIGMOID:
        ਹਾਲਤ GGML_UNARY_OP_HARDSIGMOID:
        ਹਾਲਤ GGML_UNARY_OP_HARDSWISH:
        ਹਾਲਤ GGML_UNARY_OP_ABS:
        ਹਾਲਤ GGML_UNARY_OP_SOFTPLUS:
        ਹਾਲਤ GGML_UNARY_OP_ਕਦਮ:
        ਹਾਲਤ GGML_UNARY_OP_ਗੇੜਾ:
        ਹਾਲਤ GGML_UNARY_OP_CEIL:
        ਹਾਲਤ GGML_UNARY_OP_FLOOR:
        ਹਾਲਤ GGML_UNARY_OP_TRUNC:
        ਹਾਲਤ GGML_UNARY_OP_SGN:
            ggml_vk_unary(ctx, compute_ctx, src0, ਗੰਢ);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_UNARY_OP_XIELU:
            ggml_vk_xielu(ctx, compute_ctx, src0, ਗੰਢ);
            ਰੋਕੋ;
        ਮੂਲ:
            ਵਾਪਸ ਝੂਠ;
        }
        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_GLU:
        ਚੋਣ (ggml_get_glu_op(ਗੰਢ)) {
        ਹਾਲਤ GGML_GLU_OP_GEGLU:
        ਹਾਲਤ GGML_GLU_OP_REGLU:
        ਹਾਲਤ GGML_GLU_OP_SWIGLU:
        ਹਾਲਤ GGML_GLU_OP_SWIGLU_OAI:
        ਹਾਲਤ GGML_GLU_OP_GEGLU_ERF:
        ਹਾਲਤ GGML_GLU_OP_GEGLU_ਤੇਜ਼:
        ਹਾਲਤ GGML_GLU_OP_SWIGLU_CLAMP:
            ggml_vk_glu(ctx, compute_ctx, src0, src1, ਗੰਢ);
            ਰੋਕੋ;
        ਮੂਲ:
            ਵਾਪਸ ਝੂਠ;
        }
        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_DIAG_ਨਕਾਬ_INF:
        ggml_vk_diag_ਨਕਾਬ_inf(ctx, compute_ctx, src0, ਗੰਢ);

        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_SOFT_MAX:
        ਜੇ (ctx->fused_topk_moe_ਬਹੁਲਕ != TOPK_MOE_ਗਿਣਤੀ) {
            ggml_vk_topk_moe(ctx, compute_ctx, cgraph, ਗੰਢ_idx);
        } ਨਹੀਂਤਾਂ {
            ggml_vk_soft_max(ctx, compute_ctx, src0, src1, src2, ਗੰਢ);
        }

        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_SOFT_MAX_ਪਿੱਠ:
        ggml_vk_soft_max_ਪਿੱਠ(ctx, compute_ctx, src0, src1, ਗੰਢ);

        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_ROPE:
        ggml_vk_rope(ctx, compute_ctx, cgraph, ਗੰਢ_idx, ਝੂਠ);

        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_ROPE_ਪਿੱਠ:
        ggml_vk_rope(ctx, compute_ctx, cgraph, ਗੰਢ_idx, ਸੱਚ);

        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_ARGSORT:
        ਜੇ (ctx->fused_topk_moe_ਬਹੁਲਕ != TOPK_MOE_ਗਿਣਤੀ) {
            ggml_vk_topk_moe(ctx, compute_ctx, cgraph, ਗੰਢ_idx);
        } ਨਹੀਂਤਾਂ {
            ggml_vk_argsort(ctx, compute_ctx, src0, ਗੰਢ);
        }

        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_ਸਿਖਰ_K:
        ggml_vk_topk(ctx, compute_ctx, src0, ਗੰਢ);

        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_ਜੋੜ:
        ggml_vk_ਜੋੜ(ctx, compute_ctx, src0, ਗੰਢ);

        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_ਜੋੜ_ਪੰਕਤੀਆਂ:
        ggml_vk_ਜੋੜ_ਪੰਕਤੀਆਂ(ctx, compute_ctx, src0, ਗੰਢ);

        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_CUMSUM:
        ggml_vk_cumsum(ctx, compute_ctx, src0, ਗੰਢ);

        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_DSV4_HC_COMB:
        ggml_vk_dsv4_hc_comb(ctx, compute_ctx, src0, src1, src2, ਗੰਢ);

        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_DSV4_HC_PRE:
        ggml_vk_dsv4_hc_pre(ctx, compute_ctx, src0, src1, ਗੰਢ);

        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_DSV4_HC_POST:
        ggml_vk_dsv4_hc_post(ctx, compute_ctx, src0, src1, src2, src3, ਗੰਢ);

        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_ਔਸਤ:
        ggml_vk_ਔਸਤ(ctx, compute_ctx, src0, ਗੰਢ);

        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_ARGMAX:
        ggml_vk_argmax(ctx, compute_ctx, src0, ਗੰਢ);

        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_CROSS_ENTROPY_LOSS:
        ggml_vk_cross_entropy_loss(ctx, compute_ctx, ਗੰਢ);

        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_CROSS_ENTROPY_LOSS_ਪਿੱਠ:
        ggml_vk_cross_entropy_loss_ਪਿੱਠ(ctx, compute_ctx, ਗੰਢ);

        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_ਗਿਣਤੀ_EQUAL:
        ggml_vk_ਗਿਣਤੀ_equal(ctx, compute_ctx, src0, src1, ਗੰਢ);

        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_SOLVE_TRI:
        ggml_vk_solve_tri(ctx, compute_ctx, src0, src1, ਗੰਢ);

        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_IM2_ਖਾਨਾ:
        ggml_vk_im2col(ctx, compute_ctx, src0, src1, ਗੰਢ);

        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_IM2_ਖਾਨਾ_3_D:
        ggml_vk_im2col_3d(ctx, compute_ctx, src0, src1, ਗੰਢ);

        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_TIMESTEP_EMBEDDING:
        ggml_vk_timestep_embedding(ctx, compute_ctx, src0, ਗੰਢ);

        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_COL2IM_1D:
        ggml_vk_col2im_1d(ctx, compute_ctx, src0, ਗੰਢ);

        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_CONV_TRANSPOSE_1D:
        ggml_vk_conv_transpose_1d(ctx, compute_ctx, src0, src1, ਗੰਢ);

        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_POOL_1D:
        ggml_vk_pool_1d(ctx, compute_ctx, src0, ਗੰਢ);

        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_POOL_2D:
        ggml_vk_pool_2d(ctx, compute_ctx, src0, ਗੰਢ);

        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_CONV_2D:
    ਹਾਲਤ GGML_OP_CONV_TRANSPOSE_2D:
        ggml_vk_conv_2d(ctx, compute_ctx, src0, src1, ਗੰਢ);

        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_CONV_3D:
        ggml_vk_conv_3d(ctx, compute_ctx, src0, src1, ਗੰਢ);

        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_CONV_2D_DW:
        ggml_vk_conv_2d_dw(ctx, compute_ctx, src0, src1, ਗੰਢ);

        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_LEAKY_RELU:
        ggml_vk_leaky_relu(ctx, compute_ctx, src0, ਗੰਢ);

        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_MUL_MAT:
        ggml_vk_mul_mat(ctx, compute_ctx, cgraph, ਗੰਢ_idx);

        ਰੋਕੋ;
    ਹਾਲਤ GGML_OP_MUL_MAT_ID:
        ggml_vk_mul_mat_id(ctx, compute_ctx, cgraph, ਗੰਢ_idx);

        ਰੋਕੋ;

    ਹਾਲਤ GGML_OP_FLASH_ATTN_EXT:
        ggml_vk_flash_attn(ctx, compute_ctx, src0, src1, src2, src3, ਗੰਢ->src[4], ਗੰਢ);

        ਰੋਕੋ;

    ਹਾਲਤ GGML_OP_RWKV_WKV6:
        ggml_vk_rwkv_wkv6(ctx, compute_ctx, ਗੰਢ);

        ਰੋਕੋ;

    ਹਾਲਤ GGML_OP_RWKV_WKV7:
        ggml_vk_rwkv_wkv7(ctx, compute_ctx, ਗੰਢ);

        ਰੋਕੋ;

    ਹਾਲਤ GGML_OP_GATED_ਰੇਖਿਕ_ATTN:
        ggml_vk_gated_ਰੇਖਿਕ_attn(ctx, compute_ctx, ਗੰਢ);

        ਰੋਕੋ;

    ਹਾਲਤ GGML_OP_LIGHTNING_INDEXER:
        ggml_vk_lightning_indexer(ctx, compute_ctx, ਗੰਢ);

        ਰੋਕੋ;

    ਹਾਲਤ GGML_OP_GATED_DELTA_NET:
        ggml_vk_gated_delta_net(ctx, compute_ctx, ਗੰਢ);

        ਰੋਕੋ;

    ਹਾਲਤ GGML_OP_SSM_ਜਾਂਚ:
        ggml_vk_ssm_ਜਾਂਚ(ctx, compute_ctx, ਗੰਢ);

        ਰੋਕੋ;

    ਹਾਲਤ GGML_OP_SSM_CONV:
        ggml_vk_ssm_conv(ctx, compute_ctx, cgraph, ਗੰਢ_idx);

        ਰੋਕੋ;

    ਹਾਲਤ GGML_OP_OPT_ਕਦਮ_ADAMW:
        ggml_vk_opt_ਕਦਮ_adamw(ctx, compute_ctx, ਗੰਢ);

        ਰੋਕੋ;

    ਹਾਲਤ GGML_OP_OPT_ਕਦਮ_SGD:
        ggml_vk_opt_ਕਦਮ_sgd(ctx, compute_ctx, src0, src1, src2, ਗੰਢ);

        ਰੋਕੋ;
    ਮੂਲ:
        ਵਾਪਸ ਝੂਠ;
    }

    // the submit path below can end the command buffer, so close the region first
    dbg.close();

    ctx->tensor_ctxs[ਗੰਢ_idx] = compute_ctx;

#ਜੇ defined(GGML_VULKAN_CHECK_RESULTS)
    // Force context reset on each node so that each tensor ends up in its own context
    // and can be run and compared to its CPU equivalent separately
    ਆਖਰੀ_ਗੰਢ = ਸੱਚ;
#ਜੇ_ਅੰਤ

    ਜੇ (submit || ਆਖਰੀ_ਗੰਢ) {
        ggml_vk_ctx_ਸਿਰਾ(compute_ctx);

        // TODO probably it'd be better to pass a exit_node flag to ggml_vk_compute_forward
        ਜੇ (ਆਖਰੀ_ਗੰਢ) {
            compute_ctx->exit_tensor_idx = ਗੰਢ_idx_begin;
        }
        ਨਹੀਂਤਾਂ {
            compute_ctx->exit_tensor_idx = -1;
        }

        ctx->compute_ctx.reset();

        ggml_vk_compute_forward(ctx, cgraph, ਗੰਢ_begin, ਗੰਢ_idx_begin, almost_ready);
    }
    ਵਾਪਸ ਸੱਚ;
}

ਖਾਲੀ ggml_vk_compute_forward(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, ggml_tensor * tensor, ਪੂਰਨ tensor_idx, ਤਾਰਕਿਕ almost_ready) {
    GGML_ਅਣਵਰਤਿਆ(cgraph);
    GGML_ਅਣਵਰਤਿਆ(tensor);

    VK_LOG_DEBUG("ggml_vk_compute_forward(" << tensor << ", name=" << tensor->name << ", op=" << ggml_op_name(tensor->op) << ", type=" << tensor->type << ", ne0=" << tensor->ne[0] << ", ne1=" << tensor->ne[1] << ", ne2=" << tensor->ne[2] << ", ne3=" << tensor->ne[3] << ", nb0=" << tensor->nb[0] << ", nb1=" << tensor->nb[1] << ", nb2=" << tensor->nb[2] << ", nb3=" << tensor->nb[3] << ", view_src=" << tensor->view_src << ", view_offs=" << tensor->view_offs << ")");

    vk_context subctx = ctx->tensor_ctxs[tensor_idx].ਤਾਲਾ();

    // Only run if ctx hasn't been submitted yet
    ਜੇ (!subctx->seqs.ਖਾਲੀ_ਥਾਵਾਂ()) {
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_VULKAN_CHECK_RESULTS
        ggml_vk_check_results_0(ctx, cgraph, tensor_idx);
#ਜੇ_ਅੰਤ

        // Do staging buffer copies
        ਲਈ (ਸਵੈਚਾਲਿਤ& cpy : subctx->ਵਿੱਚ_memcpys) {
            ਸਮ੍ਰਿਤੀ_ਨਕਲ(cpy.dst, cpy.src, cpy.n);
        }

        ਲਈ (ਸਵੈਚਾਲਿਤ& mset : subctx->memsets) {
            memset(mset.dst, mset.val, mset.n);
        }

        ਜੇ (ctx->device->serialize_submissions) {
            ggml_vk_submit(subctx, ctx->fence);
        } ਨਹੀਂਤਾਂ ਜੇ (almost_ready && !ctx->almost_ready_fence_pending) {
            ggml_vk_submit(subctx, ctx->almost_ready_fence);
            ctx->almost_ready_fence_pending = ਸੱਚ;
        } ਨਹੀਂਤਾਂ {
            ggml_vk_submit(subctx, {});
        }
        ctx->submit_pending = ਸੱਚ;

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_VULKAN_CHECK_RESULTS
        ggml_vk_synchronize(ctx);
        ggml_vk_check_results_1(ctx, cgraph, tensor_idx);
#ਜੇ_ਅੰਤ
    }

    ਜੇ (tensor_idx == subctx->exit_tensor_idx) {
        // Do staging buffer copies
        ਲਈ (ਸਵੈਚਾਲਿਤ& cpy : subctx->ਬਾਹਰਲਾ_memcpys) {
            ਸਮ੍ਰਿਤੀ_ਨਕਲ(cpy.dst, cpy.src, cpy.n);
        }
        subctx->ਵਿੱਚ_memcpys.clear();
        subctx->ਬਾਹਰਲਾ_memcpys.clear();
        subctx->memsets.clear();
    }
}

ਖਾਲੀ ggml_vk_ਗ੍ਰਾਫ_cleanup(ggml_backend_vk_context * ctx) {
    VK_LOG_DEBUG("ggml_vk_graph_cleanup()");
    ctx->prealloc_y_ਆਖਰੀ_pipeline_ਵਰਤਿਆ = {};
    ctx->prealloc_y_ਆਖਰੀ_tensor_ਵਰਤਿਆ = ਖਾਲੀ_ਸੂਚਕ;
    ctx->prealloc_y_ਆਖਰੀ_k_padded = ਝੂਠ;

    ctx->unsynced_nodes_written.clear();
    ctx->unsynced_nodes_read.clear();
    ctx->prealloc_x_need_sync = ctx->prealloc_y_need_sync = ctx->prealloc_ਵੰਡੋ_k_need_sync = ਝੂਠ;

    ggml_vk_command_pool_cleanup(ctx->device, ctx->compute_cmd_pool);
    ਜੇ (ctx->device->async_use_transfer_ਕਤਾਰ) {
        ggml_vk_command_pool_cleanup(ctx->device, ctx->transfer_cmd_pool);
    }

    ਲਈ (size_t i = 0; i < ctx->gc.semaphores.size(); i++) {
        ctx->device->device.destroySemaphore({ ctx->gc.semaphores[i].s });
    }
    ctx->gc.semaphores.clear();

    ਲਈ (size_t i = 0; i < ctx->gc.tl_semaphores.size(); i++) {
        ctx->device->device.destroySemaphore({ ctx->gc.tl_semaphores[i].s });
    }
    ctx->gc.tl_semaphores.clear();
    ctx->semaphore_idx = 0;

    ctx->event_idx = 0;

    ਲਈ (ਸਵੈਚਾਲਿਤ& event : ctx->gc.events) {
        ctx->device->device.resetEvent(event);
    }

    ctx->tensor_ctxs.clear();
    ctx->gc.contexts.clear();
    ctx->pipeline_descriptor_ਸੈੱਟ_requirements = 0;
    ctx->descriptor_ਸੈੱਟ_idx = 0;
}

ਖਾਲੀ ggml_vk_cleanup(ggml_backend_vk_context * ctx) {
    VK_LOG_DEBUG("ggml_vk_cleanup(" << ctx->name << ")");
    // discard any unsubmitted command buffers
    ctx->compute_ctx.reset();
    // wait for any pending command buffers to finish
    ggml_vk_synchronize(ctx);

    ggml_vk_ਗ੍ਰਾਫ_cleanup(ctx);

    ggml_vk_destroy_ਬਫਰ(ctx->prealloc_x);
    ggml_vk_destroy_ਬਫਰ(ctx->prealloc_y);
    ggml_vk_destroy_ਬਫਰ(ctx->prealloc_ਵੰਡੋ_k);
    ggml_vk_destroy_ਬਫਰ(ctx->prealloc_ਜੋੜੋ_rms_partials);
    ggml_vk_destroy_ਬਫਰ(ctx->sync_staging);

    ctx->prealloc_y_ਆਖਰੀ_pipeline_ਵਰਤਿਆ = ਖਾਲੀ_ਸੂਚਕ;
    ctx->prealloc_y_ਆਖਰੀ_tensor_ਵਰਤਿਆ = ਖਾਲੀ_ਸੂਚਕ;
    ctx->prealloc_y_ਆਖਰੀ_k_padded = ਝੂਠ;

    ctx->prealloc_ਮਾਪ_x = 0;
    ctx->prealloc_ਮਾਪ_y = 0;
    ctx->prealloc_ਮਾਪ_ਵੰਡੋ_k = 0;

    ਲਈ (ਸਵੈਚਾਲਿਤ& event : ctx->gc.events) {
        ctx->device->device.destroyEvent(event);
    }
    ctx->gc.events.clear();

    ctx->device->device.destroyFence(ctx->fence);
    ctx->device->device.destroyFence(ctx->almost_ready_fence);

    ਲਈ (ਸਵੈਚਾਲਿਤ& pool : ctx->descriptor_pools) {
        ctx->device->device.destroyDescriptorPool(pool);
    }
    ctx->descriptor_pools.clear();
    ctx->descriptor_sets.clear();
    ctx->descriptor_ਸੈੱਟ_bindings.clear();

    ctx->compute_cmd_pool.destroy(ctx->device->device);
    ਜੇ (ctx->device->async_use_transfer_ਕਤਾਰ) {
        ctx->device->device.destroySemaphore(ctx->transfer_semaphore.s);

        ctx->transfer_cmd_pool.destroy(ctx->device->device);
    }
    ਜੇ (vk_perf_logger_enabled) {
        ctx->perf_logger->print_timings(ਸੱਚ);
    }
}

ਪੂਰਨ ggml_vk_get_device_ਗਿਣਤੀ() {
    ggml_vk_instance_init();

    ਵਾਪਸ vk_instance.device_indices.size();
}

ਖਾਲੀ ggml_vk_get_device_description(ਪੂਰਨ device, ਅੱਖਰ * description, size_t description_ਮਾਪ) {
    ggml_vk_instance_init();

    std::vector<vk::PhysicalDevice> devices = vk_instance.instance.enumeratePhysicalDevices();

    vk::PhysicalDeviceProperties props;
    devices[device].getProperties(&props);

    snprintf(description, description_ਮਾਪ, "%s", props.deviceName.ਅੰਕੜੇ());
}

ਤਾਰਕਿਕ ggml_backend_ਬਫਰ_is_vk(ggml_backend_ਬਫਰ_t ਬਫਰ) {
    ਵਾਪਸ ਬਫਰ->buft->iface.get_name == ggml_backend_vk_ਬਫਰ_type_name;
}

ਖਾਲੀ ggml_backend_vk_ਬਫਰ_free_ਬਫਰ(ggml_backend_ਬਫਰ_t ਬਫਰ) {
    VK_LOG_ਸਮ੍ਰਿਤੀ("ggml_backend_vk_buffer_free_buffer()");
    ggml_backend_vk_ਬਫਰ_context * ctx = (ggml_backend_vk_ਬਫਰ_context *)ਬਫਰ->context;
    ggml_vk_destroy_ਬਫਰ(ctx->dev_ਬਫਰ);
    ਮਿਟਾਓ ctx;
}

ਖਾਲੀ * ggml_backend_vk_ਬਫਰ_get_ਅਧਾਰ(ggml_backend_ਬਫਰ_t ਬਫਰ) {
    ਵਾਪਸ vk_ptr_ਅਧਾਰ;

    ਅਣਵਰਤਿਆ(ਬਫਰ);
}

ਗਿਣਤੀ ggml_ਸਥਿਤੀ ggml_backend_vk_ਬਫਰ_init_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ, ggml_tensor * tensor) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_init_tensor(" << ਬਫਰ << " (" << ਬਫਰ->context << "), " << tensor << ")");
    ਜੇ (tensor->view_src != ਖਾਲੀ_ਸੂਚਕ) {
        GGML_ASSERT(tensor->view_src->ਬਫਰ->buft == ਬਫਰ->buft);
    }
    ਵਾਪਸ GGML_ਸਥਿਤੀ_SUCCESS;
}

ਖਾਲੀ ggml_backend_vk_ਬਫਰ_memset_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ, ggml_tensor * tensor, uint8_t value, size_t ਵਿੱਥ, size_t size) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_memset_tensor(" << ਬਫਰ << ", " << tensor << ", " << value << ", " << ਵਿੱਥ << ", " << size << ")");
    ggml_backend_vk_ਬਫਰ_context * buf_ctx = (ggml_backend_vk_ਬਫਰ_context *)ਬਫਰ->context;
    vk_ਬਫਰ buf = buf_ctx->dev_ਬਫਰ;

    ਜੇ (size == 0) {
        ਵਾਪਸ;
    }

    uint32_t val32 = (uint32_t)value * 0x01010101;
    ggml_vk_ਬਫਰ_memset(buf, vk_tensor_ਵਿੱਥ(tensor) + tensor->view_offs + ਵਿੱਥ, val32, size);
}

ਖਾਲੀ ggml_backend_vk_ਬਫਰ_ਸੈੱਟ_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ, ggml_tensor * tensor, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ, size_t size) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_set_tensor(" << ਬਫਰ << ", " << tensor << ", " << ਅੰਕੜੇ << ", " << ਵਿੱਥ << ", " << size << ")");
    ggml_backend_vk_ਬਫਰ_context * buf_ctx = (ggml_backend_vk_ਬਫਰ_context *)ਬਫਰ->context;
    vk_ਬਫਰ buf = buf_ctx->dev_ਬਫਰ;

    ਜੇ (size == 0) {
        ਵਾਪਸ;
    }

    ggml_vk_ਬਫਰ_write(buf, vk_tensor_ਵਿੱਥ(tensor) + tensor->view_offs + ਵਿੱਥ, ਅੰਕੜੇ, size);
}

ਖਾਲੀ ggml_backend_vk_ਬਫਰ_ਸੈੱਟ_tensor_2d(ggml_backend_ਬਫਰ_t ਬਫਰ, ggml_tensor * tensor, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ,
                                                 size_t size, size_t n_copies, size_t stride_tensor, size_t stride_ਅੰਕੜੇ) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_set_tensor_2d(" << ਬਫਰ << ", " << tensor << ", " << ਅੰਕੜੇ << ", " << ਵਿੱਥ << ", " << size << ", " <<
                 n_copies << ", " << stride_tensor << ", " << stride_ਅੰਕੜੇ << ")");
    ggml_backend_vk_ਬਫਰ_context * buf_ctx = (ggml_backend_vk_ਬਫਰ_context *)ਬਫਰ->context;
    vk_ਬਫਰ buf = buf_ctx->dev_ਬਫਰ;

    ਜੇ (size == 0) {
        ਵਾਪਸ;
    }

    ggml_vk_ਬਫਰ_write_2d(buf, vk_tensor_ਵਿੱਥ(tensor) + tensor->view_offs + ਵਿੱਥ, ਅੰਕੜੇ, stride_ਅੰਕੜੇ, stride_tensor, size, n_copies);
}

ਖਾਲੀ ggml_backend_vk_ਬਫਰ_get_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ, ਅਚਲ ggml_tensor * tensor, ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ, size_t size) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_get_tensor(" << ਬਫਰ << ", " << tensor << ", " << ਅੰਕੜੇ << ", " << ਵਿੱਥ << ", " << size << ")");
    ggml_backend_vk_ਬਫਰ_context * buf_ctx = (ggml_backend_vk_ਬਫਰ_context *)ਬਫਰ->context;

    ਜੇ (size == 0) {
        ਵਾਪਸ;
    }

    vk_ਬਫਰ buf = buf_ctx->dev_ਬਫਰ;

    ggml_vk_ਬਫਰ_read(buf, vk_tensor_ਵਿੱਥ(tensor) + tensor->view_offs + ਵਿੱਥ, ਅੰਕੜੇ, size);
}

ਖਾਲੀ ggml_backend_vk_ਬਫਰ_get_tensor_2d(ggml_backend_ਬਫਰ_t ਬਫਰ, ਅਚਲ ggml_tensor * tensor, ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ,
                                                 size_t size, size_t n_copies, size_t stride_tensor, size_t stride_ਅੰਕੜੇ) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_get_tensor_2d(" << ਬਫਰ << ", " << tensor << ", " << ਅੰਕੜੇ << ", " << ਵਿੱਥ << ", " << size << ", " <<
                 n_copies << ", " << stride_tensor << ", " << stride_ਅੰਕੜੇ << ")");
    ggml_backend_vk_ਬਫਰ_context * buf_ctx = (ggml_backend_vk_ਬਫਰ_context *)ਬਫਰ->context;

    ਜੇ (size == 0) {
        ਵਾਪਸ;
    }

    vk_ਬਫਰ buf = buf_ctx->dev_ਬਫਰ;

    ggml_vk_ਬਫਰ_read_2d(buf, vk_tensor_ਵਿੱਥ(tensor) + tensor->view_offs + ਵਿੱਥ, ਅੰਕੜੇ, stride_tensor, stride_ਅੰਕੜੇ, size, n_copies);
}

ਤਾਰਕਿਕ ggml_backend_vk_ਬਫਰ_cpy_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ, ਅਚਲ ggml_tensor * src, ggml_tensor * dst) {
    ਜੇ (ggml_nbytes(src) == 0) {
        ਵਾਪਸ ਸੱਚ;
    }

    ਜੇ (ggml_backend_ਬਫਰ_is_vk(src->ਬਫਰ)) {
        ggml_backend_vk_ਬਫਰ_context * src_buf_ctx = (ggml_backend_vk_ਬਫਰ_context *)src->ਬਫਰ->context;
        ggml_backend_vk_ਬਫਰ_context * dst_buf_ctx = (ggml_backend_vk_ਬਫਰ_context *)dst->ਬਫਰ->context;

        vk_ਬਫਰ src_buf = src_buf_ctx->dev_ਬਫਰ;
        vk_ਬਫਰ dst_buf = dst_buf_ctx->dev_ਬਫਰ;

        ggml_vk_ਬਫਰ_copy(dst_buf, vk_tensor_ਵਿੱਥ(dst) + dst->view_offs, src_buf, vk_tensor_ਵਿੱਥ(src) + src->view_offs, ggml_nbytes(src));

        ਵਾਪਸ ਸੱਚ;
    }
    ਵਾਪਸ ਝੂਠ;

    ਅਣਵਰਤਿਆ(ਬਫਰ);
}

ਖਾਲੀ ggml_backend_vk_ਬਫਰ_clear(ggml_backend_ਬਫਰ_t ਬਫਰ, uint8_t value) {
    ggml_backend_vk_ਬਫਰ_context * ctx = (ggml_backend_vk_ਬਫਰ_context *)ਬਫਰ->context;

    ggml_vk_ਬਫਰ_memset(ctx->dev_ਬਫਰ, 0, value, ਬਫਰ->size);
}

ਅਚਲ ਅੱਖਰ * ggml_backend_vk_ਬਫਰ_type_name(ggml_backend_ਬਫਰ_type_t buft) {
    ggml_backend_vk_ਬਫਰ_type_context * ctx = (ggml_backend_vk_ਬਫਰ_type_context *)buft->context;

    ਵਾਪਸ ctx->name.c_str();
}

ggml_backend_ਬਫਰ_t ggml_backend_vk_ਬਫਰ_type_alloc_ਬਫਰ(ggml_backend_ਬਫਰ_type_t buft, size_t size) {
    VK_LOG_ਸਮ੍ਰਿਤੀ("ggml_backend_vk_buffer_type_alloc_buffer(" << size << ")");
    ggml_backend_vk_ਬਫਰ_type_context * ctx = (ggml_backend_vk_ਬਫਰ_type_context *) buft->context;

    vk_ਬਫਰ dev_ਬਫਰ = ਖਾਲੀ_ਸੂਚਕ;
    ਕੋਸ਼ਿਸ਼ {
        dev_ਬਫਰ = ggml_vk_create_ਬਫਰ_device(ctx->device, size);
    } ਫੜੋ (ਅਚਲ vk::SystemError& e) {
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }

    ggml_backend_vk_ਬਫਰ_context * bufctx = ਨਵਾਂ ggml_backend_vk_ਬਫਰ_context(ctx->device, std::move(dev_ਬਫਰ), ctx->name);

    ਵਾਪਸ ggml_backend_ਬਫਰ_init(buft, ggml_backend_vk_ਬਫਰ_interface, bufctx, size);
}

size_t ggml_backend_vk_ਬਫਰ_type_get_alignment(ggml_backend_ਬਫਰ_type_t buft) {
    ggml_backend_vk_ਬਫਰ_type_context * ctx = (ggml_backend_vk_ਬਫਰ_type_context *) buft->context;
    ਵਾਪਸ ctx->device->properties.limits.min_Storage_ਬਫਰ_ਵਿੱਥ_Alignment;
}

size_t ggml_backend_vk_ਬਫਰ_type_get_max_ਮਾਪ(ggml_backend_ਬਫਰ_type_t buft) {
    ggml_backend_vk_ਬਫਰ_type_context * ctx = (ggml_backend_vk_ਬਫਰ_type_context *) buft->context;
    ਵਾਪਸ ctx->device->suballocation_ਖੰਡ_ਮਾਪ;
}

size_t ggml_backend_vk_ਬਫਰ_type_get_alloc_ਮਾਪ(ggml_backend_ਬਫਰ_type_t buft, ਅਚਲ ggml_tensor * tensor) {
    ਵਾਪਸ ggml_nbytes(tensor);

    ਅਣਵਰਤਿਆ(buft);
}

ggml_backend_ਬਫਰ_type_t ggml_backend_vk_ਬਫਰ_type(size_t dev_ਅੰਕ) {
    ggml_vk_instance_init();

    VK_LOG_DEBUG("ggml_backend_vk_buffer_type(" << dev_ਅੰਕ << ")");

    vk_device dev = ggml_vk_get_device(dev_ਅੰਕ);

    ਵਾਪਸ &dev->ਬਫਰ_type;
}

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_vk_host_ਬਫਰ_type_name(ggml_backend_ਬਫਰ_type_t buft) {
    ਵਾਪਸ GGML_VK_NAME "_Host";

    ਅਣਵਰਤਿਆ(buft);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_vk_host_ਬਫਰ_free_ਬਫਰ(ggml_backend_ਬਫਰ_t ਬਫਰ) {
    VK_LOG_ਸਮ੍ਰਿਤੀ("ggml_backend_vk_host_buffer_free_buffer()");
    ggml_vk_host_free(ਬਫਰ->context);
}

ਸਥਿਰ ggml_backend_ਬਫਰ_t ggml_backend_vk_host_ਬਫਰ_type_alloc_ਬਫਰ(ggml_backend_ਬਫਰ_type_t buft, size_t size) {
    VK_LOG_ਸਮ੍ਰਿਤੀ("ggml_backend_vk_host_buffer_type_alloc_buffer(" << size << ")");

    size += 32;  // Behave like the CPU buffer type
    ਖਾਲੀ * ptr = ggml_vk_host_malloc(size);
    ਜੇ (ptr == ਖਾਲੀ_ਸੂਚਕ) {
        GGML_LOG_WARN("ggml_vulkan: Failed to allocate pinned memory\n");
        // fallback to cpu buffer
        ਵਾਪਸ ggml_backend_buft_alloc_ਬਫਰ(ggml_backend_cpu_ਬਫਰ_type(), size);
    }

    ggml_backend_ਬਫਰ_t ਬਫਰ = ggml_backend_cpu_ਬਫਰ_from_ptr(ptr, size);
    ਬਫਰ->buft = buft;
    ਬਫਰ->iface.free_ਬਫਰ = ggml_backend_vk_host_ਬਫਰ_free_ਬਫਰ;

    ਵਾਪਸ ਬਫਰ;

    ਅਣਵਰਤਿਆ(buft);
}

ਸਥਿਰ size_t ggml_backend_vk_host_ਬਫਰ_type_get_alignment(ggml_backend_ਬਫਰ_type_t buft) {
    ਵਾਪਸ vk_instance.devices[0]->properties.limits.min_ਸਮ੍ਰਿਤੀ_Map_Alignment;

    ਅਣਵਰਤਿਆ(buft);
}

ਸਥਿਰ size_t ggml_backend_vk_host_ਬਫਰ_type_get_max_ਮਾਪ(ggml_backend_ਬਫਰ_type_t buft) {
    ਵਾਪਸ vk_instance.devices[0]->suballocation_ਖੰਡ_ਮਾਪ;

    ਅਣਵਰਤਿਆ(buft);
}

ggml_backend_ਬਫਰ_type_t ggml_backend_vk_host_ਬਫਰ_type() {
    ਸਥਿਰ ਬਣਤਰ ggml_backend_ਬਫਰ_type ggml_backend_vk_ਬਫਰ_type_host = {
        /* .iface    = */ {
            /* .get_name            = */ ggml_backend_vk_host_ਬਫਰ_type_name,
            /* .alloc_buffer        = */ ggml_backend_vk_host_ਬਫਰ_type_alloc_ਬਫਰ,
            /* .alloc_buffer_n      = */ ਖਾਲੀ_ਸੂਚਕ,
            /* .get_alignment       = */ ggml_backend_vk_host_ਬਫਰ_type_get_alignment,
            /* .get_max_size        = */ ggml_backend_vk_host_ਬਫਰ_type_get_max_ਮਾਪ,
            /* .get_alloc_size      = */ ggml_backend_cpu_ਬਫਰ_type()->iface.get_alloc_ਮਾਪ,
            /* .get_alloc_size_n    = */ ਨਿਰਾ,
            /* .is_host             = */ ggml_backend_cpu_ਬਫਰ_type()->iface.is_host,
        },
        /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_vk_reg(), 0),
        /* .context  = */ ਖਾਲੀ_ਸੂਚਕ,
    };

    // Make sure device 0 is initialized
    ggml_vk_instance_init();
    ggml_vk_get_device(0);

    ਵਾਪਸ &ggml_backend_vk_ਬਫਰ_type_host;
}

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_vk_name(ggml_backend_t backend) {
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;

    ਵਾਪਸ ctx->name.c_str();
}

ਖਾਲੀ ggml_backend_vk_free(ggml_backend_t backend) {
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    VK_LOG_DEBUG("ggml_backend_vk_free(" << ctx->name << ")");

    ggml_vk_cleanup(ctx);

    ਮਿਟਾਓ ctx;
    ਮਿਟਾਓ backend;
}

ਸਥਿਰ ggml_backend_ਬਫਰ_type_t ggml_backend_vk_get_default_ਬਫਰ_type(ggml_backend_t backend) {
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;

    ਵਾਪਸ &ctx->device->ਬਫਰ_type;
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_vk_ਸੈੱਟ_tensor_2d_async(ggml_backend_t backend, ggml_tensor * tensor, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ,
                                                size_t size, size_t n_copies, size_t stride_tensor, size_t stride_ਅੰਕੜੇ) {
    VK_LOG_DEBUG("ggml_backend_vk_set_tensor_2d_async(" << size << ", " << n_copies << ")");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    GGML_ASSERT((tensor->ਬਫਰ->buft == ggml_backend_vk_get_default_ਬਫਰ_type(backend) || tensor->ਬਫਰ->buft == ggml_backend_vk_host_ਬਫਰ_type()) && "unsupported buffer type");

    ਜੇ (size == 0) {
        ਵਾਪਸ;
    }

    ggml_backend_vk_ਬਫਰ_context * buf_ctx = (ggml_backend_vk_ਬਫਰ_context *)tensor->ਬਫਰ->context;

    vk_context cpy_ctx;

    ਜੇ (ctx->device->async_use_transfer_ਕਤਾਰ) {
        cpy_ctx = ggml_vk_get_transfer_ctx(ctx);
    } ਨਹੀਂਤਾਂ {
        cpy_ctx = ggml_vk_get_compute_ctx(ctx);
    }

    vk_ਬਫਰ buf = buf_ctx->dev_ਬਫਰ;

    ਸਵੈਚਾਲਿਤ dst_ਵਿੱਥ = vk_tensor_ਵਿੱਥ(tensor) + tensor->view_offs + ਵਿੱਥ;

    ਤਾਰਕਿਕ ret = ggml_vk_ਬਫਰ_write_2d_async(cpy_ctx, buf, dst_ਵਿੱਥ, ਅੰਕੜੇ, stride_ਅੰਕੜੇ, stride_tensor, size, n_copies);

    ਜੇ (!ret) {
        ਅਚਲ size_t staging_ਮਾਪ = size * n_copies;
        ggml_vk_ensure_sync_staging_ਬਫਰ(ctx, staging_ਮਾਪ);
        ggml_vk_sync_buffers(ਖਾਲੀ_ਸੂਚਕ, cpy_ctx);

        std::vector<vk::ਬਫਰ_Copy> slices(1);
        ਜੇ (size == stride_tensor) {
            slices[0].src_ਵਿੱਥ_2 = 0;
            slices[0].dst_ਵਿੱਥ_2 = dst_ਵਿੱਥ;
            slices[0].size = staging_ਮਾਪ;
        } ਨਹੀਂਤਾਂ {
            slices.resize(n_copies);
            ਲਈ (size_t i = 0; i < n_copies; i++) {
                slices[i].src_ਵਿੱਥ_2 = i * size;
                slices[i].dst_ਵਿੱਥ_2 = dst_ਵਿੱਥ + i * stride_tensor;
                slices[i].size = size;
            }
        }

        cpy_ctx->s->ਬਫਰ->buf.copy_ਬਫਰ(ctx->sync_staging->ਬਫਰ, buf->ਬਫਰ, slices);

        ਜੇ (size == stride_ਅੰਕੜੇ) {
            deferred_memcpy(ctx->sync_staging->ptr, ਅੰਕੜੇ, staging_ਮਾਪ, &cpy_ctx->ਵਿੱਚ_memcpys);
        } ਨਹੀਂਤਾਂ {
            ਲਈ (size_t i = 0; i < n_copies; i++) {
                deferred_memcpy((uint8_t *)ctx->sync_staging->ptr + i * size, (ਅਚਲ uint8_t *)ਅੰਕੜੇ + i * stride_ਅੰਕੜੇ, size, &cpy_ctx->ਵਿੱਚ_memcpys);
            }
        }
        ggml_vk_synchronize(ctx);
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_vk_ਸੈੱਟ_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ, size_t size) {
    VK_LOG_DEBUG("ggml_backend_vk_set_tensor_async(" << size << ")");
    ggml_backend_vk_ਸੈੱਟ_tensor_2d_async(backend, tensor, ਅੰਕੜੇ, ਵਿੱਥ, size, 1, size, size);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_vk_get_tensor_2d_async(ggml_backend_t backend, ਅਚਲ ggml_tensor * tensor, ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ,
                                                size_t size, size_t n_copies, size_t stride_tensor, size_t stride_ਅੰਕੜੇ) {
    VK_LOG_DEBUG("ggml_backend_vk_get_tensor_2d_async(" << size << ", " << n_copies << ")");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    GGML_ASSERT((tensor->ਬਫਰ->buft == ggml_backend_vk_get_default_ਬਫਰ_type(backend) || tensor->ਬਫਰ->buft == ggml_backend_vk_host_ਬਫਰ_type()) && "unsupported buffer type");

    ਜੇ (size == 0) {
        ਵਾਪਸ;
    }

    ggml_backend_vk_ਬਫਰ_context * buf_ctx = (ggml_backend_vk_ਬਫਰ_context *)tensor->ਬਫਰ->context;

    vk_context compute_ctx = ggml_vk_get_compute_ctx(ctx);

    vk_ਬਫਰ buf = buf_ctx->dev_ਬਫਰ;

    ਸਵੈਚਾਲਿਤ src_ਵਿੱਥ = vk_tensor_ਵਿੱਥ(tensor) + tensor->view_offs + ਵਿੱਥ;
    ਤਾਰਕਿਕ ret = ggml_vk_ਬਫਰ_read_2d_async(compute_ctx, buf, src_ਵਿੱਥ, ਅੰਕੜੇ, stride_tensor, stride_ਅੰਕੜੇ, size, n_copies);

    ਜੇ (!ret) {
        ਅਚਲ size_t staging_ਮਾਪ = size * n_copies;
        ggml_vk_ensure_sync_staging_ਬਫਰ(ctx, staging_ਮਾਪ);
        ggml_vk_sync_buffers(ਖਾਲੀ_ਸੂਚਕ, compute_ctx);

        std::vector<vk::ਬਫਰ_Copy> slices(1);
        ਜੇ (size == stride_tensor) {
            slices[0].src_ਵਿੱਥ_2 = src_ਵਿੱਥ;
            slices[0].dst_ਵਿੱਥ_2 = 0;
            slices[0].size = staging_ਮਾਪ;
        } ਨਹੀਂਤਾਂ {
            slices.resize(n_copies);
            ਲਈ (size_t i = 0; i < n_copies; i++) {
                slices[i].src_ਵਿੱਥ_2 = src_ਵਿੱਥ + i * stride_tensor;
                slices[i].dst_ਵਿੱਥ_2 = i * size;
                slices[i].size = size;
            }
        }

        compute_ctx->s->ਬਫਰ->buf.copy_ਬਫਰ(buf->ਬਫਰ, ctx->sync_staging->ਬਫਰ, slices);

        ਜੇ (size == stride_ਅੰਕੜੇ) {
            deferred_memcpy(ਅੰਕੜੇ, ctx->sync_staging->ptr, staging_ਮਾਪ, &compute_ctx->ਬਾਹਰਲਾ_memcpys);
        } ਨਹੀਂਤਾਂ {
            ਲਈ (size_t i = 0; i < n_copies; i++) {
                deferred_memcpy((uint8_t *)ਅੰਕੜੇ + i * stride_ਅੰਕੜੇ, (ਅਚਲ uint8_t *)ctx->sync_staging->ptr + i * size, size, &compute_ctx->ਬਾਹਰਲਾ_memcpys);
            }
        }
        ggml_vk_synchronize(ctx);
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_vk_get_tensor_async(ggml_backend_t backend, ਅਚਲ ggml_tensor * tensor, ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ, size_t size) {
    VK_LOG_DEBUG("ggml_backend_vk_get_tensor_async(" << size << ")");
    ggml_backend_vk_get_tensor_2d_async(backend, tensor, ਅੰਕੜੇ, ਵਿੱਥ, size, 1, size, size);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_vk_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, ਅਚਲ ggml_tensor * src, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_backend_vk_cpy_tensor_async(" << src << " -> " << dst << ", size=" << ggml_nbytes(src) << ")");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend_dst->context;

    // Skip zero-size tensors
    ਜੇ (ggml_nbytes(src) == 0) {
        ਵਾਪਸ ਸੱਚ;
    }

    ਜੇ (dst->ਬਫਰ->buft != ggml_backend_vk_get_default_ਬਫਰ_type(backend_dst)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ggml_backend_vk_ਬਫਰ_context * dst_buf_ctx = (ggml_backend_vk_ਬਫਰ_context *)dst->ਬਫਰ->context;
    vk_ਬਫਰ dst_buf = dst_buf_ctx->dev_ਬਫਰ;

    ਜੇ (ggml_backend_ਬਫਰ_is_vk(src->ਬਫਰ)) {
        ggml_backend_vk_ਬਫਰ_context * src_buf_ctx = (ggml_backend_vk_ਬਫਰ_context *)src->ਬਫਰ->context;

        // Async copy only works within the same device
        ਜੇ (src_buf_ctx->dev_ਬਫਰ->device != dst_buf->device) {
            ਵਾਪਸ ਝੂਠ;
        }

        vk_context compute_ctx = ggml_vk_get_compute_ctx(ctx);

        ggml_vk_ਬਫਰ_copy_async(compute_ctx, dst_buf, vk_tensor_ਵਿੱਥ(dst) + dst->view_offs,
                                   src_buf_ctx->dev_ਬਫਰ, vk_tensor_ਵਿੱਥ(src) + src->view_offs,
                                   ggml_nbytes(src));
        ਵਾਪਸ ਸੱਚ;
    }

    ਜੇ (ggml_backend_ਬਫਰ_is_host(src->ਬਫਰ)) {
        vk_ਬਫਰ pinned_buf = ਖਾਲੀ_ਸੂਚਕ;
        size_t pinned_ਵਿੱਥ = 0;
        ggml_vk_host_get(ctx->device, src->ਅੰਕੜੇ, pinned_buf, pinned_ਵਿੱਥ);
        ਜੇ (pinned_buf == ਖਾਲੀ_ਸੂਚਕ) {
            ਵਾਪਸ ਝੂਠ;
        }

        // If the backend is idle, use a CPU copy to avoid GPU synchronization overhead.
        ਸਥਿਰ constexpr size_t max_cpu_copy_ਮਾਪ = 128 * 1024;
        ਅਚਲ ਤਾਰਕਿਕ src_backend_synchronous = backend_src->iface.synchronize == ਖਾਲੀ_ਸੂਚਕ;
        ਅਚਲ ਤਾਰਕਿਕ transfer_idle = !ctx->device->async_use_transfer_ਕਤਾਰ ||
                                   ctx->transfer_semaphore_ਆਖਰੀ_submitted == ctx->transfer_semaphore.value;
        ਅਚਲ ਤਾਰਕਿਕ backend_idle = ctx->compute_ctx.expired() && ctx->transfer_ctx.expired() &&
                                  !ctx->submit_pending && !ctx->almost_ready_fence_pending && transfer_idle;
        ਅਚਲ ਤਾਰਕਿਕ dst_host_coherent =
            (dst_buf->ਸਮ੍ਰਿਤੀ_property_flags & (vk::ਸਮ੍ਰਿਤੀ_Property_Flag_ਬਿੱਟ::eHostVisible | vk::ਸਮ੍ਰਿਤੀ_Property_Flag_ਬਿੱਟ::eHostCoherent)) ==
            (vk::ਸਮ੍ਰਿਤੀ_Property_Flag_ਬਿੱਟ::eHostVisible | vk::ਸਮ੍ਰਿਤੀ_Property_Flag_ਬਿੱਟ::eHostCoherent);

        ਜੇ ((backend_src == backend_dst || src_backend_synchronous) && backend_idle && dst_host_coherent && ggml_nbytes(src) <= max_cpu_copy_ਮਾਪ) {
            ggml_vk_ਬਫਰ_write(dst_buf, vk_tensor_ਵਿੱਥ(dst) + dst->view_offs, src->ਅੰਕੜੇ, ggml_nbytes(src));
            ਵਾਪਸ ਸੱਚ;
        }

        vk_context cpy_ctx;
        ਜੇ (ctx->device->async_use_transfer_ਕਤਾਰ) {
            cpy_ctx = ggml_vk_get_transfer_ctx(ctx);
        } ਨਹੀਂਤਾਂ {
            cpy_ctx = ggml_vk_get_compute_ctx(ctx);
        }

        ਵਾਪਸ ggml_vk_ਬਫਰ_write_async(cpy_ctx, dst_buf,
                                          vk_tensor_ਵਿੱਥ(dst) + dst->view_offs,
                                          src->ਅੰਕੜੇ, ggml_nbytes(src));
    }

    ਵਾਪਸ ਝੂਠ;
}

ਖਾਲੀ ggml_vk_synchronize(ggml_backend_vk_context * ctx) {
    VK_LOG_DEBUG("ggml_vk_synchronize()");

    ਤਾਰਕਿਕ do_transfer = !ctx->compute_ctx.expired();

    ਜੇ (ggml_vk_submit_transfer_ctx(ctx)) {
        ctx->submit_pending = ਸੱਚ;
    }

    vk_context compute_ctx;
    vk_command_ਬਫਰ* cmd_buf = ਖਾਲੀ_ਸੂਚਕ;
    ਜੇ (do_transfer) {
        compute_ctx = ctx->compute_ctx.ਤਾਲਾ();
        ਜੇ (compute_ctx->s) {
            cmd_buf = compute_ctx->s->ਬਫਰ;
        }

        ggml_vk_ctx_ਸਿਰਾ(compute_ctx);

        ਲਈ (ਸਵੈਚਾਲਿਤ& cpy : compute_ctx->ਵਿੱਚ_memcpys) {
            ਸਮ੍ਰਿਤੀ_ਨਕਲ(cpy.dst, cpy.src, cpy.n);
        }

        ਜੇ (ctx->device->serialize_submissions) {
            ggml_vk_submit(compute_ctx, ctx->fence);
            VK_CHECK(ctx->device->device.waitForFences({ ctx->fence }, ਸੱਚ, UINT64_MAX), "synchronize waitForFences", ctx->device);
            ctx->device->device.resetFences({ ctx->fence });
        } ਨਹੀਂਤਾਂ {
            ggml_vk_submit(compute_ctx, {});
        }
        ctx->submit_pending = ਸੱਚ;
    }

    ਜੇ (ctx->submit_pending) {
        ਜੇ (ctx->device->serialize_submissions) {
            ctx->submit_pending = ਝੂਠ;
        } ਨਹੀਂਤਾਂ ਜੇ (ctx->device->async_use_transfer_ਕਤਾਰ && ctx->transfer_semaphore_ਆਖਰੀ_submitted < ctx->transfer_semaphore.value) {
            vk::TimelineSemaphoreSubmitInfo tl_info{
                1, &ctx->transfer_semaphore.value,
                0, ਖਾਲੀ_ਸੂਚਕ,
            };
            vk::PipelineStageFlags stage = ctx->device->transfer_ਕਤਾਰ->stage_flags;
            vk::SubmitInfo si{
                1, &ctx->transfer_semaphore.s, &stage,
                0, ਖਾਲੀ_ਸੂਚਕ,
                0, ਖਾਲੀ_ਸੂਚਕ,
            };
            si.ਸੈੱਟ_PNext(&tl_info);
            ctx->device->compute_ਕਤਾਰ->handle->submit({ si }, ctx->fence);
            ctx->transfer_semaphore_ਆਖਰੀ_submitted = ctx->transfer_semaphore.value;
        } ਨਹੀਂਤਾਂ {
            ctx->device->compute_ਕਤਾਰ->handle->submit({}, ctx->fence);
        }
        ਜੇ (!ctx->device->serialize_submissions) {
            ggml_vk_wait_for_fence(ctx);
        }
        ctx->submit_pending = ਝੂਠ;
        ਜੇ (cmd_buf) {
            cmd_buf->ਵਿੱਚ_use = ਝੂਠ;
            cmd_buf->buf.reset();
        }
    }

    ਜੇ (do_transfer) {
        ਲਈ (ਸਵੈਚਾਲਿਤ& cpy : compute_ctx->ਬਾਹਰਲਾ_memcpys) {
            ਸਮ੍ਰਿਤੀ_ਨਕਲ(cpy.dst, cpy.src, cpy.n);
        }
        ctx->compute_ctx.reset();
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_vk_synchronize(ggml_backend_t backend) {
    VK_LOG_DEBUG("ggml_backend_vk_synchronize()");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;

    ggml_vk_synchronize(ctx);

    ggml_vk_ਗ੍ਰਾਫ_cleanup(ctx);
}

ਤਾਰਕਿਕ ggml_vk_is_ਖਾਲੀ_ਥਾਵਾਂ(ggml_tensor * ਗੰਢ) {
    ਵਾਪਸ ggml_is_ਖਾਲੀ_ਥਾਵਾਂ(ਗੰਢ) || ਗੰਢ->op == GGML_OP_NONE || ਗੰਢ->op == GGML_OP_RESHAPE || ਗੰਢ->op == GGML_OP_TRANSPOSE || ਗੰਢ->op == GGML_OP_VIEW || ਗੰਢ->op == GGML_OP_ਕ੍ਰਮ_ਬਦਲੋ;
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_vk_can_fuse_unary_mul(ਅਚਲ ਬਣਤਰ ggml_cgraph * cgraph, ਪੂਰਨ unary_idx, ਪੂਰਨ mul_idx) {
    ਅਚਲ ggml_tensor * unary = cgraph->nodes[unary_idx];
    ਅਚਲ ggml_tensor * mul = cgraph->nodes[mul_idx];

    ਜੇ (ggml_vk_unary_mul_op_ਸੂਚਕਾਂਕ(ggml_get_unary_op(unary)) < 0) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (unary->type != mul->type) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (mul->src[0] != unary && mul->src[1] != unary) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਅਚਲ ggml_tensor * other = (mul->src[0] == unary) ? mul->src[1] : mul->src[0];
    ਜੇ (other == ਖਾਲੀ_ਸੂਚਕ || other->type != unary->type) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (!ggml_is_contiguous_1(other) || !ggml_is_contiguous_1(unary->src[0])) {
        ਵਾਪਸ ਝੂਠ;
    }
    // fastmod needs src to tile into dst
    ਜੇ (mul->src[0] == unary) {
        ਵਾਪਸ ggml_can_repeat(other, unary);
    }
    ਵਾਪਸ ggml_can_repeat(unary, mul->src[0]);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_vk_can_fuse_unary_mul_pair(ਅਚਲ ਬਣਤਰ ggml_cgraph * cgraph, ਪੂਰਨ ਗੰਢ_idx) {
    ਅਚਲ ਗਿਣਤੀ ggml_op ops[]    = { GGML_OP_UNARY, GGML_OP_MUL };
    ਅਚਲ ਪੂਰਨ           outputs[] = { ਗੰਢ_idx + 1 };
    ਵਾਪਸ ggml_can_fuse_subgraph(cgraph, ਗੰਢ_idx, 2, ops, outputs, 1) &&
           ggml_vk_can_fuse_unary_mul(cgraph, ਗੰਢ_idx, ਗੰਢ_idx + 1);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_vk_can_fuse_hc_post_gate(ਅਚਲ ਬਣਤਰ ggml_cgraph * cgraph, ਪੂਰਨ ਗੰਢ_idx) {
    ਅਚਲ ggml_tensor * scale_ਵਿੱਚ  = cgraph->nodes[ਗੰਢ_idx];
    ਅਚਲ ggml_tensor * sigmoid   = cgraph->nodes[ਗੰਢ_idx + 1];
    ਅਚਲ ggml_tensor * scale_ਬਾਹਰਲਾ = cgraph->nodes[ਗੰਢ_idx + 2];

    // the shader folds scale -> sigmoid -> scale; a bias on either scale is not handled
    ਵਾਪਸ ggml_get_unary_op(sigmoid) == GGML_UNARY_OP_SIGMOID &&
           ggml_get_op_params_f32(scale_ਵਿੱਚ, 1) == 0.0f &&
           ggml_get_op_params_f32(scale_ਬਾਹਰਲਾ, 1) == 0.0f &&
           scale_ਵਿੱਚ->src[0]->type == GGML_TYPE_F32 &&
           ggml_are_same_shape(scale_ਵਿੱਚ->src[0], scale_ਬਾਹਰਲਾ);
}

ਤਾਰਕਿਕ ggml_vk_can_fuse(ਅਚਲ ggml_backend_vk_context * ctx, ਅਚਲ ਬਣਤਰ ggml_cgraph * cgraph, ਪੂਰਨ ਗੰਢ_idx, std::initializer_list<ਗਿਣਤੀ ggml_op> ops) {
    ਜੇ (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_MUL) {
        ਵਾਪਸ ggml_vk_can_fuse_unary_mul_pair(cgraph, ਗੰਢ_idx);
    }

    ਜੇ (!ggml_can_fuse(cgraph, ਗੰਢ_idx, ops)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ ((ops.size() == 2 || ops.size() == 3 || ops.size() == 4) &&
        ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL) {
        // additional constraints specific to this fusion
        ਅਚਲ ggml_tensor *rms_norm = cgraph->nodes[ਗੰਢ_idx];
        ਅਚਲ ggml_tensor *mul = cgraph->nodes[ਗੰਢ_idx + 1];

        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);
        // rms_norm only supports f32
        ਜੇ (mul->src[0]->type != GGML_TYPE_F32 ||
            mul->src[1]->type != GGML_TYPE_F32 ||
            mul->type != GGML_TYPE_F32) {
            ਵਾਪਸ ਝੂਠ;
        }
        // if rms_norm is the B operand, then we don't handle broadcast
        ਜੇ (rms_norm == mul->src[1] &&
            !ggml_are_same_shape(mul->src[0], rms_norm)) {
            ਵਾਪਸ ਝੂਠ;
        }
        // rms_norm shader assumes contiguous rows
        ਜੇ (!ggml_is_contiguous_ਪੰਕਤੀਆਂ(mul->src[0]) || !ggml_is_contiguous_ਪੰਕਤੀਆਂ(mul->src[1])) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਜੇ (ops.size() >= 3 && ops.begin()[2] == GGML_OP_ਜੋੜੋ) {
            ਅਚਲ ggml_tensor *ਜੋੜੋ_2 = cgraph->nodes[ਗੰਢ_idx + 2];
            ਅਚਲ ggml_tensor *residual = ਜੋੜੋ_2->src[0] == mul ? ਜੋੜੋ_2->src[1] : ਜੋੜੋ_2->src[0];
            ਜੇ (ਜੋੜੋ_2->src[0] != mul && ਜੋੜੋ_2->src[1] != mul) {
                ਵਾਪਸ ਝੂਠ;
            }
            ਜੇ (residual->type != GGML_TYPE_F32 || ਜੋੜੋ_2->type != GGML_TYPE_F32 ||
                !ggml_are_same_shape(ਜੋੜੋ_2, residual) || !ggml_is_contiguous(residual) ||
                !ggml_is_contiguous(ਜੋੜੋ_2) || get_misalign_ਬਾਈਟਾਂ(ctx, residual) != 0) {
                ਵਾਪਸ ਝੂਠ;
            }

            ਅਚਲ ggml_tensor *dst = ਜੋੜੋ_2;
            ਜੇ (ops.size() == 4) {
                ਜੇ (ops.begin()[3] != GGML_OP_MUL) {
                    ਵਾਪਸ ਝੂਠ;
                }

                ਅਚਲ ggml_tensor *post_mul = cgraph->nodes[ਗੰਢ_idx + 3];
                ਅਚਲ ggml_tensor *scale = post_mul->src[0] == ਜੋੜੋ_2 ? post_mul->src[1] : post_mul->src[0];
                ਜੇ (post_mul->src[0] != ਜੋੜੋ_2 && post_mul->src[1] != ਜੋੜੋ_2) {
                    ਵਾਪਸ ਝੂਠ;
                }
                // The shader reads data_e[0], so the final multiply must use a scalar.
                ਜੇ (scale->type != GGML_TYPE_F32 || post_mul->type != GGML_TYPE_F32 ||
                    ggml_nelements(scale) != 1 || !ggml_is_contiguous(post_mul) ||
                    get_misalign_ਬਾਈਟਾਂ(ctx, scale) != 0) {
                    ਵਾਪਸ ਝੂਠ;
                }
                dst = post_mul;
            }

            ਜੇ (get_misalign_ਬਾਈਟਾਂ(ctx, dst) != 0) {
                ਵਾਪਸ ਝੂਠ;
            }
        }
    }

    ਸਵੈਚਾਲਿਤ ਅਚਲ &mm_ਜੋੜੋ_ok = [&](ਅਚਲ ggml_tensor *mul, ਅਚਲ ggml_tensor *ਜੋੜੋ_2) {
        ਅਚਲ ggml_tensor *bias = ਜੋੜੋ_2->src[0] == mul ? ਜੋੜੋ_2->src[1] : ਜੋੜੋ_2->src[0];

        // mat-vec only
        ਜੇ (ggml_nrows(mul) != 1) {
            ਵਾਪਸ ਝੂਠ;
        }
        // shaders assume the types match
        ਜੇ (mul->type != bias->type) {
            ਵਾਪਸ ਝੂਠ;
        }
        // shaders reuse the D shape for bias
        ਜੇ (!ggml_are_same_shape(mul, bias) ||
            !ggml_are_same_stride(mul, bias)) {
            ਵਾਪਸ ਝੂਠ;
        }
        // unaligned bias isn't handled
        ਜੇ (get_misalign_ਬਾਈਟਾਂ(ctx, bias) != 0) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਵਾਪਸ ਸੱਚ;
    };

    ਜੇ ((ops.size() == 2 || ops.size() == 3) && ops.begin()[0] == GGML_OP_MUL_MAT && ops.begin()[1] == GGML_OP_ਜੋੜੋ) {
        // additional constraints specific to this fusion
        ਅਚਲ ggml_tensor *mul = cgraph->nodes[ਗੰਢ_idx];
        ਅਚਲ ggml_tensor *ਜੋੜੋ_2 = cgraph->nodes[ਗੰਢ_idx + 1];

        ਜੇ (!mm_ਜੋੜੋ_ok(mul, ਜੋੜੋ_2)) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਜੇ (ops.size() == 3) {
            ਜੇ (ops.begin()[2] != GGML_OP_ਜੋੜੋ) {
                ਵਾਪਸ ਝੂਠ;
            }
            ਜੇ (!mm_ਜੋੜੋ_ok(ਜੋੜੋ_2, cgraph->nodes[ਗੰਢ_idx + 2])) {
                ਵਾਪਸ ਝੂਠ;
            }
        }
    }

    ਸਵੈਚਾਲਿਤ ਅਚਲ &mmid_mul_ok = [&](ਅਚਲ ggml_tensor *mmid, ਅਚਲ ggml_tensor *mul) {
        ਅਚਲ ggml_tensor *scale = mul->src[1];

        ਜੇ (mmid != mul->src[0]) {
            ਵਾਪਸ ਝੂਠ;
        }
        // mat-vec only
        ਜੇ (!ggml_vk_use_mul_mat_vec_id(cgraph, ਗੰਢ_idx)) {
            ਵਾਪਸ ਝੂਠ;
        }
        // shaders assume the types match
        ਜੇ (mmid->type != scale->type) {
            ਵਾਪਸ ਝੂਠ;
        }
        // shaders assume the bias is contiguous
        ਜੇ (!ggml_is_contiguous(scale)) {
            ਵਾਪਸ ਝੂਠ;
        }
        // unaligned bias isn't handled
        ਜੇ (get_misalign_ਬਾਈਟਾਂ(ctx, scale) != 0) {
            ਵਾਪਸ ਝੂਠ;
        }
        // shader only indexes by expert index
        ਜੇ (scale->ne[0] != 1 ||
            scale->ne[1] != mul->ne[1] ||
            scale->ne[2] != 1 ||
            scale->ne[3] != 1) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਵਾਪਸ ਸੱਚ;
    };

    ਜੇ ((ops.size() == 2 || ops.size() == 3) && ops.begin()[0] == GGML_OP_MUL_MAT_ID && ops.begin()[1] == GGML_OP_ਜੋੜੋ_ID) {
        // additional constraints specific to this fusion
        ਅਚਲ ggml_tensor *mul = cgraph->nodes[ਗੰਢ_idx];
        ਅਚਲ ggml_tensor *ਜੋੜੋ_2 = cgraph->nodes[ਗੰਢ_idx + 1];
        ਅਚਲ ggml_tensor *bias = ਜੋੜੋ_2->src[1];

        ਜੇ (mul != ਜੋੜੋ_2->src[0]) {
            ਵਾਪਸ ਝੂਠ;
        }
        // mat-vec only
        ਜੇ (!ggml_vk_use_mul_mat_vec_id(cgraph, ਗੰਢ_idx)) {
            ਵਾਪਸ ਝੂਠ;
        }
        // shaders assume the types match
        ਜੇ (mul->type != bias->type) {
            ਵਾਪਸ ਝੂਠ;
        }
        // shaders assume the bias is contiguous
        ਜੇ (!ggml_is_contiguous(bias)) {
            ਵਾਪਸ ਝੂਠ;
        }
        // the ID tensor must be the same for mul_mat_id and add_id
        ਜੇ (mul->src[2] != ਜੋੜੋ_2->src[2]) {
            ਵਾਪਸ ਝੂਠ;
        }
        // unaligned bias isn't handled
        ਜੇ (get_misalign_ਬਾਈਟਾਂ(ctx, bias) != 0) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਜੇ (ops.size() == 3) {
            ਜੇ (ops.begin()[2] != GGML_OP_MUL) {
                ਵਾਪਸ ਝੂਠ;
            }
            ਅਚਲ ggml_tensor *mul = cgraph->nodes[ਗੰਢ_idx + 2];
            ਵਾਪਸ mmid_mul_ok(ਜੋੜੋ_2, mul);
        }
    }

    ਜੇ (ops.size() == 2 && ops.begin()[0] == GGML_OP_MUL_MAT_ID && ops.begin()[1] == GGML_OP_MUL) {
        // additional constraints specific to this fusion
        ਅਚਲ ggml_tensor *mmid = cgraph->nodes[ਗੰਢ_idx];
        ਅਚਲ ggml_tensor *mul = cgraph->nodes[ਗੰਢ_idx + 1];

        ਜੇ (!mmid_mul_ok(mmid, mul)) {
            ਵਾਪਸ ਝੂਠ;
        }
    }

    ਵਾਪਸ ਸੱਚ;
}

ਤਾਰਕਿਕ ggml_vk_can_fuse_ssm_conv(ਅਚਲ ggml_backend_vk_context * ctx, ਅਚਲ ਬਣਤਰ ggml_cgraph * cgraph,
                                      ਪੂਰਨ ਗੰਢ_idx, ਪੂਰਨ ਅੰਕ_extra) {
    ਅਚਲ ggml_tensor * conv = cgraph->nodes[ਗੰਢ_idx];
    ਜੇ (conv->op != GGML_OP_SSM_CONV) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ ggml_tensor * silu = ਖਾਲੀ_ਸੂਚਕ;
    ਅਚਲ ggml_tensor * bias = ਖਾਲੀ_ਸੂਚਕ;

    ਜੇ (ਅੰਕ_extra == 1) {
        ਜੇ (!ggml_can_fuse(cgraph, ਗੰਢ_idx, { GGML_OP_SSM_CONV, GGML_OP_UNARY })) {
            ਵਾਪਸ ਝੂਠ;
        }
        silu = cgraph->nodes[ਗੰਢ_idx + 1];
    } ਨਹੀਂਤਾਂ ਜੇ (ਅੰਕ_extra == 2) {
        ਜੇ (!ggml_can_fuse(cgraph, ਗੰਢ_idx, { GGML_OP_SSM_CONV, GGML_OP_ਜੋੜੋ, GGML_OP_UNARY })) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਅਚਲ ggml_tensor * ਜੋੜੋ_2 = cgraph->nodes[ਗੰਢ_idx + 1];
        silu = cgraph->nodes[ਗੰਢ_idx + 2];
        bias = (ਜੋੜੋ_2->src[0] == conv) ? ਜੋੜੋ_2->src[1] : ਜੋੜੋ_2->src[0];

        ਜੇ (bias->type != GGML_TYPE_F32 || !ggml_is_contiguous(bias)) {
            ਵਾਪਸ ਝੂਠ;
        }
        // bias must be channel-wise (one element per channel of the conv output)
        ਜੇ (ggml_nelements(bias) != conv->ne[0] || bias->ne[0] != conv->ne[0]) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਜੇ (ਜੋੜੋ_2->type != GGML_TYPE_F32) {
            ਵਾਪਸ ਝੂਠ;
        }
        // The shader doesn't apply per-tensor offsets, so reject misaligned bias.
        ਜੇ (get_misalign_ਬਾਈਟਾਂ(ctx, bias) != 0) {
            ਵਾਪਸ ਝੂਠ;
        }
    } ਨਹੀਂਤਾਂ {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (ggml_get_unary_op(silu) != GGML_UNARY_OP_SILU) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (conv->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
        ਵਾਪਸ ਝੂਠ;
    }
    // The shader writes to the fused dst using its own strides, but the push constants don't
    // carry a per-tensor offset, so the binding must be naturally aligned.
    ਜੇ (get_misalign_ਬਾਈਟਾਂ(ctx, silu) != 0) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਵਾਪਸ ਸੱਚ;
}

ਤਾਰਕਿਕ ggml_vk_can_fuse_topk_moe(ggml_backend_vk_context * ctx, ਅਚਲ ਬਣਤਰ ggml_cgraph * cgraph,
                                      ਪੂਰਨ ਗੰਢ_idx, topk_moe_ਬਹੁਲਕ mode) {

    ਅਚਲ ggml_tensor * softmax;
    ਅਚਲ ggml_tensor * weights;
    ਅਚਲ ggml_tensor * get_ਪੰਕਤੀਆਂ;
    ਅਚਲ ggml_tensor * argsort;

    ਚੋਣ (mode) {
    ਹਾਲਤ TOPK_MOE_EARLY_SOFTMAX_NORM:
        softmax = cgraph->nodes[ਗੰਢ_idx + 0];
        weights = cgraph->nodes[ਗੰਢ_idx + 9];
        get_ਪੰਕਤੀਆਂ = cgraph->nodes[ਗੰਢ_idx + 4];
        argsort = cgraph->nodes[ਗੰਢ_idx + 2];
        ਰੋਕੋ;
    ਹਾਲਤ TOPK_MOE_SIGMOID_NORM_BIAS:
        softmax = cgraph->nodes[ਗੰਢ_idx + 0]; // really sigmoid
        weights = cgraph->nodes[ਗੰਢ_idx + 10];
        get_ਪੰਕਤੀਆਂ = cgraph->nodes[ਗੰਢ_idx + 5];
        argsort = cgraph->nodes[ਗੰਢ_idx + 3];
        ਜੇ (ggml_get_unary_op(softmax) != GGML_UNARY_OP_SIGMOID) {
            ਵਾਪਸ ਝੂਠ;
        }
        // bias is expected to be 1D
        ਜੇ (ggml_nrows(cgraph->nodes[ਗੰਢ_idx + 2]->src[1]) != 1 ||
            !ggml_is_contiguous(cgraph->nodes[ਗੰਢ_idx + 2]->src[1])) {
            ਵਾਪਸ ਝੂਠ;
        }
        // sigmoid fusion seems to generate infinities on moltenvk
        ਜੇ (ctx->device->driver_id == vk::DriverId::eMoltenvk) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਰੋਕੋ;
    ਹਾਲਤ TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS:
        softmax = cgraph->nodes[ਗੰਢ_idx + 0]; // really softplus
        weights = cgraph->nodes[ਗੰਢ_idx + 11];
        get_ਪੰਕਤੀਆਂ = cgraph->nodes[ਗੰਢ_idx + 6];
        argsort = cgraph->nodes[ਗੰਢ_idx + 4];
        ਜੇ (ggml_get_unary_op(softmax) != GGML_UNARY_OP_SOFTPLUS) {
            ਵਾਪਸ ਝੂਠ;
        }
        // bias is expected to be 1D
        ਜੇ (ggml_nrows(cgraph->nodes[ਗੰਢ_idx + 3]->src[1]) != 1 ||
            !ggml_is_contiguous(cgraph->nodes[ਗੰਢ_idx + 3]->src[1])) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਰੋਕੋ;
    ਹਾਲਤ TOPK_MOE_EARLY_SOFTMAX:
        softmax = cgraph->nodes[ਗੰਢ_idx + 0];
        weights = cgraph->nodes[ਗੰਢ_idx + 4];
        get_ਪੰਕਤੀਆਂ = cgraph->nodes[ਗੰਢ_idx + 4];
        argsort = cgraph->nodes[ਗੰਢ_idx + 2];
        ਰੋਕੋ;
    ਹਾਲਤ TOPK_MOE_LATE_SOFTMAX:
        softmax = cgraph->nodes[ਗੰਢ_idx + 4];
        weights = cgraph->nodes[ਗੰਢ_idx + 5];
        get_ਪੰਕਤੀਆਂ = cgraph->nodes[ਗੰਢ_idx + 2];
        argsort = cgraph->nodes[ਗੰਢ_idx + 0];
        ਰੋਕੋ;
    ਮੂਲ:
        ਵਾਪਸ ਝੂਠ;
    }

    ggml_tensor * probs = get_ਪੰਕਤੀਆਂ->src[0];
    ਜੇ (probs->op != GGML_OP_RESHAPE) {
        ਵਾਪਸ ਝੂਠ;
    }
    probs = probs->src[0];
    ggml_tensor * selection_probs = argsort->src[0];

    ਜੇ (probs != selection_probs &&
        mode != TOPK_MOE_SIGMOID_NORM_BIAS &&
        mode != TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (!ggml_is_contiguous(softmax->src[0]) || !ggml_is_contiguous(weights)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (softmax->op == GGML_OP_SOFT_MAX) {
        ਅਚਲ ਦਸ਼ਮਲਵ * op_params = (ਅਚਲ ਦਸ਼ਮਲਵ *)softmax->op_params;

        ਦਸ਼ਮਲਵ scale = op_params[0];
        ਦਸ਼ਮਲਵ max_bias = op_params[1];

        ਜੇ (scale != 1.0f || max_bias != 0.0f) {
            ਵਾਪਸ ਝੂਠ;
        }

        // don't fuse when masks or sinks are present
        ਜੇ (softmax->src[1] || softmax->src[2]) {
            ਵਾਪਸ ਝੂਠ;
        }
    }

    ਅਚਲ ਪੂਰਨ n_expert = softmax->ne[0];
    ਜੇ (n_expert > (1 << (ਅੰਕ_topk_moe_pipelines-1))) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (!ctx->device->subgroup_arithmetic ||
        !ctx->device->subgroup_shuffle ||
        !ctx->device->subgroup_require_ਭਰਿਆ_support ||
        ctx->device->disable_fusion) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਵਾਪਸ ਸੱਚ;
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_vk_ਮੇਲ_ops(ਅਚਲ ਬਣਤਰ ggml_cgraph * cgraph, ਪੂਰਨ ਗੰਢ_idx,
                              ਅਚਲ std::initializer_list<ggml_op> & ops) {
    ਜੇ (ਗੰਢ_idx + (ਪੂਰਨ) ops.size() > cgraph->n_nodes) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਲਈ (size_t j = 0; j < ops.size(); ++j) {
        ਅਚਲ ggml_tensor * ਗੰਢ = cgraph->nodes[ਗੰਢ_idx + j];
        ਜੇ (ਗੰਢ->op != ops.begin()[j] ||
            (ਗੰਢ->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 ||
            (ਗੰਢ->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
            ਵਾਪਸ ਝੂਠ;
        }
    }
    ਵਾਪਸ ਸੱਚ;
}

ਤਾਰਕਿਕ ggml_vk_can_fuse_topk_qsa(ggml_backend_vk_context * ctx, ਅਚਲ ਬਣਤਰ ggml_cgraph * cgraph, ਪੂਰਨ ਗੰਢ_idx) {
    ਜੇ (ctx->device->disable_fusion || !ctx->device->pipeline_topk_radix_qsa) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ ਪੂਰਨ n_ops = topk_qsa_ਨਮੂਨਾ.size();
    ਜੇ (!ggml_vk_ਮੇਲ_ops(cgraph, ਗੰਢ_idx, topk_qsa_ਨਮੂਨਾ) ||
        !ggml_check_ਕਿਨਾਰੇ(cgraph, ਗੰਢ_idx, topk_qsa_ਕਿਨਾਰੇ)) {
        ਵਾਪਸ ਝੂਠ;
    }

    // elided nodes must be single-use (cpy counts its own src[1] self-reference)
    ਲਈ (ਪੂਰਨ j = 0; j < n_ops - 1; ++j) {
        ਅਚਲ ggml_tensor * ਗੰਢ = cgraph->nodes[ਗੰਢ_idx + j];
        ਅਚਲ int32_t want = ਗੰਢ->op == GGML_OP_CPY ? 2 : 1;
        ਜੇ (ggml_ਗੰਢ_get_use_ਗਿਣਤੀ(cgraph, ਗੰਢ_idx + j) != want) {
            ਵਾਪਸ ਝੂਠ;
        }
    }

    ਅਚਲ ggml_tensor * get_ਪੰਕਤੀਆਂ = cgraph->nodes[ਗੰਢ_idx + 0];
    ਅਚਲ ggml_tensor * ਜੋੜੋ_2      = cgraph->nodes[ਗੰਢ_idx + n_ops - 2];
    ਅਚਲ ggml_tensor * ਸਿਖਰ_k    = cgraph->nodes[ਗੰਢ_idx + n_ops - 1];

    ਅਚਲ ggml_tensor * scores   = get_ਪੰਕਤੀਆਂ->src[0]; // [n_tps, n_blocks, n_stream]
    ਅਚਲ ggml_tensor * cell_blk = get_ਪੰਕਤੀਆਂ->src[1]; // [n_kv, n_stream]
    ਅਚਲ ggml_tensor * expanded = ਜੋੜੋ_2->src[0];      // [n_kv, n_tps, n_stream]

    // raw mask: follow the reshape/cpy chain back to the materialized f16 input
    ਅਚਲ ggml_tensor * ਨਕਾਬ = ਜੋੜੋ_2->src[1];
    ਜਦੋਂ (ਨਕਾਬ && (ਨਕਾਬ->op == GGML_OP_RESHAPE || ਨਕਾਬ->op == GGML_OP_CPY)) {
        ਨਕਾਬ = ਨਕਾਬ->src[0];
    }
    ਜੇ (!ਨਕਾਬ || ਨਕਾਬ->type != GGML_TYPE_F16) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (scores->type != GGML_TYPE_F32 || cell_blk->type != GGML_TYPE_I32 || ਸਿਖਰ_k->type != GGML_TYPE_I32) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (!ggml_is_contiguous(scores) || !ggml_is_contiguous(cell_blk) || !ggml_is_contiguous(ਨਕਾਬ) ||
        !ggml_is_contiguous(expanded) || !ggml_is_contiguous(ਸਿਖਰ_k)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ int64_t n_tps    = scores->ne[0];
    ਅਚਲ int64_t n_ਖੰਡਾਂ = scores->ne[1];
    ਅਚਲ int64_t n_ਧਾਰਾ = scores->ne[2];
    ਅਚਲ int64_t n_kv     = cell_blk->ne[0];
    ਅਚਲ int64_t ਚੌੜਾਈ    = ਸਿਖਰ_k->ne[0];

    // pin the indexer layout the shader's addressing assumes
    ਜੇ (scores->ne[3] != 1 || cell_blk->ne[1] != n_ਧਾਰਾ || ggml_nrows(cell_blk) != n_ਧਾਰਾ ||
        ggml_nelements(ਨਕਾਬ) != n_kv * n_tps * n_ਧਾਰਾ ||
        expanded->ne[0] != n_kv || expanded->ne[1] != n_tps || expanded->ne[2] != n_ਧਾਰਾ ||
        ਸਿਖਰ_k->ne[1] != n_tps || ਸਿਖਰ_k->ne[2] != n_ਧਾਰਾ || ਸਿਖਰ_k->ne[3] != 1 ||
        n_ਖੰਡਾਂ <= 0 || n_kv <= 0 || ਚੌੜਾਈ <= 0 || ਚੌੜਾਈ > n_kv) {
        ਵਾਪਸ ਝੂਠ;
    }

    // only worth it in the radix regime; small k uses the faster tournament unfused
    ਅਚਲ uint32_t k_min_pipeline = std::max((uint32_t) log2f(ਦਸ਼ਮਲਵ(ਚੌੜਾਈ)) + 1, ctx->device->subgroup_ਮਾਪ_log2);
    ਜੇ (k_min_pipeline < ਅੰਕ_topk_pipelines && ctx->device->pipeline_topk_f32[k_min_pipeline]) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਵਾਪਸ ਸੱਚ;
}

ਤਾਰਕਿਕ ggml_vk_can_fuse_rope_ਸੈੱਟ_ਪੰਕਤੀਆਂ(ggml_backend_vk_context * ctx, ਅਚਲ ਬਣਤਰ ggml_cgraph * cgraph,
                                           ਪੂਰਨ ਗੰਢ_idx) {
    ਅਚਲ ggml_tensor *rope = cgraph->nodes[ਗੰਢ_idx + 0];
    ਅਚਲ ggml_tensor *view = cgraph->nodes[ਗੰਢ_idx + 1];
    ਅਚਲ ggml_tensor *ਸੈੱਟ_ਪੰਕਤੀਆਂ = cgraph->nodes[ਗੰਢ_idx + 2];

    // The set_rows epilogue uses one index per ne2 slice and does not encode ne3.
    ਜੇ (rope->src[0]->ne[3] != 1) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (ਸੈੱਟ_ਪੰਕਤੀਆਂ->type != GGML_TYPE_F32 && ਸੈੱਟ_ਪੰਕਤੀਆਂ->type != GGML_TYPE_F16) {
        ਵਾਪਸ ਝੂਠ;
    }

    // The shader reads each aligned I64 index as a uvec2 and uses its low 32 bits.
    ਜੇ (ਸੈੱਟ_ਪੰਕਤੀਆਂ->src[1]->type != GGML_TYPE_I64 || !ggml_is_contiguous(ਸੈੱਟ_ਪੰਕਤੀਆਂ->src[1]) ||
        ਸੈੱਟ_ਪੰਕਤੀਆਂ->nb[0] != ggml_type_ਮਾਪ(ਸੈੱਟ_ਪੰਕਤੀਆਂ->type) || get_misalign_ਬਾਈਟਾਂ(ctx, ਸੈੱਟ_ਪੰਕਤੀਆਂ->src[1]) != 0) {
        ਵਾਪਸ ਝੂਠ;
    }

    // SET_ROWS consumes one flattened [ne0*ne1] row for each ne2 slice.
    ਜੇ (!ggml_is_contiguous(view) ||
        view->ne[0] != rope->ne[0] * rope->ne[1] || view->ne[1] != rope->ne[2] ||
        view->ne[2] != 1 || view->ne[3] != 1 ||
        ggml_nelements(ਸੈੱਟ_ਪੰਕਤੀਆਂ->src[1]) != rope->ne[2]) {
        ਵਾਪਸ ਝੂਠ;
    }

    // Only norm/neox/mrope/imrope shaders have the fusion code
    ਅਚਲ ਪੂਰਨ mode = ((ਅਚਲ int32_t *) rope->op_params)[2];
    ਜੇ (mode != GGML_ROPE_TYPE_ਸਾਧਾਰਨ && mode != GGML_ROPE_TYPE_NEOX &&
        mode != GGML_ROPE_TYPE_MROPE && mode != GGML_ROPE_TYPE_IMROPE) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਵਾਪਸ ਸੱਚ;
}

ਤਾਰਕਿਕ ggml_vk_can_fuse_rms_norm_ਸੈੱਟ_ਪੰਕਤੀਆਂ(ggml_backend_vk_context * ctx, ਅਚਲ ਬਣਤਰ ggml_cgraph * cgraph,
                                               ਪੂਰਨ ਗੰਢ_idx) {
    ਅਚਲ ggml_tensor * rms = cgraph->nodes[ਗੰਢ_idx];
    ਅਚਲ ggml_tensor * view = cgraph->nodes[ਗੰਢ_idx + 1];
    ਅਚਲ ggml_tensor * ਸੈੱਟ_ਪੰਕਤੀਆਂ = cgraph->nodes[ਗੰਢ_idx + 2];

    // The RMS kernel reads F32 and writes directly to the F32 or F16 SET_ROWS destination.
    ਜੇ (rms->src[0]->type != GGML_TYPE_F32 || rms->type != GGML_TYPE_F32 ||
        (ਸੈੱਟ_ਪੰਕਤੀਆਂ->type != GGML_TYPE_F32 && ਸੈੱਟ_ਪੰਕਤੀਆਂ->type != GGML_TYPE_F16) ||
        ਸੈੱਟ_ਪੰਕਤੀਆਂ->src[1]->type != GGML_TYPE_I64 || !ggml_is_contiguous(ਸੈੱਟ_ਪੰਕਤੀਆਂ->src[1]) ||
        ਸੈੱਟ_ਪੰਕਤੀਆਂ->nb[0] != ggml_type_ਮਾਪ(ਸੈੱਟ_ਪੰਕਤੀਆਂ->type) || get_misalign_ਬਾਈਟਾਂ(ctx, ਸੈੱਟ_ਪੰਕਤੀਆਂ->src[1]) != 0) {
        ਵਾਪਸ ਝੂਠ;
    }
    // As with the ROPE epilogue, each ne2 slice supplies one flattened row and ne3 is not encoded.
    ਜੇ (rms->ne[3] != 1 || !ggml_is_contiguous(rms->src[0]) || !ggml_is_contiguous(view)) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (view->ne[0] != rms->ne[0] * rms->ne[1] || view->ne[1] != rms->ne[2] ||
        view->ne[2] != 1 || view->ne[3] != 1 ||
        ggml_nelements(ਸੈੱਟ_ਪੰਕਤੀਆਂ->src[1]) != rms->ne[2]) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਵਾਪਸ ਸੱਚ;
}

ਤਾਰਕਿਕ ggml_vk_can_fuse_snake(ggml_backend_vk_context * ctx, ਅਚਲ ਬਣਤਰ ggml_cgraph * cgraph, ਪੂਰਨ ਗੰਢ_idx) {
    GGML_ਅਣਵਰਤਿਆ(ctx);
    ਜੇ (!ggml_can_fuse(cgraph, ਗੰਢ_idx, snake_ਨਮੂਨਾ)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ ggml_tensor * mul0     = cgraph->nodes[ਗੰਢ_idx + 0];
    ਅਚਲ ggml_tensor * sin_ਗੰਢ = cgraph->nodes[ਗੰਢ_idx + 1];
    ਅਚਲ ggml_tensor * sqr      = cgraph->nodes[ਗੰਢ_idx + 2];
    ਅਚਲ ggml_tensor * mul1     = cgraph->nodes[ਗੰਢ_idx + 3];
    ਅਚਲ ggml_tensor * ਜੋੜੋ_2      = cgraph->nodes[ਗੰਢ_idx + 4];

    ਅਚਲ ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
    ਅਚਲ ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];

    ਅਚਲ ggml_tensor * inv_b    = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];
    ਅਚਲ ggml_tensor * x_ਵਿੱਚ_ਜੋੜੋ = (ਜੋੜੋ_2->src[0] == mul1) ? ਜੋੜੋ_2->src[1] : ਜੋੜੋ_2->src[0];

    ਜੇ (x_ਵਿੱਚ_ਜੋੜੋ != x) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (x->type != GGML_TYPE_F32 && x->type != GGML_TYPE_F16 && x->type != GGML_TYPE_BF16) {
        ਵਾਪਸ ਝੂਠ;
    }
    // Shader bindings: data_a is A_TYPE so it follows x's precision, while
    // data_b and data_c are hardcoded float, so the broadcast operands must
    // be F32 regardless of x's type.
    ਜੇ (a->type     != GGML_TYPE_F32) ਵਾਪਸ ਝੂਠ;
    ਜੇ (inv_b->type != GGML_TYPE_F32) ਵਾਪਸ ਝੂਠ;
    // Chain intermediates and output share x's precision (single A_TYPE / D_TYPE pipeline).
    ਜੇ (mul0->type     != x->type) ਵਾਪਸ ਝੂਠ;
    ਜੇ (sin_ਗੰਢ->type != x->type) ਵਾਪਸ ਝੂਠ;
    ਜੇ (sqr->type      != x->type) ਵਾਪਸ ਝੂਠ;
    ਜੇ (mul1->type     != x->type) ਵਾਪਸ ਝੂਠ;
    ਜੇ (ਜੋੜੋ_2->type      != x->type) ਵਾਪਸ ਝੂਠ;
    ਜੇ (!ggml_are_same_shape(a, inv_b)) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (a->ne[0] != 1 || a->ne[1] != x->ne[1]) {
        ਵਾਪਸ ਝੂਠ;
    }
    // Dispatch is 2D over (ne0, ne1), so x and add must be 2D and a / inv_b
    // must collapse to [1, C, 1, 1]. Higher dims are not handled by the shader.
    ਜੇ (x->ne[2]     != 1 || x->ne[3]     != 1) ਵਾਪਸ ਝੂਠ;
    ਜੇ (ਜੋੜੋ_2->ne[2]   != 1 || ਜੋੜੋ_2->ne[3]   != 1) ਵਾਪਸ ਝੂਠ;
    ਜੇ (a->ne[2]     != 1 || a->ne[3]     != 1) ਵਾਪਸ ਝੂਠ;
    ਜੇ (inv_b->ne[2] != 1 || inv_b->ne[3] != 1) ਵਾਪਸ ਝੂਠ;
    // Shader uses idx = i0 + i1 * ne0 and reads data_b[i1] / data_c[i1],
    // so every operand must be contiguous.
    ਜੇ (!ggml_is_contiguous(x) || !ggml_is_contiguous(ਜੋੜੋ_2) ||
        !ggml_is_contiguous(a) || !ggml_is_contiguous(inv_b)) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਵਾਪਸ ਸੱਚ;
}

ਤਾਰਕਿਕ ggml_vk_tensors_overlap(ਅਚਲ ggml_tensor * a, ਅਚਲ ggml_tensor * b, ਤਾਰਕਿਕ elementwise) {
    ggml_backend_vk_ਬਫਰ_context * a_buf_ctx = (ggml_backend_vk_ਬਫਰ_context *)a->ਬਫਰ->context;
    vk_ਬਫਰ a_buf = a_buf_ctx->dev_ਬਫਰ;
    ggml_backend_vk_ਬਫਰ_context * b_buf_ctx = (ggml_backend_vk_ਬਫਰ_context *)b->ਬਫਰ->context;
    vk_ਬਫਰ b_buf = b_buf_ctx->dev_ਬਫਰ;
    ਜੇ (a_buf == b_buf) {
        ਸਵੈਚਾਲਿਤ a_ਅਧਾਰ = vk_tensor_ਵਿੱਥ(a) + a->view_offs;
        ਸਵੈਚਾਲਿਤ a_ਮਾਪ = ggml_nbytes(a);
        ਸਵੈਚਾਲਿਤ b_ਅਧਾਰ = vk_tensor_ਵਿੱਥ(b) + b->view_offs;
        ਸਵੈਚਾਲਿਤ b_ਮਾਪ = ggml_nbytes(b);

        ਜੇ (elementwise && a_ਅਧਾਰ == b_ਅਧਾਰ && a_ਮਾਪ == b_ਮਾਪ) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਜੇ ((b_ਅਧਾਰ <= a_ਅਧਾਰ && a_ਅਧਾਰ < b_ਅਧਾਰ + b_ਮਾਪ) ||
            (a_ਅਧਾਰ <= b_ਅਧਾਰ && b_ਅਧਾਰ < a_ਅਧਾਰ + a_ਮਾਪ)) {
            ਵਾਪਸ ਸੱਚ;
        }
    }
    ਵਾਪਸ ਝੂਠ;
}

ਤਾਰਕਿਕ ggml_vk_can_fuse_rms_norm_mul_rope(ggml_backend_vk_context * ctx, ਅਚਲ ਬਣਤਰ ggml_cgraph * cgraph,
                                               ਪੂਰਨ ਗੰਢ_idx) {
    ਅਚਲ ggml_tensor *rms = cgraph->nodes[ਗੰਢ_idx + 0];
    ਅਚਲ ggml_tensor *mul = cgraph->nodes[ਗੰਢ_idx + 1];
    ਅਚਲ ggml_tensor *rope = cgraph->nodes[ਗੰਢ_idx + 2];

    ਅਚਲ ਪੂਰਨ mode = ((ਅਚਲ int32_t *) rope->op_params)[2];

    // noncontig tensors aren't tested, and don't seem common in practice
    ਜੇ (!ggml_is_contiguous(rms) ||
        !ggml_is_contiguous(mul) ||
        !ggml_is_contiguous(rope)) {
        ਵਾਪਸ ਝੂਠ;
    }

    // only norm/neox are handled in the shader
    ਜੇ (mode != GGML_ROPE_TYPE_NEOX && mode != GGML_ROPE_TYPE_ਸਾਧਾਰਨ) {
        ਵਾਪਸ ਝੂਠ;
    }

    // shared memory size for passing data from mul->rope
    ਜੇ (mul->ne[0] > 1024) {
        ਵਾਪਸ ਝੂਠ;
    }

    // conditions for pipeline creation
    ਜੇ (ਆਕਾਰ(vk_op_rms_norm_mul_rope_ਧੱਕੋ_constants) > ctx->device->properties.limits.max_ਧੱਕੋ_Constants_ਮਾਪ) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਵਾਪਸ ਸੱਚ;
}

uint32_t ggml_vk_fuse_multi_ਜੋੜੋ(ggml_backend_vk_context * ctx, ਅਚਲ ਬਣਤਰ ggml_cgraph * cgraph, ਪੂਰਨ ਗੰਢ_idx) {

    ਅਚਲ ggml_tensor *ਪਹਿਲਾ_ਗੰਢ = cgraph->nodes[ਗੰਢ_idx];
    ਜੇ (ਪਹਿਲਾ_ਗੰਢ->op != GGML_OP_ਜੋੜੋ) {
        ਵਾਪਸ 0;
    }

    ਜੇ (!ctx->device->multi_ਜੋੜੋ) {
        ਵਾਪਸ 0;
    }

    int32_t ਅੰਕ_adds = 1;
    ਜਦੋਂ (ਗੰਢ_idx + ਅੰਕ_adds < cgraph->n_nodes &&
           cgraph->nodes[ਗੰਢ_idx + ਅੰਕ_adds]->op == GGML_OP_ਜੋੜੋ &&
           ਅੰਕ_adds < MAX_FUSED_ADDS) {
        ਅੰਕ_adds++;
    }

    // The shader currently requires same shapes (but different strides are allowed),
    // everything f32, and no misalignment
    ਲਈ (int32_t i = 0; i < ਅੰਕ_adds; ++i) {
        ਅਚਲ ggml_tensor *ਅਗਲਾ_ਗੰਢ = cgraph->nodes[ਗੰਢ_idx + i];
        ਜੇ (!ggml_are_same_shape(ਪਹਿਲਾ_ਗੰਢ, ਅਗਲਾ_ਗੰਢ->src[0]) ||
            !ggml_are_same_shape(ਪਹਿਲਾ_ਗੰਢ, ਅਗਲਾ_ਗੰਢ->src[1]) ||
            ਅਗਲਾ_ਗੰਢ->type != GGML_TYPE_F32 ||
            ਅਗਲਾ_ਗੰਢ->src[0]->type != GGML_TYPE_F32 ||
            ਅਗਲਾ_ਗੰਢ->src[1]->type != GGML_TYPE_F32 ||
            get_misalign_ਬਾਈਟਾਂ(ctx, ਅਗਲਾ_ਗੰਢ) ||
            get_misalign_ਬਾਈਟਾਂ(ctx, ਅਗਲਾ_ਗੰਢ->src[0]) ||
            get_misalign_ਬਾਈਟਾਂ(ctx, ਅਗਲਾ_ਗੰਢ->src[1])) {
            ਅੰਕ_adds = i;
        }
    }

    // Verify we can fuse these
    ggml_op adds[MAX_FUSED_ADDS];
    ਲਈ (int32_t i = 0; i < ਅੰਕ_adds; ++i) {
        adds[i] = GGML_OP_ਜੋੜੋ;
    }

    // decrease num_adds if they can't all be fused
    ਜਦੋਂ (ਅੰਕ_adds > 1 && !ggml_can_fuse(cgraph, ਗੰਢ_idx, adds, ਅੰਕ_adds)) {
        ਅੰਕ_adds--;
    }

    // a single add is not "fused", so just return zero
    ਜੇ (ਅੰਕ_adds == 1) {
        ਵਾਪਸ 0;
    }
    ਵਾਪਸ ਅੰਕ_adds;
}

ਸਥਿਰ int32_t ਲੱਭੋ_ਪਹਿਲਾ_ਸੈੱਟ(uint32_t x) {
    int32_t ret = 0;
    ਜੇ (!x) {
        ਵਾਪਸ -1;
    }
    ਜਦੋਂ (!(x & 1)) {
        x >>= 1;
        ret++;
    }
    ਵਾਪਸ ret;
}

ਸਥਿਰ ggml_ਸਥਿਤੀ ggml_backend_vk_ਗ੍ਰਾਫ_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    VK_LOG_DEBUG("ggml_backend_vk_graph_compute(" << cgraph->n_nodes << " nodes)");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;

    ctx->device->diag_cgraph = ਖਾਲੀ_ਸੂਚਕ;
    ctx->device->diag_prev_ਸ਼ੁਰੂ = -1;
    ctx->device->diag_prev_ਸਿਰਾ = -1;

    ਜੇ (vk_instance.debug_utils_support) {
        ctx->device->debug_cmdbuf_idx = 0;
    }

    // queue scope, so it encloses every submit this evaluation makes.
    // closed when the function returns
    ggml_vk_debug_label ਕਤਾਰ_dbg(ctx->device->compute_ਕਤਾਰ->handle.get(), "ggml_backend_vk_graph_compute");

    ctx->prealloc_ਮਾਪ_ਜੋੜੋ_rms_partials_ਵਿੱਥ = 0;
    ctx->do_ਜੋੜੋ_rms_partials = ਝੂਠ;
    ctx->do_ਜੋੜੋ_rms_partials_ਵਿੱਥ_calculation = ਝੂਠ;

    ਪੂਰਨ ਆਖਰੀ_ਗੰਢ = cgraph->n_nodes - 1;

    // If the last op in the cgraph isn't backend GPU, the command buffer doesn't get closed properly
    ਜਦੋਂ (ਆਖਰੀ_ਗੰਢ > 0 && (ggml_vk_is_ਖਾਲੀ_ਥਾਵਾਂ(cgraph->nodes[ਆਖਰੀ_ਗੰਢ]) || ((cgraph->nodes[ਆਖਰੀ_ਗੰਢ]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0))) {
        ਆਖਰੀ_ਗੰਢ -= 1;
    }

    // Reserve tensor context space for all nodes
    ctx->tensor_ctxs.resize(cgraph->n_nodes);

    ਤਾਰਕਿਕ ਪਹਿਲਾ_ਗੰਢ_ਵਿੱਚ_batch = ਸੱਚ; // true if next node will be first node in a batch
    ਪੂਰਨ submit_ਗੰਢ_idx = 0; // index to first node in a batch

    ggml_vk_submit_transfer_ctx(ctx);

    vk_context compute_ctx;
    ਜੇ (vk_perf_logger_enabled) {
        // allocate/resize the query pool
        ਜੇ (ctx->ਅੰਕ_queries < cgraph->n_nodes + 1) {
            ਜੇ (ctx->query_pool) {
                ctx->device->device.destroyQueryPool(ctx->query_pool);
            }
            vk::QueryPoolCreateInfo query_create_info;
            query_create_info.queryType = vk::QueryType::eTimestamp;
            query_create_info.query_ਗਿਣਤੀ = cgraph->n_nodes + 100;
            ctx->query_pool = ctx->device->device.createQueryPool(query_create_info);
            ctx->ਅੰਕ_queries = query_create_info.query_ਗਿਣਤੀ;
            ctx->query_fusion_ਨਾਂ.resize(ctx->ਅੰਕ_queries);
            ctx->query_fusion_ਗੰਢ_ਗਿਣਤੀ.resize(ctx->ਅੰਕ_queries);
            ctx->query_nodes.resize(ctx->ਅੰਕ_queries);
            ctx->query_ਗੰਢ_idx.resize(ctx->ਅੰਕ_queries);
        }

        ctx->device->device.resetQueryPool(ctx->query_pool, 0, cgraph->n_nodes+1);
        std::fill(ctx->query_fusion_ਨਾਂ.begin(), ctx->query_fusion_ਨਾਂ.end(), ਖਾਲੀ_ਸੂਚਕ);
        std::fill(ctx->query_fusion_ਗੰਢ_ਗਿਣਤੀ.begin(), ctx->query_fusion_ਗੰਢ_ਗਿਣਤੀ.end(), 0);
        std::fill(ctx->query_nodes.begin(), ctx->query_nodes.end(), ਖਾਲੀ_ਸੂਚਕ);
        std::fill(ctx->query_ਗੰਢ_idx.begin(), ctx->query_ਗੰਢ_idx.end(), 0);

        GGML_ASSERT(ctx->compute_ctx.expired());
        compute_ctx = ggml_vk_get_compute_ctx(ctx);
        ctx->query_idx = 0;
        compute_ctx->s->ਬਫਰ->buf.writeTimestamp(vk::Pipeline_Stage_Flag_ਬਿੱਟ::eAllCommands, ctx->query_pool, ctx->query_idx++);
        ggml_vk_sync_buffers(ctx, compute_ctx);
    }

    ctx->prealloc_y_ਆਖਰੀ_pipeline_ਵਰਤਿਆ = ਖਾਲੀ_ਸੂਚਕ;
    ctx->prealloc_y_ਆਖਰੀ_tensor_ਵਰਤਿਆ = ਖਾਲੀ_ਸੂਚਕ;
    ctx->prealloc_y_ਆਖਰੀ_k_padded = ਝੂਠ;

    ਜੇ (ctx->prealloc_ਮਾਪ_ਜੋੜੋ_rms_partials) {
        ggml_vk_preallocate_buffers(ctx, ਖਾਲੀ_ਸੂਚਕ);
        compute_ctx = ggml_vk_get_compute_ctx(ctx);
        // initialize partial sums to zero.
        ggml_vk_ਬਫਰ_memset_async(compute_ctx, ctx->prealloc_ਜੋੜੋ_rms_partials, 0, 0, ctx->prealloc_ਮਾਪ_ਜੋੜੋ_rms_partials);
        ggml_vk_sync_buffers(ctx, compute_ctx);
    }

    // Submit after enough work has accumulated, to overlap CPU cmdbuffer generation with GPU execution.
    // Estimate the amount of compute work using flops, and submit every 200 GFLOP
    // (and scaled down based on total graph flops, so smaller models submit earlier).
    // Also submit at least every 100 nodes, in case there are workloads without heavy compute.
    uint32_t submitted_nodes = 0;
    uint32_t submit_ਗਿਣਤੀ = 0;
    uint64_t batch_flops = 0;
    uint64_t ਕੁੱਲ_flops = 0;
    uint64_t flops_cap = 200'000'000'000ULL;

    // On weaker AMD GPUs larger submissions can hit a driver timeout, submit more often to avoid this
    ਜੇ (ctx->device->vendor_id == VK_VENDOR_ID_AMD && ctx->device->shader_core_ਗਿਣਤੀ > 0) {
        ਜੇ (ctx->device->architecture == AMD_GCN && ctx->device->shader_core_ਗਿਣਤੀ < 32) {
            flops_cap = 500'000'000ULL * ctx->device->shader_core_ਗਿਣਤੀ;
        } ਨਹੀਂਤਾਂ ਜੇ (ctx->device->architecture != AMD_GCN && ctx->device->shader_core_ਗਿਣਤੀ < 24) {
            flops_cap = 2'000'000'000ULL * ctx->device->shader_core_ਗਿਣਤੀ;
        }
    }
    uint64_t flops_ਪ੍ਰਤੀ_submit = std::min(flops_cap, ctx->ਆਖਰੀ_ਕੁੱਲ_flops / 40u);

    ਸਵੈਚਾਲਿਤ ਅਚਲ submit_after = [&](ਪੂਰਨ start, ਪੂਰਨ end) {
        ਜੇ (ctx->device->serialize_submissions) {
            ਕੋਸ਼ਿਸ਼ {
                ਸਵੈਚਾਲਿਤ res = ctx->device->device.waitForFences({ ctx->fence }, ਸੱਚ, UINT64_MAX);
                ਜੇ (res != vk::ਨਤੀਜਾ_2::eSuccess) {
                    GGML_LOG_ERROR("ggml_vulkan: waitForFences error during serialized submission\n");
                    ਸੁੱਟੋ vk::SystemError(vk::make_error_code(res), "ggml_vulkan: waitForFences during serialized submission");
                }
            } ਫੜੋ (vk::Device_ਗੁਆਚਾ_Error &) {
                ggml_vk_print_device_fault_info(ctx->device);
                GGML_LOG_ERROR("ggml_vulkan: device lost on %s waiting for submission (nodes %d to %d):\n",
                        ctx->device->name.c_str(), start, end);
                ggml_vk_print_ਗੰਢ_list(cgraph, start, end);
                ਸੁੱਟੋ;
            }
            ctx->device->device.resetFences({ ctx->fence });
            ctx->submit_pending = ਝੂਠ;
            ctx->device->diag_cgraph = cgraph;
            ctx->device->diag_prev_ਸ਼ੁਰੂ = start;
            ctx->device->diag_prev_ਸਿਰਾ = end;
        }
        ਪਹਿਲਾ_ਗੰਢ_ਵਿੱਚ_batch = ਸੱਚ;
        submitted_nodes = 0;
        batch_flops = 0;
        ਜੇ (submit_ਗਿਣਤੀ < 3) {
            flops_ਪ੍ਰਤੀ_submit *= 2;
        }
        submit_ਗਿਣਤੀ++;
    };

    ਲਈ (ਪੂਰਨ i = 0; i < cgraph->n_nodes; i++) {
        ਜੇ (ਪਹਿਲਾ_ਗੰਢ_ਵਿੱਚ_batch) {
            submit_ਗੰਢ_idx = i;
        }

        {
            ਸਵੈਚਾਲਿਤ ਗੰਢ_flops = ggml_vk_get_ਗੰਢ_flops(cgraph->nodes[i]);
            ਕੁੱਲ_flops += ਗੰਢ_flops;

            // Flush the current batch before recording a node that would push it over the flop threshold
            ਜੇ (flops_ਪ੍ਰਤੀ_submit != 0 && submitted_nodes > 0 && batch_flops + ਗੰਢ_flops >= flops_ਪ੍ਰਤੀ_submit) {
                vk_context flush_ctx = ggml_vk_get_compute_ctx(ctx);
                ggml_vk_ctx_ਸਿਰਾ(flush_ctx);
                flush_ctx->exit_tensor_idx = -1;
                ctx->compute_ctx.reset();
                ggml_vk_compute_forward(ctx, cgraph, cgraph->nodes[submit_ਗੰਢ_idx], submit_ਗੰਢ_idx, ਝੂਠ);
                submit_after(submit_ਗੰਢ_idx, i - 1);
                submit_ਗੰਢ_idx = i;
            }

            batch_flops += ਗੰਢ_flops;
        }

        // op_srcs_fused_elementwise indicates whether an op's srcs all contribute to
        // the fused result in an elementwise-way. This affects whether the memory for
        // the src is allowed to overlap the memory for the destination.
        // The array is sized to handle the largest fusion (asserted later).
        ਤਾਰਕਿਕ op_srcs_fused_elementwise[13];

        ctx->fused_topk_moe_ਬਹੁਲਕ = TOPK_MOE_ਗਿਣਤੀ;
        ctx->fused_topk_moe_scale = ਝੂਠ;
        ctx->fused_topk_qsa = ਝੂਠ;
        ctx->fused_hc_post_gate = ਝੂਠ;
        ctx->fused_rms_norm_ਬਹੁਲਕ = RMS_NORM_ਗਿਣਤੀ;
        ਅਚਲ ਅੱਖਰ *fusion_string {};
        ਜੇ (!ctx->device->disable_fusion) {
            uint32_t ਅੰਕ_adds = ggml_vk_fuse_multi_ਜੋੜੋ(ctx, cgraph, i);
            ਜੇ (ਅੰਕ_adds) {
                ctx->ਅੰਕ_additional_fused_ops = ਅੰਕ_adds - 1;
                fusion_string = "MULTI_ADD";
                std::fill_n(op_srcs_fused_elementwise, ctx->ਅੰਕ_additional_fused_ops + 1, ਸੱਚ);
            } ਨਹੀਂਤਾਂ ਜੇ (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_MUL_MAT, GGML_OP_ਜੋੜੋ, GGML_OP_ਜੋੜੋ })) {
                ctx->ਅੰਕ_additional_fused_ops = 2;
                fusion_string = "MUL_MAT_ADD_ADD";
                op_srcs_fused_elementwise[0] = ਝੂਠ;
                op_srcs_fused_elementwise[1] = ਸੱਚ;
                op_srcs_fused_elementwise[2] = ਸੱਚ;
            } ਨਹੀਂਤਾਂ ਜੇ (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_MUL_MAT, GGML_OP_ਜੋੜੋ })) {
                ctx->ਅੰਕ_additional_fused_ops = 1;
                fusion_string = "MUL_MAT_ADD";
                op_srcs_fused_elementwise[0] = ਝੂਠ;
                op_srcs_fused_elementwise[1] = ਸੱਚ;
            } ਨਹੀਂਤਾਂ ਜੇ (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_MUL_MAT_ID, GGML_OP_ਜੋੜੋ_ID, GGML_OP_MUL })) {
                ctx->ਅੰਕ_additional_fused_ops = 2;
                fusion_string = "MUL_MAT_ID_ADD_ID_MUL";
                op_srcs_fused_elementwise[0] = ਝੂਠ;
                op_srcs_fused_elementwise[1] = ਸੱਚ;
                op_srcs_fused_elementwise[2] = ਸੱਚ;
            } ਨਹੀਂਤਾਂ ਜੇ (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_MUL_MAT_ID, GGML_OP_ਜੋੜੋ_ID })) {
                ctx->ਅੰਕ_additional_fused_ops = 1;
                fusion_string = "MUL_MAT_ID_ADD_ID";
                op_srcs_fused_elementwise[0] = ਝੂਠ;
                op_srcs_fused_elementwise[1] = ਸੱਚ;
            } ਨਹੀਂਤਾਂ ਜੇ (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_MUL_MAT_ID, GGML_OP_MUL })) {
                ctx->ਅੰਕ_additional_fused_ops = 1;
                fusion_string = "MUL_MAT_ID_MUL";
                op_srcs_fused_elementwise[0] = ਝੂਠ;
                op_srcs_fused_elementwise[1] = ਸੱਚ;
            } ਨਹੀਂਤਾਂ ਜੇ (ggml_can_fuse_subgraph(cgraph, i, rms_norm_mul_rope_view_ਸੈੱਟ_ਪੰਕਤੀਆਂ_ਨਮੂਨਾ, { i + 4 }) &&
                       ggml_check_ਕਿਨਾਰੇ(cgraph, i, rms_norm_mul_rope_view_ਸੈੱਟ_ਪੰਕਤੀਆਂ_ਕਿਨਾਰੇ) &&
                       ggml_vk_can_fuse_rms_norm_mul_rope(ctx, cgraph, i) &&
                       ggml_vk_can_fuse_rope_ਸੈੱਟ_ਪੰਕਤੀਆਂ(ctx, cgraph, i + 2)) {
                ctx->ਅੰਕ_additional_fused_ops = 4;
                ctx->fused_rms_norm_ਬਹੁਲਕ = RMS_NORM_MUL_ROPE_VIEW_ਸੈੱਟ_ਪੰਕਤੀਆਂ;
                fusion_string = "RMS_NORM_MUL_ROPE_VIEW_SET_ROWS";
                op_srcs_fused_elementwise[0] = ਝੂਠ;
                op_srcs_fused_elementwise[1] = ਝੂਠ;
                op_srcs_fused_elementwise[2] = ਝੂਠ;
                op_srcs_fused_elementwise[3] = ਝੂਠ;
                op_srcs_fused_elementwise[4] = ਝੂਠ;
            } ਨਹੀਂਤਾਂ ਜੇ (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE }) &&
                       ggml_vk_can_fuse_rms_norm_mul_rope(ctx, cgraph, i)) {
                ctx->ਅੰਕ_additional_fused_ops = 2;
                ctx->fused_rms_norm_ਬਹੁਲਕ = RMS_NORM_MUL_ROPE;
                fusion_string = "RMS_NORM_MUL_ROPE";
                // rope is approximately elementwise - whole rows are done by a single workgroup and it's row-wise
                op_srcs_fused_elementwise[0] = ਝੂਠ;
                op_srcs_fused_elementwise[1] = ਸੱਚ;
                op_srcs_fused_elementwise[2] = ਸੱਚ;
            } ਨਹੀਂਤਾਂ ਜੇ (ggml_can_fuse_subgraph(cgraph, i, hc_post_gate_ਨਮੂਨਾ, { i + 3 }) &&
                       ggml_check_ਕਿਨਾਰੇ(cgraph, i, hc_post_gate_ਕਿਨਾਰੇ) &&
                       ggml_vk_can_fuse_hc_post_gate(cgraph, i)) {
                ctx->ਅੰਕ_additional_fused_ops = hc_post_gate_ਨਮੂਨਾ.size() - 1;
                ctx->fused_hc_post_gate = ਸੱਚ;
                fusion_string = "HC_POST_GATE";
                std::fill_n(op_srcs_fused_elementwise, ctx->ਅੰਕ_additional_fused_ops + 1, ਝੂਠ);
            } ਨਹੀਂਤਾਂ ਜੇ (ggml_vk_can_fuse(ctx, cgraph, i, rms_norm_mul_ਜੋੜੋ_mul_ਨਮੂਨਾ)) {
                ctx->ਅੰਕ_additional_fused_ops = 3;
                ctx->fused_rms_norm_ਬਹੁਲਕ = RMS_NORM_MUL_ਜੋੜੋ_MUL;
                fusion_string = "RMS_NORM_MUL_ADD_MUL";
                std::fill_n(op_srcs_fused_elementwise, 4, ਸੱਚ);
            } ਨਹੀਂਤਾਂ ਜੇ (ggml_vk_can_fuse(ctx, cgraph, i, rms_norm_mul_ਜੋੜੋ_ਨਮੂਨਾ)) {
                ctx->ਅੰਕ_additional_fused_ops = 2;
                ctx->fused_rms_norm_ਬਹੁਲਕ = RMS_NORM_MUL_ਜੋੜੋ;
                fusion_string = "RMS_NORM_MUL_ADD";
                std::fill_n(op_srcs_fused_elementwise, 3, ਸੱਚ);
            } ਨਹੀਂਤਾਂ ਜੇ (ggml_can_fuse_subgraph(cgraph, i, rms_norm_view_ਸੈੱਟ_ਪੰਕਤੀਆਂ_ਨਮੂਨਾ, { i + 2 }) &&
                       ggml_check_ਕਿਨਾਰੇ(cgraph, i, rms_norm_view_ਸੈੱਟ_ਪੰਕਤੀਆਂ_ਕਿਨਾਰੇ) &&
                       ggml_vk_can_fuse_rms_norm_ਸੈੱਟ_ਪੰਕਤੀਆਂ(ctx, cgraph, i)) {
                ctx->ਅੰਕ_additional_fused_ops = 2;
                ctx->fused_rms_norm_ਬਹੁਲਕ = RMS_NORM_VIEW_ਸੈੱਟ_ਪੰਕਤੀਆਂ;
                fusion_string = "RMS_NORM_VIEW_SET_ROWS";
                std::fill_n(op_srcs_fused_elementwise, 3, ਝੂਠ);
            } ਨਹੀਂਤਾਂ ਜੇ (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
                ctx->ਅੰਕ_additional_fused_ops = 1;
                ctx->fused_rms_norm_ਬਹੁਲਕ = RMS_NORM_MUL;
                fusion_string = "RMS_NORM_MUL";
                // rms_norm is not elementwise, but whole rows must be consumed and the scale factor computed before
                // they are overwritten, and one workgroup per row. So close enough.
                op_srcs_fused_elementwise[0] = ਸੱਚ;
                op_srcs_fused_elementwise[1] = ਸੱਚ;
            } ਨਹੀਂਤਾਂ ਜੇ (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL })) {
                ctx->ਅੰਕ_additional_fused_ops = 1;
                ਚੋਣ (ggml_get_unary_op(cgraph->nodes[i])) {
                    ਹਾਲਤ GGML_UNARY_OP_GELU:     fusion_string = "GELU_MUL";     ਰੋਕੋ;
                    ਹਾਲਤ GGML_UNARY_OP_SIGMOID:  fusion_string = "SIGMOID_MUL";  ਰੋਕੋ;
                    ਹਾਲਤ GGML_UNARY_OP_SILU:     fusion_string = "SILU_MUL";     ਰੋਕੋ;
                    ਮੂਲ:                     fusion_string = "SOFTPLUS_MUL"; ਰੋਕੋ;
                }
                op_srcs_fused_elementwise[0] = ਸੱਚ;
                op_srcs_fused_elementwise[1] = ਸੱਚ;
            } ਨਹੀਂਤਾਂ ਜੇ (ggml_vk_can_fuse_ssm_conv(ctx, cgraph, i, 2)) {
                ctx->ਅੰਕ_additional_fused_ops = 2;
                fusion_string = "SSM_CONV_BIAS_SILU";
                // ssm_conv reads multiple input tokens per output, so it's not elementwise w.r.t. its srcs.
                // The downstream add and silu are elementwise on the conv output.
                op_srcs_fused_elementwise[0] = ਝੂਠ;
                op_srcs_fused_elementwise[1] = ਸੱਚ;
                op_srcs_fused_elementwise[2] = ਸੱਚ;
            } ਨਹੀਂਤਾਂ ਜੇ (ggml_vk_can_fuse_ssm_conv(ctx, cgraph, i, 1)) {
                ctx->ਅੰਕ_additional_fused_ops = 1;
                fusion_string = "SSM_CONV_SILU";
                op_srcs_fused_elementwise[0] = ਝੂਠ;
                op_srcs_fused_elementwise[1] = ਸੱਚ;
            } ਨਹੀਂਤਾਂ ਜੇ (ggml_can_fuse_subgraph(cgraph, i, rope_view_ਸੈੱਟ_ਪੰਕਤੀਆਂ_ਨਮੂਨਾ, { i + 2 }) &&
                       ggml_check_ਕਿਨਾਰੇ(cgraph, i, rope_view_ਸੈੱਟ_ਪੰਕਤੀਆਂ_ਕਿਨਾਰੇ) &&
                       ggml_vk_can_fuse_rope_ਸੈੱਟ_ਪੰਕਤੀਆਂ(ctx, cgraph, i)) {
                ctx->ਅੰਕ_additional_fused_ops = 2;
                fusion_string = "ROPE_VIEW_SET_ROWS";
                op_srcs_fused_elementwise[0] = ਝੂਠ;
                op_srcs_fused_elementwise[1] = ਝੂਠ;
                op_srcs_fused_elementwise[2] = ਝੂਠ;
            } ਨਹੀਂਤਾਂ ਜੇ (ggml_vk_can_fuse_snake(ctx, cgraph, i)) {
                ctx->ਅੰਕ_additional_fused_ops = 4;
                fusion_string = "SNAKE";
                // elementwise=true: snake.comp is safe under exact aliasing because each
                // thread reads data_x[idx] into a register before writing data_d[idx]
                // with a data dependency on that register. The overlap check still
                // rejects partial overlaps (different base or size).
                std::fill_n(op_srcs_fused_elementwise, 5, ਸੱਚ);
            } ਨਹੀਂਤਾਂ ਜੇ (ggml_vk_can_fuse_topk_qsa(ctx, cgraph, i)) {
                ctx->ਅੰਕ_additional_fused_ops = topk_qsa_ਨਮੂਨਾ.size() - 1;
                ctx->fused_topk_qsa = ਸੱਚ;
                fusion_string = "TOPK_QSA";
                std::fill_n(op_srcs_fused_elementwise, ctx->ਅੰਕ_additional_fused_ops + 1, ਝੂਠ);
            } ਨਹੀਂਤਾਂ ਜੇ (ggml_can_fuse_subgraph(cgraph, i, topk_moe_early_softmax_norm, { i + 3, i + 9 }) &&
                       ggml_check_ਕਿਨਾਰੇ(cgraph, i, topk_moe_early_softmax_norm_ਕਿਨਾਰੇ) &&
                       ggml_vk_can_fuse_topk_moe(ctx, cgraph, i, TOPK_MOE_EARLY_SOFTMAX_NORM)) {
                ctx->ਅੰਕ_additional_fused_ops = topk_moe_early_softmax_norm.size() - 1;
                // view of argsort writes to memory
                ctx->fused_ops_write_ਨਕਾਬ |= 1 << 3;
                ctx->fused_topk_moe_ਬਹੁਲਕ = TOPK_MOE_EARLY_SOFTMAX_NORM;
                fusion_string = "TOPK_MOE_EARLY_SOFTMAX_NORM";
                std::fill_n(op_srcs_fused_elementwise, ctx->ਅੰਕ_additional_fused_ops + 1, ਝੂਠ);
            } ਨਹੀਂਤਾਂ ਜੇ (ggml_can_fuse_subgraph(cgraph, i, topk_moe_sigmoid_norm_bias, { i + 4, i + 10 }) &&
                       ggml_check_ਕਿਨਾਰੇ(cgraph, i, topk_moe_sigmoid_norm_bias_ਕਿਨਾਰੇ) &&
                       ggml_vk_can_fuse_topk_moe(ctx, cgraph, i, TOPK_MOE_SIGMOID_NORM_BIAS)) {
                ctx->ਅੰਕ_additional_fused_ops = topk_moe_sigmoid_norm_bias.size() - 1;
                // view of argsort writes to memory
                ctx->fused_ops_write_ਨਕਾਬ |= 1 << 4;
                ctx->fused_topk_moe_ਬਹੁਲਕ = TOPK_MOE_SIGMOID_NORM_BIAS;
                fusion_string = "TOPK_MOE_SIGMOID_NORM_BIAS";
                std::fill_n(op_srcs_fused_elementwise, ctx->ਅੰਕ_additional_fused_ops + 1, ਝੂਠ);
            } ਨਹੀਂਤਾਂ ਜੇ (ggml_can_fuse_subgraph(cgraph, i, topk_moe_sqrt_softplus_norm_bias, { i + 5, i + 11 }) &&
                       ggml_check_ਕਿਨਾਰੇ(cgraph, i, topk_moe_sqrt_softplus_norm_bias_ਕਿਨਾਰੇ) &&
                       ggml_vk_can_fuse_topk_moe(ctx, cgraph, i, TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS)) {
                ctx->ਅੰਕ_additional_fused_ops = topk_moe_sqrt_softplus_norm_bias.size() - 1;
                // view of argsort writes to memory
                ctx->fused_ops_write_ਨਕਾਬ |= 1 << 5;
                ctx->fused_topk_moe_ਬਹੁਲਕ = TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS;
                fusion_string = "TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS";
                std::fill_n(op_srcs_fused_elementwise, ctx->ਅੰਕ_additional_fused_ops + 1, ਝੂਠ);
            } ਨਹੀਂਤਾਂ ਜੇ (ggml_can_fuse_subgraph(cgraph, i, topk_moe_early_softmax, { i + 3, i + 4 }) &&
                       ggml_check_ਕਿਨਾਰੇ(cgraph, i, topk_moe_early_softmax_ਕਿਨਾਰੇ) &&
                       ggml_vk_can_fuse_topk_moe(ctx, cgraph, i, TOPK_MOE_EARLY_SOFTMAX)) {
                ctx->ਅੰਕ_additional_fused_ops = topk_moe_early_softmax.size() - 1;
                // view of argsort writes to memory
                ctx->fused_ops_write_ਨਕਾਬ |= 1 << 3;
                ctx->fused_topk_moe_ਬਹੁਲਕ = TOPK_MOE_EARLY_SOFTMAX;
                fusion_string = "TOPK_MOE_EARLY_SOFTMAX";
                std::fill_n(op_srcs_fused_elementwise, ctx->ਅੰਕ_additional_fused_ops + 1, ਝੂਠ);
            } ਨਹੀਂਤਾਂ ਜੇ (ggml_can_fuse_subgraph(cgraph, i, topk_moe_late_softmax, { i + 1, i + 5 }) &&
                       ggml_check_ਕਿਨਾਰੇ(cgraph, i, topk_moe_late_softmax_ਕਿਨਾਰੇ) &&
                       ggml_vk_can_fuse_topk_moe(ctx, cgraph, i, TOPK_MOE_LATE_SOFTMAX)) {
                ctx->ਅੰਕ_additional_fused_ops = topk_moe_late_softmax.size() - 1;
                // view of argsort writes to memory
                ctx->fused_ops_write_ਨਕਾਬ |= 1 << 1;
                ctx->fused_topk_moe_ਬਹੁਲਕ = TOPK_MOE_LATE_SOFTMAX;
                fusion_string = "TOPK_MOE_LATE_SOFTMAX";
                std::fill_n(op_srcs_fused_elementwise, ctx->ਅੰਕ_additional_fused_ops + 1, ਝੂਠ);
            }
            ਜੇ (ctx->fused_topk_moe_ਬਹੁਲਕ != TOPK_MOE_ਗਿਣਤੀ) {
                // Look for an additional scale op to fuse - occurs in deepseek2 and nemotron3 nano.
                ਜੇ (ggml_can_fuse_subgraph(cgraph, i + ctx->ਅੰਕ_additional_fused_ops - 1, { GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE }, { i + ctx->ਅੰਕ_additional_fused_ops + 1 }) ||
                    ggml_can_fuse_subgraph(cgraph, i + ctx->ਅੰਕ_additional_fused_ops, { GGML_OP_GET_ਪੰਕਤੀਆਂ, GGML_OP_SCALE }, { i + ctx->ਅੰਕ_additional_fused_ops + 1 })) {
                    ctx->fused_topk_moe_scale = ਸੱਚ;
                    ctx->ਅੰਕ_additional_fused_ops++;
                    op_srcs_fused_elementwise[ctx->ਅੰਕ_additional_fused_ops] = ਝੂਠ;
                }
            }
        }
        GGML_ASSERT(ctx->ਅੰਕ_additional_fused_ops < (ਪੂਰਨ)(ਆਕਾਰ(op_srcs_fused_elementwise) / ਆਕਾਰ(op_srcs_fused_elementwise[0])));
        ctx->fused_ops_write_ਨਕਾਬ |= 1 << ctx->ਅੰਕ_additional_fused_ops;

        // Check whether fusion would overwrite src operands while they're still in use.
        // If so, disable fusion.
        ਜੇ (ctx->ਅੰਕ_additional_fused_ops) {
            // There are up to two output nodes - topk_moe has two.
            uint32_t ਬਿੱਟ = ctx->fused_ops_write_ਨਕਾਬ & ~(1 << ctx->ਅੰਕ_additional_fused_ops);
            ggml_tensor *output_nodes[2] {};
            output_nodes[0] = cgraph->nodes[i + ctx->ਅੰਕ_additional_fused_ops];
            ਜੇ (ਬਿੱਟ) {
                ਪੂਰਨ output_idx = ਲੱਭੋ_ਪਹਿਲਾ_ਸੈੱਟ(ਬਿੱਟ);
                GGML_ASSERT(ਬਿੱਟ == (1u << output_idx));
                output_nodes[1] = cgraph->nodes[i + output_idx];
            }

            ਤਾਰਕਿਕ need_disable = ਝੂਠ;

            ਲਈ (ਪੂਰਨ j = 0; j < 2; ++j) {
                ggml_tensor *dst = output_nodes[j];
                ਜੇ (!dst) {
                    ਜਾਰੀ;
                }
                // Loop over all srcs of all nodes in the fusion. If the src overlaps
                // the destination and the src is not an intermediate node that's being
                // elided, then disable fusion.
                ਲਈ (ਪੂਰਨ k = 0; k <= ctx->ਅੰਕ_additional_fused_ops; ++k) {
                    ਲਈ (uint32_t s = 0; s < GGML_MAX_SRC; ++s) {
                        ggml_tensor *src = cgraph->nodes[i + k]->src[s];
                        ਜੇ (!src || src->op == GGML_OP_NONE) {
                            ਜਾਰੀ;
                        }
                        ਜੇ (ggml_vk_tensors_overlap(src, dst, op_srcs_fused_elementwise[k])) {
                            ਤਾਰਕਿਕ ਮਿਲਿਆ = ਝੂਠ;
                            ਲਈ (ਪੂਰਨ n = 0; n < k; ++n) {
                                ਜੇ (cgraph->nodes[i + n] == src) {
                                    ਮਿਲਿਆ = ਸੱਚ;
                                    ਰੋਕੋ;
                                }
                            }
                            ਜੇ (!ਮਿਲਿਆ) {
                                need_disable = ਸੱਚ;
                            }
                        }
                    }
                }
            }
            ਜੇ (need_disable) {
                ctx->ਅੰਕ_additional_fused_ops = 0;
                ctx->fused_ops_write_ਨਕਾਬ = 1;
                ctx->fused_topk_moe_ਬਹੁਲਕ = TOPK_MOE_ਗਿਣਤੀ;
                ctx->fused_topk_moe_scale = ਝੂਠ;
                ctx->fused_topk_qsa = ਝੂਠ;
                ctx->fused_hc_post_gate = ਝੂਠ;
                ctx->fused_rms_norm_ਬਹੁਲਕ = RMS_NORM_ਗਿਣਤੀ;
                fusion_string = ਖਾਲੀ_ਸੂਚਕ;
            }
        }

        // Signal the almost_ready fence when the graph is mostly complete (< 20% remaining)
        ਤਾਰਕਿਕ almost_ready = (cgraph->n_nodes - i) < cgraph->n_nodes / 5;
        ਤਾਰਕਿਕ submit = (submitted_nodes >= ctx->device->max_nodes_ਪ੍ਰਤੀ_submit) ||
                      (flops_ਪ੍ਰਤੀ_submit != 0 && batch_flops >= flops_ਪ੍ਰਤੀ_submit) ||
                      (i + ctx->ਅੰਕ_additional_fused_ops >= ਆਖਰੀ_ਗੰਢ) ||
                      (almost_ready && !ctx->almost_ready_fence_pending);

        ਤਾਰਕਿਕ enqueued = ggml_vk_build_ਗ੍ਰਾਫ(ctx, cgraph, i, cgraph->nodes[submit_ਗੰਢ_idx], submit_ਗੰਢ_idx, i + ctx->ਅੰਕ_additional_fused_ops >= ਆਖਰੀ_ਗੰਢ, almost_ready, submit);

        ਜੇ (vk_perf_logger_enabled && enqueued) {
            compute_ctx = ggml_vk_get_compute_ctx(ctx);
            ਜੇ (!vk_perf_logger_concurrent) {
                // track a single node/fusion for the current query
                ctx->query_nodes[ctx->query_idx] = cgraph->nodes[i];
                ctx->query_fusion_ਨਾਂ[ctx->query_idx] = fusion_string;
                compute_ctx->s->ਬਫਰ->buf.writeTimestamp(vk::Pipeline_Stage_Flag_ਬਿੱਟ::eAllCommands, ctx->query_pool, ctx->query_idx++);
                ggml_vk_sync_buffers(ctx, compute_ctx);
            } ਨਹੀਂਤਾਂ {
                // track a fusion string and number of fused ops for the current node_idx
                ctx->query_fusion_ਨਾਂ[i] = fusion_string;
                ctx->query_fusion_ਗੰਢ_ਗਿਣਤੀ[i] = ctx->ਅੰਕ_additional_fused_ops;
            }
        }

        ਜੇ (enqueued) {
            ++submitted_nodes;

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ GGML_VULKAN_CHECK_RESULTS
            ਜੇ (ਪਹਿਲਾ_ਗੰਢ_ਵਿੱਚ_batch) {
                ਪਹਿਲਾ_ਗੰਢ_ਵਿੱਚ_batch = ਝੂਠ;
            }
#ਜੇ_ਅੰਤ
        }

        ਜੇ (submit && enqueued) {
            submit_after(submit_ਗੰਢ_idx, i + (ਪੂਰਨ)ctx->ਅੰਕ_additional_fused_ops);
        }
        i += ctx->ਅੰਕ_additional_fused_ops;
        ctx->ਅੰਕ_additional_fused_ops = 0;
        ctx->fused_ops_write_ਨਕਾਬ = 0;
    }

    ctx->ਆਖਰੀ_ਕੁੱਲ_flops = ਕੁੱਲ_flops;

    ਜੇ (vk_perf_logger_enabled) {
        // End the command buffer and submit/wait
        GGML_ASSERT(!ctx->compute_ctx.expired());
        compute_ctx = ctx->compute_ctx.ਤਾਲਾ();
        ggml_vk_ctx_ਸਿਰਾ(compute_ctx);

        ggml_vk_submit(compute_ctx, ctx->device->fence);
        VK_CHECK(ctx->device->device.waitForFences({ ctx->device->fence }, ਸੱਚ, UINT64_MAX), "GGML_VULKAN_PERF waitForFences", ctx->device);
        ctx->device->device.resetFences({ ctx->device->fence });
        ctx->compute_ctx.reset();

        // Get the results and pass them to the logger
        std::vector<uint64_t> timestamps(cgraph->n_nodes + 1);
        VK_CHECK(ctx->device->device.getQueryPoolResults(ctx->query_pool, 0, ctx->query_idx, (cgraph->n_nodes + 1)*ਆਕਾਰ(uint64_t), timestamps.ਅੰਕੜੇ(), ਆਕਾਰ(uint64_t), vk::Query_ਨਤੀਜਾ_Flag_ਬਿੱਟ::e64 | vk::Query_ਨਤੀਜਾ_Flag_ਬਿੱਟ::eWait), "get timestamp results", ctx->device);
        ਜੇ (!vk_perf_logger_concurrent) {
            // Log each op separately
            ਲਈ (ਪੂਰਨ i = 1; i < ctx->query_idx; i++) {
                ਸਵੈਚਾਲਿਤ ਗੰਢ = ctx->query_nodes[i];
                ਸਵੈਚਾਲਿਤ name = ctx->query_fusion_ਨਾਂ[i];
                ctx->perf_logger->log_timing(ਗੰਢ, name, uint64_t((timestamps[i] - timestamps[i-1]) * ctx->device->properties.limits.timestampPeriod));
            }
        } ਨਹੀਂਤਾਂ {
            // Log each group of nodes
            ਪੂਰਨ prev_ਗੰਢ_idx = 0;
            ਲਈ (ਪੂਰਨ i = 1; i < ctx->query_idx; i++) {
                ਸਵੈਚਾਲਿਤ cur_ਗੰਢ_idx = ctx->query_ਗੰਢ_idx[i];
                std::vector<ggml_tensor *> nodes;
                std::vector<ਅਚਲ ਅੱਖਰ *> ਨਾਂ;
                ਲਈ (ਪੂਰਨ ਗੰਢ_idx = prev_ਗੰਢ_idx; ਗੰਢ_idx < cur_ਗੰਢ_idx; ++ਗੰਢ_idx) {
                    ਜੇ (ggml_op_is_ਖਾਲੀ_ਥਾਵਾਂ(cgraph->nodes[ਗੰਢ_idx]->op)) {
                        ਜਾਰੀ;
                    }
                    nodes.push_back(cgraph->nodes[ਗੰਢ_idx]);
                    ਨਾਂ.push_back(ctx->query_fusion_ਨਾਂ[ਗੰਢ_idx]);
                    ਗੰਢ_idx += ctx->query_fusion_ਗੰਢ_ਗਿਣਤੀ[ਗੰਢ_idx];
                }
                prev_ਗੰਢ_idx = cur_ਗੰਢ_idx;
                ctx->perf_logger->log_timing(nodes, ਨਾਂ, uint64_t((timestamps[i] - timestamps[i-1]) * ctx->device->properties.limits.timestampPeriod));
            }
        }
        ctx->perf_logger->print_timings();
    }

    ਜੇ (!ctx->device->support_async) {
        ggml_vk_synchronize(ctx);
    }

    ਵਾਪਸ GGML_ਸਥਿਤੀ_SUCCESS;

    ਅਣਵਰਤਿਆ(backend);
}

ਖਾਲੀ ggml_vk_ਗ੍ਰਾਫ_optimize(ggml_backend_t backend, ਬਣਤਰ ggml_cgraph * graph, ਬਣਤਰ ggml_backend_ਗ੍ਰਾਫ_optimize_params * params)
{
    VK_LOG_DEBUG("ggml_vk_graph_optimize(" << graph->n_nodes << " nodes)");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;

    ਜੇ (ctx->device->disable_ਗ੍ਰਾਫ_optimize) {
        ਵਾਪਸ;
    }

    ਸਵੈਚਾਲਿਤ ਅਚਲ &is_ਖਾਲੀ_ਥਾਵਾਂ = [](ਅਚਲ ggml_tensor * ਗੰਢ) -> ਤਾਰਕਿਕ {
        ਵਾਪਸ ਗੰਢ->op == GGML_OP_NONE || ਗੰਢ->op == GGML_OP_RESHAPE || ਗੰਢ->op == GGML_OP_TRANSPOSE || ਗੰਢ->op == GGML_OP_VIEW || ਗੰਢ->op == GGML_OP_ਕ੍ਰਮ_ਬਦਲੋ;
    };

    ਸਵੈਚਾਲਿਤ ਅਚਲ &is_src_ਦਾ = [&is_ਖਾਲੀ_ਥਾਵਾਂ](ਅਚਲ ggml_tensor *dst, ਅਚਲ ggml_tensor *src) -> ਤਾਰਕਿਕ {
        ਸਵੈਚਾਲਿਤ ਅਚਲ &ਅਧਾਰ = [](ਅਚਲ ggml_tensor * tensor) {
            ਵਾਪਸ tensor->view_src ? tensor->view_src : tensor;
        };
        ਲਈ (uint32_t s = 0; s < GGML_MAX_SRC; ++s) {
            ਜੇ (dst->src[s] == src) {
                ਵਾਪਸ ਸੱਚ;
            }
            ਜੇ (is_ਖਾਲੀ_ਥਾਵਾਂ(dst) || is_ਖਾਲੀ_ਥਾਵਾਂ(src)) {
                ਜਾਰੀ;
            }
            // A source view of dst may read storage written through a different view by src.
            ਜੇ (dst->src[s] && ਅਧਾਰ(dst->src[s]) == ਅਧਾਰ(src)) {
                ਵਾਪਸ ਸੱਚ;
            }
            // Moving dst forward may overwrite storage still read through a view by src.
            ਜੇ (src->src[s] && ਅਧਾਰ(dst) == ਅਧਾਰ(src->src[s])) {
                ਵਾਪਸ ਸੱਚ;
            }
        }
        // implicit dependency if they view the same tensor
        ਜੇ (ਅਧਾਰ(dst) == ਅਧਾਰ(src)) {
            ਵਾਪਸ ਸੱਚ;
        }
        ਵਾਪਸ ਝੂਠ;
    };

    std::vector<ggml_tensor *> new_ਕ੍ਰਮ;
    std::vector<ਤਾਰਕਿਕ> ਵਰਤਿਆ(graph->n_nodes, ਝੂਠ);
    std::set<ggml_tensor *> ਵਰਤਿਆ_ਗੰਢ_ਸੈੱਟ;

    ਪੂਰਨ ਪਹਿਲਾ_ਅਣਵਰਤਿਆ = 0;

    // scheduled or zero-compute nodes in [lo, hi)
    ਸਵੈਚਾਲਿਤ ਅਚਲ &ਖਾਲੀ_ਥਾਵਾਂ_or_scheduled_between = [&](ਪੂਰਨ lo, ਪੂਰਨ hi) -> ਤਾਰਕਿਕ {
        ਲਈ (ਪੂਰਨ v = lo; v < hi; ++v) {
            ਜੇ (!ਵਰਤਿਆ[v] && !is_ਖਾਲੀ_ਥਾਵਾਂ(graph->nodes[v])) {
                ਵਾਪਸ ਝੂਠ;
            }
        }
        ਵਾਪਸ ਸੱਚ;
    };
    ਜਦੋਂ (ਪਹਿਲਾ_ਅਣਵਰਤਿਆ < graph->n_nodes) {
        std::vector<ਪੂਰਨ> current_ਸੈੱਟ;

        // Check for fusion patterns and avoid reordering them
        ਸਵੈਚਾਲਿਤ ਅਚਲ &ਮੇਲ_ਨਮੂਨਾ = [&](ਅਚਲ std::initializer_list<ggml_op> &pattern, ਪੂਰਨ start) -> ਤਾਰਕਿਕ {
            ਜੇ (start + (ਪੂਰਨ)pattern.size() <= graph->n_nodes) {
                ਤਾਰਕਿਕ is_ਨਮੂਨਾ = ਸੱਚ;
                ਲਈ (size_t j = 0; j < pattern.size(); ++j) {
                    ਜੇ (graph->nodes[start + j]->op != pattern.begin()[j] || ਵਰਤਿਆ[start + j]) {
                        is_ਨਮੂਨਾ = ਝੂਠ;
                    }
                }
                ਵਾਪਸ is_ਨਮੂਨਾ;
            }
            ਵਾਪਸ ਝੂਠ;
        };

        ਸਵੈਚਾਲਿਤ ਅਚਲ &keep_ਨਮੂਨਾ = [&](ਅਚਲ std::initializer_list<ggml_op> &pattern) -> ਤਾਰਕਿਕ {
            ਜੇ (ਮੇਲ_ਨਮੂਨਾ(pattern, ਪਹਿਲਾ_ਅਣਵਰਤਿਆ)) {
                ਲਈ (size_t j = 0; j < pattern.size(); ++j) {
                    new_ਕ੍ਰਮ.push_back(graph->nodes[ਪਹਿਲਾ_ਅਣਵਰਤਿਆ + j]);
                    ਵਰਤਿਆ_ਗੰਢ_ਸੈੱਟ.insert(graph->nodes[ਪਹਿਲਾ_ਅਣਵਰਤਿਆ + j]);
                    ਵਰਤਿਆ[ਪਹਿਲਾ_ਅਣਵਰਤਿਆ + j] = ਸੱਚ;
                }
                ਜਦੋਂ (ਪਹਿਲਾ_ਅਣਵਰਤਿਆ < graph->n_nodes && ਵਰਤਿਆ[ਪਹਿਲਾ_ਅਣਵਰਤਿਆ]) {
                    ਪਹਿਲਾ_ਅਣਵਰਤਿਆ++;
                }
                ਵਾਪਸ ਸੱਚ;
            }
            ਵਾਪਸ ਝੂਠ;
        };

        ਸਵੈਚਾਲਿਤ ਅਚਲ &ਜੋੜੋ_ਨਮੂਨਾ_alloc_deps = [&](ਅਚਲ std::initializer_list<ggml_op> &pattern, ਪੂਰਨ ਆਖਰੀ_ਗੰਢ) {
            // Keep external inputs alive through the fused output.
            std::set<ggml_tensor *> ਵੇਖਿਆ;
            ਲਈ (size_t j = 0; j < pattern.size(); ++j) {
                ggml_tensor * ਗੰਢ = graph->nodes[ਪਹਿਲਾ_ਅਣਵਰਤਿਆ + j];
                ਲਈ (uint32_t s = 0; s < GGML_MAX_SRC; ++s) {
                    ggml_tensor * src = ਗੰਢ->src[s];
                    ਜੇ (src && ਵੇਖਿਆ.insert(src).second) {
                        params->ਜੋੜੋ_alloc_dep(params->user_ਅੰਕੜੇ, src, graph->nodes[ਆਖਰੀ_ਗੰਢ]);
                    }
                }
                ਵੇਖਿਆ.insert(ਗੰਢ);
            }
        };

        ਸਵੈਚਾਲਿਤ ਅਚਲ &keep_topk_moe_ਨਮੂਨਾ = [&](ਅਚਲ std::initializer_list<ggml_op> &pattern) -> ਤਾਰਕਿਕ {
            ਜੇ (!ਮੇਲ_ਨਮੂਨਾ(pattern, ਪਹਿਲਾ_ਅਣਵਰਤਿਆ)) {
                ਵਾਪਸ ਝੂਠ;
            }

            ਪੂਰਨ ਆਖਰੀ_ਗੰਢ = ਪਹਿਲਾ_ਅਣਵਰਤਿਆ + (ਪੂਰਨ) pattern.size() - 1;
            // Some TOPK_MOE variants fuse a trailing scale.
            ਜੇ (ਆਖਰੀ_ਗੰਢ + 1 < graph->n_nodes && graph->nodes[ਆਖਰੀ_ਗੰਢ + 1]->op == GGML_OP_SCALE) {
                ਆਖਰੀ_ਗੰਢ++;
            }

            ਜੋੜੋ_ਨਮੂਨਾ_alloc_deps(pattern, ਆਖਰੀ_ਗੰਢ);

            ਵਾਪਸ keep_ਨਮੂਨਾ(pattern);
        };

        ਜੇ (keep_topk_moe_ਨਮੂਨਾ(topk_moe_early_softmax_norm)) {
            ਜਾਰੀ;
        }
        ਜੇ (keep_topk_moe_ਨਮੂਨਾ(topk_moe_sigmoid_norm_bias)) {
            ਜਾਰੀ;
        }
        ਜੇ (keep_topk_moe_ਨਮੂਨਾ(topk_moe_sqrt_softplus_norm_bias)) {
            ਜਾਰੀ;
        }
        ਜੇ (keep_topk_moe_ਨਮੂਨਾ(topk_moe_early_softmax)) {
            ਜਾਰੀ;
        }
        ਜੇ (keep_topk_moe_ਨਮੂਨਾ(topk_moe_late_softmax)) {
            ਜਾਰੀ;
        }
        ਜੇ (keep_ਨਮੂਨਾ(snake_ਨਮੂਨਾ)) {
            ਜਾਰੀ;
        }
        ਜੇ (keep_ਨਮੂਨਾ(topk_qsa_ਨਮੂਨਾ)) {
            ਜਾਰੀ;
        }

        ਜੇ (keep_ਨਮੂਨਾ(rms_norm_mul_ਜੋੜੋ_mul_ਨਮੂਨਾ)) {
            ਜਾਰੀ;
        }
        ਜੇ (keep_ਨਮੂਨਾ(rms_norm_mul_ਜੋੜੋ_ਨਮੂਨਾ)) {
            ਜਾਰੀ;
        }
        ਜੇ (keep_ਨਮੂਨਾ(rms_norm_mul_rope_view_ਸੈੱਟ_ਪੰਕਤੀਆਂ_ਨਮੂਨਾ)) {
            ਜਾਰੀ;
        }
        ਜੇ (keep_ਨਮੂਨਾ(rms_norm_view_ਸੈੱਟ_ਪੰਕਤੀਆਂ_ਨਮੂਨਾ)) {
            ਜਾਰੀ;
        }
        ਜੇ (keep_ਨਮੂਨਾ(rope_view_ਸੈੱਟ_ਪੰਕਤੀਆਂ_ਨਮੂਨਾ)) {
            ਜਾਰੀ;
        }
        ਜੇ (ਮੇਲ_ਨਮੂਨਾ(hc_post_gate_ਨਮੂਨਾ, ਪਹਿਲਾ_ਅਣਵਰਤਿਆ)) {
            ਜੋੜੋ_ਨਮੂਨਾ_alloc_deps(hc_post_gate_ਨਮੂਨਾ, ਪਹਿਲਾ_ਅਣਵਰਤਿਆ + (ਪੂਰਨ) hc_post_gate_ਨਮੂਨਾ.size() - 1);
            keep_ਨਮੂਨਾ(hc_post_gate_ਨਮੂਨਾ);
            ਜਾਰੀ;
        }

        // First, grab the next unused node.
        current_ਸੈੱਟ.push_back(ਪਹਿਲਾ_ਅਣਵਰਤਿਆ);

        // Loop through the next N nodes. Grab any that don't depend on other nodes that
        // haven't already been run. Nodes that have already been run have used[i] set
        // to true. Allow nodes that depend on the previous node if it's a fusion pattern
        // that we support (e.g. RMS_NORM + MUL).
        // This first pass only grabs "real" (non-view nodes). Second pass grabs view nodes.
        // The goal is to not interleave real and view nodes in a way that breaks fusion.
        ਅਚਲ ਪੂਰਨ ਅੰਕ_ਮੰਜ਼ਿਲ_CHECK = 20;
        ਲਈ (ਪੂਰਨ j = ਪਹਿਲਾ_ਅਣਵਰਤਿਆ+1; j < std::min(ਪਹਿਲਾ_ਅਣਵਰਤਿਆ + ਅੰਕ_ਮੰਜ਼ਿਲ_CHECK, graph->n_nodes); ++j) {
            ਜੇ (ਵਰਤਿਆ[j]) {
                ਜਾਰੀ;
            }
            ਜੇ (is_ਖਾਲੀ_ਥਾਵਾਂ(graph->nodes[j])) {
                ਜਾਰੀ;
            }
            // Protect every interior QSA node (not just the start): the mask branch is
            // independent, so it gets pulled out and breaks keep_pattern otherwise.
            ਸਵੈਚਾਲਿਤ ਅਚਲ &ਵਿੱਚ_qsa_ਨਮੂਨਾ = [&](ਪੂਰਨ n) -> ਤਾਰਕਿਕ {
                ਲਈ (ਪੂਰਨ o = 0; o < (ਪੂਰਨ) topk_qsa_ਨਮੂਨਾ.size(); ++o) {
                    ਜੇ (n - o >= 0 && ਮੇਲ_ਨਮੂਨਾ(topk_qsa_ਨਮੂਨਾ, n - o)) {
                        ਵਾਪਸ ਸੱਚ;
                    }
                }
                ਵਾਪਸ ਝੂਠ;
            };
            ਜੇ (ਮੇਲ_ਨਮੂਨਾ(topk_moe_early_softmax_norm, j) ||
                ਮੇਲ_ਨਮੂਨਾ(topk_moe_sigmoid_norm_bias, j) ||
                ਮੇਲ_ਨਮੂਨਾ(topk_moe_sqrt_softplus_norm_bias, j) ||
                ਮੇਲ_ਨਮੂਨਾ(topk_moe_early_softmax, j) ||
                ਮੇਲ_ਨਮੂਨਾ(topk_moe_late_softmax, j) ||
                ਮੇਲ_ਨਮੂਨਾ(snake_ਨਮੂਨਾ, j) ||
                ਵਿੱਚ_qsa_ਨਮੂਨਾ(j) ||
                ਮੇਲ_ਨਮੂਨਾ(rms_norm_mul_ਜੋੜੋ_mul_ਨਮੂਨਾ, j) ||
                ਮੇਲ_ਨਮੂਨਾ(rms_norm_mul_ਜੋੜੋ_ਨਮੂਨਾ, j) ||
                ਮੇਲ_ਨਮੂਨਾ(rms_norm_mul_rope_view_ਸੈੱਟ_ਪੰਕਤੀਆਂ_ਨਮੂਨਾ, j) ||
                ਮੇਲ_ਨਮੂਨਾ(rms_norm_view_ਸੈੱਟ_ਪੰਕਤੀਆਂ_ਨਮੂਨਾ, j) ||
                ਮੇਲ_ਨਮੂਨਾ(rope_view_ਸੈੱਟ_ਪੰਕਤੀਆਂ_ਨਮੂਨਾ, j) ||
                ਮੇਲ_ਨਮੂਨਾ(hc_post_gate_ਨਮੂਨਾ, j)) {
                ਜਾਰੀ;
            }
            ਤਾਰਕਿਕ ok = ਸੱਚ;
            ਲਈ (ਪੂਰਨ c = ਪਹਿਲਾ_ਅਣਵਰਤਿਆ; c < j; ++c) {
                ਜੇ (!ਵਰਤਿਆ[c] &&
                    is_src_ਦਾ(graph->nodes[j], graph->nodes[c]) &&
                    !(c == current_ਸੈੱਟ.back() && graph->nodes[c]->op == GGML_OP_RMS_NORM && graph->nodes[j]->op == GGML_OP_MUL && ਖਾਲੀ_ਥਾਵਾਂ_or_scheduled_between(c+1, j)) &&
                    !(c == current_ਸੈੱਟ.back() && graph->nodes[c]->op == GGML_OP_UNARY && graph->nodes[j]->op == GGML_OP_MUL && ਖਾਲੀ_ਥਾਵਾਂ_or_scheduled_between(c+1, j)) &&
                    !(j == c+1 && c == current_ਸੈੱਟ.back() && graph->nodes[c]->op == GGML_OP_MUL_MAT && graph->nodes[j]->op == GGML_OP_ਜੋੜੋ) &&
                    !(j == c+1 && c == current_ਸੈੱਟ.back() && graph->nodes[c]->op == GGML_OP_MUL_MAT_ID && graph->nodes[j]->op == GGML_OP_ਜੋੜੋ_ID) &&
                    !(j == c+1 && c == current_ਸੈੱਟ.back() && graph->nodes[c]->op == GGML_OP_MUL_MAT_ID && graph->nodes[j]->op == GGML_OP_MUL) &&
                    !(j == c+1 && c == current_ਸੈੱਟ.back() && graph->nodes[c]->op == GGML_OP_ਜੋੜੋ && graph->nodes[j]->op == GGML_OP_ਜੋੜੋ) &&
                    !(j == c+1 && c == current_ਸੈੱਟ.back() && graph->nodes[c]->op == GGML_OP_SSM_CONV && graph->nodes[j]->op == GGML_OP_ਜੋੜੋ) &&
                    !(j == c+1 && c == current_ਸੈੱਟ.back() && graph->nodes[c]->op == GGML_OP_SSM_CONV && graph->nodes[j]->op == GGML_OP_UNARY)) {
                    ok = ਝੂਠ;
                    ਰੋਕੋ;
                }
            }
            ਜੇ (ok) {
                current_ਸੈੱਟ.push_back(j);

                ਪੂਰਨ rope_idx = j;

                // When we've found RMS_NORM + MUL, try to find a ROPE that uses it
                ਜੇ (j > 0 &&
                    graph->nodes[j]->op == GGML_OP_MUL &&
                    graph->nodes[j-1]->op == GGML_OP_RMS_NORM) {
                    ਲਈ (ਪੂਰਨ k = j + 1; k < std::min(j + 15, graph->n_nodes); ++k) {
                        ਜੇ (graph->nodes[k]->op == GGML_OP_ROPE &&
                            graph->nodes[k]->src[0] == graph->nodes[j] &&
                            // Check that other srcs are already valid
                            graph->nodes[k]->src[1]->op == GGML_OP_NONE &&
                            (graph->nodes[k]->src[2] == ਖਾਲੀ_ਸੂਚਕ || graph->nodes[k]->src[2]->op == GGML_OP_NONE)) {
                            rope_idx = k;
                            current_ਸੈੱਟ.push_back(rope_idx);
                            ਵਰਤਿਆ[rope_idx] = ਸੱਚ;
                            ਰੋਕੋ;
                        }
                    }
                }
                // Look for ROPE/RMS_NORM + VIEW + SET_ROWS and make them consecutive
                ਜੇ (graph->nodes[rope_idx]->op == GGML_OP_ROPE || graph->nodes[rope_idx]->op == GGML_OP_RMS_NORM) {
                    ਪੂਰਨ view_idx = -1;
                    ਪੂਰਨ ਸੈੱਟ_ਪੰਕਤੀਆਂ_idx = -1;
                    ਲਈ (ਪੂਰਨ k = rope_idx + 1; k < std::min(rope_idx + 15, graph->n_nodes); ++k) {
                        ਜੇ (ਵਰਤਿਆ[k]) {
                            ਜਾਰੀ;
                        }
                        ਜੇ (view_idx == -1 && graph->nodes[k]->op == GGML_OP_VIEW && graph->nodes[k]->src[0] == graph->nodes[rope_idx]) {
                            view_idx = k;
                            ਜਾਰੀ;
                        }
                        ਜੇ (view_idx != -1 && graph->nodes[k]->op == GGML_OP_ਸੈੱਟ_ਪੰਕਤੀਆਂ && graph->nodes[k]->src[0] == graph->nodes[view_idx]) {
                            ਸੈੱਟ_ਪੰਕਤੀਆਂ_idx = k;
                            ਰੋਕੋ;
                        }
                    }
                    ਜੇ (ਸੈੱਟ_ਪੰਕਤੀਆਂ_idx != -1) {
                        ਅਚਲ ਪੂਰਨ ਗੰਢ_idxs[] = { rope_idx, view_idx, ਸੈੱਟ_ਪੰਕਤੀਆਂ_idx };
                        ਅਚਲ ggml_op ops[] = { graph->nodes[rope_idx]->op, GGML_OP_VIEW, GGML_OP_ਸੈੱਟ_ਪੰਕਤੀਆਂ };
                        ਤਾਰਕਿਕ can_pull = ggml_can_fuse_subgraph_ext(graph, ਗੰਢ_idxs, 3, ops, &ਸੈੱਟ_ਪੰਕਤੀਆਂ_idx, 1);

                        ਲਈ (ਪੂਰਨ c = rope_idx + 1; can_pull && c < ਸੈੱਟ_ਪੰਕਤੀਆਂ_idx; ++c) {
                            ਜੇ (!ਵਰਤਿਆ[c] && c != view_idx && !is_ਖਾਲੀ_ਥਾਵਾਂ(graph->nodes[c]) &&
                                is_src_ਦਾ(graph->nodes[ਸੈੱਟ_ਪੰਕਤੀਆਂ_idx], graph->nodes[c])) {
                                can_pull = ਝੂਠ;
                            }
                        }

                        ਜੇ (can_pull) {
                            current_ਸੈੱਟ.push_back(view_idx);
                            current_ਸੈੱਟ.push_back(ਸੈੱਟ_ਪੰਕਤੀਆਂ_idx);
                            ਵਰਤਿਆ[view_idx] = ਸੱਚ;
                            ਵਰਤਿਆ[ਸੈੱਟ_ਪੰਕਤੀਆਂ_idx] = ਸੱਚ;
                        }
                    }
                }
                // Look for MUL_MAT_ID + ADD_ID + MUL
                ਜੇ (j > 0 &&
                    graph->nodes[j]->op == GGML_OP_ਜੋੜੋ_ID &&
                    graph->nodes[j-1]->op == GGML_OP_MUL_MAT_ID) {
                    ਲਈ (ਪੂਰਨ k = j + 1; k < std::min(j + 15, graph->n_nodes); ++k) {
                        ਜੇ (graph->nodes[k]->op == GGML_OP_MUL &&
                            graph->nodes[k]->src[0] == graph->nodes[j] &&
                            // src1 must either be weights or already processed
                            (graph->nodes[k]->src[1]->op == GGML_OP_NONE || ਵਰਤਿਆ_ਗੰਢ_ਸੈੱਟ.find(graph->nodes[k]->src[1]) != ਵਰਤਿਆ_ਗੰਢ_ਸੈੱਟ.end())) {
                            current_ਸੈੱਟ.push_back(k);
                            ਵਰਤਿਆ[k] = ਸੱਚ;
                            ਰੋਕੋ;
                        }
                    }
                }
                // Look for MUL_MAT + ADD + ADD
                ਜੇ (j > 0 &&
                    graph->nodes[j]->op == GGML_OP_ਜੋੜੋ &&
                    graph->nodes[j-1]->op == GGML_OP_MUL_MAT) {
                    ਲਈ (ਪੂਰਨ k = j + 1; k < std::min(j + 15, graph->n_nodes); ++k) {
                        ਜੇ (graph->nodes[k]->op == GGML_OP_ਜੋੜੋ &&
                            graph->nodes[k]->src[0] == graph->nodes[j] &&
                            // src1 must either be weights or already processed
                            (graph->nodes[k]->src[1]->op == GGML_OP_NONE || ਵਰਤਿਆ_ਗੰਢ_ਸੈੱਟ.find(graph->nodes[k]->src[1]) != ਵਰਤਿਆ_ਗੰਢ_ਸੈੱਟ.end())) {
                            current_ਸੈੱਟ.push_back(k);
                            ਵਰਤਿਆ[k] = ਸੱਚ;
                            ਰੋਕੋ;
                        }
                    }
                }
                // SSM_CONV + ADD + UNARY: pull the consuming UNARY forward
                ਜੇ (j > 0 &&
                    graph->nodes[j]->op == GGML_OP_ਜੋੜੋ &&
                    graph->nodes[j-1]->op == GGML_OP_SSM_CONV) {
                    ਲਈ (ਪੂਰਨ k = j + 1; k < std::min(j + 15, graph->n_nodes); ++k) {
                        ਜੇ (graph->nodes[k]->op == GGML_OP_UNARY &&
                            graph->nodes[k]->src[0] == graph->nodes[j]) {
                            current_ਸੈੱਟ.push_back(k);
                            ਵਰਤਿਆ[k] = ਸੱਚ;
                            ਰੋਕੋ;
                        }
                    }
                }
                // UNARY + MUL: pull the consuming MUL forward
                ਜੇ (j > 0 &&
                    graph->nodes[j]->op == GGML_OP_UNARY) {
                    ਲਈ (ਪੂਰਨ k = j + 1; k < std::min(j + 15, graph->n_nodes); ++k) {
                        ggml_tensor * mul = graph->nodes[k];
                        ਜੇ (mul->op != GGML_OP_MUL || (mul->src[0] != graph->nodes[j] && mul->src[1] != graph->nodes[j])) {
                            ਜਾਰੀ;
                        }
                        ggml_tensor * other = (mul->src[0] == graph->nodes[j]) ? mul->src[1] : mul->src[0];
                        // the other src must either be weights or already processed
                        ਜੇ (!(other->op == GGML_OP_NONE || ਵਰਤਿਆ_ਗੰਢ_ਸੈੱਟ.find(other) != ਵਰਤਿਆ_ਗੰਢ_ਸੈੱਟ.end())) {
                            ਜਾਰੀ;
                        }
                        ਜੇ (!ggml_vk_can_fuse_unary_mul(graph, j, k)) {
                            ਜਾਰੀ;
                        }
                        current_ਸੈੱਟ.push_back(k);
                        ਵਰਤਿਆ[k] = ਸੱਚ;
                        ਰੋਕੋ;
                    }
                }
            }
        }
        // Second pass grabs view nodes.
        // Skip this if it would break a fusion optimization (don't split up add->rms_norm or add->add).
        ਜੇ (graph->nodes[current_ਸੈੱਟ.back()]->op != GGML_OP_ਜੋੜੋ) {
            ਲਈ (ਪੂਰਨ j = ਪਹਿਲਾ_ਅਣਵਰਤਿਆ+1; j < std::min(ਪਹਿਲਾ_ਅਣਵਰਤਿਆ + ਅੰਕ_ਮੰਜ਼ਿਲ_CHECK, graph->n_nodes); ++j) {
                ਜੇ (ਵਰਤਿਆ[j]) {
                    ਜਾਰੀ;
                }
                ਜੇ (!is_ਖਾਲੀ_ਥਾਵਾਂ(graph->nodes[j])) {
                    ਜਾਰੀ;
                }
                ਤਾਰਕਿਕ ok = ਸੱਚ;
                ਲਈ (ਪੂਰਨ c = ਪਹਿਲਾ_ਅਣਵਰਤਿਆ; c < j; ++c) {
                    ਤਾਰਕਿਕ c_ਵਿੱਚ_current_ਸੈੱਟ = std::find(current_ਸੈੱਟ.begin(), current_ਸੈੱਟ.end(), c) != current_ਸੈੱਟ.end();
                    // skip views whose srcs haven't been processed.
                    ਜੇ (!ਵਰਤਿਆ[c] &&
                        is_src_ਦਾ(graph->nodes[j], graph->nodes[c]) &&
                        !c_ਵਿੱਚ_current_ਸੈੱਟ) {
                        ok = ਝੂਠ;
                        ਰੋਕੋ;
                    }
                }
                ਜੇ (ok) {
                    current_ਸੈੱਟ.push_back(j);
                }
            }
        }

        // Push the current set into new_order
        ਲਈ (ਸਵੈਚਾਲਿਤ c : current_ਸੈੱਟ) {
            new_ਕ੍ਰਮ.push_back(graph->nodes[c]);
            ਵਰਤਿਆ_ਗੰਢ_ਸੈੱਟ.insert(graph->nodes[c]);
            ਵਰਤਿਆ[c] = ਸੱਚ;
        }
        ਜਦੋਂ (ਪਹਿਲਾ_ਅਣਵਰਤਿਆ < graph->n_nodes && ਵਰਤਿਆ[ਪਹਿਲਾ_ਅਣਵਰਤਿਆ]) {
            ਪਹਿਲਾ_ਅਣਵਰਤਿਆ++;
        }
    }
    // Replace the graph with the new order.
    ਲਈ (ਪੂਰਨ i = 0; i < graph->n_nodes; ++i) {
        graph->nodes[i] = new_ਕ੍ਰਮ[i];
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_vk_event_ਰਿਕਾਰਡ(ggml_backend_t backend, ggml_backend_event_t event) {
    VK_LOG_DEBUG("ggml_backend_vk_event_record(backend=" << backend << ", event=" << event << ")");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    vk_event *vkev = (vk_event *)event->context;

    ggml_vk_submit_transfer_ctx(ctx);

    vk_context compute_ctx = ggml_vk_get_compute_ctx(ctx);
    ਸਵੈਚਾਲਿਤ* cmd_buf = compute_ctx->s->ਬਫਰ; // retrieve pointer before it gets reset

    ਜੇ (vkev->has_event) {
        // Move existing event into submitted
        vkev->events_submitted.push_back(vkev->event);
    }

    // Grab the next event and record it, create one if necessary
    ਜੇ (vkev->events_free.ਖਾਲੀ_ਥਾਵਾਂ()) {
        vkev->event = ctx->device->device.createEvent({});
    } ਨਹੀਂਤਾਂ {
        vkev->event = vkev->events_free.back();
        vkev->events_free.pop_back();
    }

    vkev->has_event = ਸੱਚ;

    ggml_vk_ਸੈੱਟ_event(compute_ctx, vkev->event);

    vkev->tl_semaphore.value++;
    compute_ctx->s->signal_semaphores.push_back(vkev->tl_semaphore);
    ggml_vk_ctx_ਸਿਰਾ(compute_ctx);

    ggml_vk_submit(compute_ctx, {});
    ctx->submit_pending = ਸੱਚ;
    vkev->cmd_ਬਫਰ = cmd_buf;
    vkev->cmd_ਬਫਰ_use_counter = cmd_buf->use_counter;
    ctx->compute_ctx.reset();
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_vk_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    VK_LOG_DEBUG("ggml_backend_vk_event_wait(backend=" << backend << ", event=" << event << ")");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    vk_event *vkev = (vk_event *)event->context;

    vk_context compute_ctx = ggml_vk_get_compute_ctx(ctx);

    ਜੇ (vkev->has_event) {
        // Wait for latest event
        ggml_vk_wait_events(compute_ctx, { vkev->event });

        ਜੇ (ctx->device->async_use_transfer_ਕਤਾਰ) {
            vk_context transfer_ctx = ggml_vk_get_transfer_ctx(ctx);
            transfer_ctx->s->wait_semaphores.push_back(vkev->tl_semaphore);
        }
    }
}

ਸਥਿਰ ggml_backend_i ggml_backend_vk_interface = {
    /* .get_name                = */ ggml_backend_vk_name,
    /* .free                    = */ ggml_backend_vk_free,
    /* .set_tensor_async        = */ ggml_backend_vk_ਸੈੱਟ_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_vk_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_vk_ਸੈੱਟ_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_vk_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ ggml_backend_vk_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_vk_synchronize,
    /* .graph_plan_create       = */ ਨਿਰਾ,
    /* .graph_plan_free         = */ ਨਿਰਾ,
    /* .graph_plan_update       = */ ਨਿਰਾ,
    /* .graph_plan_compute      = */ ਨਿਰਾ,
    /* .graph_compute           = */ ggml_backend_vk_ਗ੍ਰਾਫ_compute,
    /* .event_record            = */ ggml_backend_vk_event_ਰਿਕਾਰਡ,
    /* .event_wait              = */ ggml_backend_vk_event_wait,
    /* .graph_optimize          = */ ggml_vk_ਗ੍ਰਾਫ_optimize,
};

ਸਥਿਰ ggml_guid_t ggml_backend_vk_guid() {
    ਸਥਿਰ ggml_guid guid = { 0xb8, 0xf7, 0x4f, 0x86, 0x40, 0x3c, 0xe1, 0x02, 0x91, 0xc8, 0xdd, 0xe9, 0x02, 0x3f, 0xc0, 0x2b };
    ਵਾਪਸ &guid;
}

ggml_backend_t ggml_backend_vk_init(size_t dev_ਅੰਕ) {
    VK_LOG_DEBUG("ggml_backend_vk_init(" << dev_ਅੰਕ << ")");

    ggml_backend_vk_context * ctx = ਨਵਾਂ ggml_backend_vk_context;
    ggml_vk_init(ctx, dev_ਅੰਕ);

    ggml_backend_t vk_backend = ਨਵਾਂ ggml_backend {
        /* .guid    = */ ggml_backend_vk_guid(),
        /* .iface   = */ ggml_backend_vk_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_vk_reg(), dev_ਅੰਕ),
        /* .context = */ ctx,
    };

    ਜੇ (!ctx->device->support_async) {
        vk_backend->iface.get_tensor_async = ਖਾਲੀ_ਸੂਚਕ;
    }

    ਵਾਪਸ vk_backend;
}

ਤਾਰਕਿਕ ggml_backend_is_vk(ggml_backend_t backend) {
    ਵਾਪਸ backend != ਨਿਰਾ && ggml_guid_matches(backend->guid, ggml_backend_vk_guid());
}

ਪੂਰਨ ggml_backend_vk_get_device_ਗਿਣਤੀ() {
    ਵਾਪਸ ggml_vk_get_device_ਗਿਣਤੀ();
}

ਖਾਲੀ ggml_backend_vk_get_device_description(ਪੂਰਨ device, ਅੱਖਰ * description, size_t description_ਮਾਪ) {
    GGML_ASSERT(device < (ਪੂਰਨ) vk_instance.device_indices.size());
    ਪੂਰਨ dev_idx = vk_instance.device_indices[device];
    ggml_vk_get_device_description(dev_idx, description, description_ਮਾਪ);
}

ਖਾਲੀ ggml_backend_vk_get_device_ਸਮ੍ਰਿਤੀ(ਪੂਰਨ device, size_t * ਮੁਕਤ, size_t * ਕੁੱਲ) {
    GGML_ASSERT(device < (ਪੂਰਨ) vk_instance.device_indices.size());
    GGML_ASSERT(device < (ਪੂਰਨ) vk_instance.device_supports_membudget.size());

    vk::PhysicalDevice vkdev = vk_instance.instance.enumeratePhysicalDevices()[vk_instance.device_indices[device]];
    vk::Physical_Device_ਸਮ੍ਰਿਤੀ_Budget_Properties_EXT budgetprops;
    vk::Physical_Device_ਸਮ੍ਰਿਤੀ_Properties2 memprops = {};
    ਅਚਲ ਤਾਰਕਿਕ membudget_supported = vk_instance.device_supports_membudget[device];
    ਅਚਲ ਤਾਰਕਿਕ is_integrated_gpu = vkdev.getProperties().deviceType == vk::PhysicalDeviceType::eIntegratedGpu;

    ਜੇ (membudget_supported) {
        memprops.p_ਅਗਲਾ = &budgetprops;
    }
    vkdev.get_ਸਮ੍ਰਿਤੀ_Properties2(&memprops);

    *ਕੁੱਲ = 0;
    *ਮੁਕਤ = 0;

    ਲਈ (uint32_t i = 0; i < memprops.ਸਮ੍ਰਿਤੀ_Properties.ਸਮ੍ਰਿਤੀ_ਢੇਰ_ਗਿਣਤੀ; ++i) {
        ਅਚਲ vk::ਸਮ੍ਰਿਤੀ_ਢੇਰ & ਢੇਰ_2 = memprops.ਸਮ੍ਰਿਤੀ_Properties.ਸਮ੍ਰਿਤੀ_Heaps[i];

        ਜੇ (is_integrated_gpu || (ਢੇਰ_2.flags & vk::ਸਮ੍ਰਿਤੀ_ਢੇਰ_Flag_ਬਿੱਟ::eDeviceLocal)) {
            *ਕੁੱਲ += ਢੇਰ_2.size;

            ਜੇ (membudget_supported && i < budgetprops.ਢੇਰ_Usage.size()) {
                // Driver may report heapBudget > heap.size so we add a safeguard here
                // See https://github.com/ggml-org/llama.cpp/issues/29277
                ਅਚਲ vk::Device_ਮਾਪ budget = std::min(budgetprops.ਢੇਰ_Budget[i], ਢੇਰ_2.size);
                *ਮੁਕਤ += budget > budgetprops.ਢੇਰ_Usage[i] ? budget - budgetprops.ਢੇਰ_Usage[i] : 0;
            } ਨਹੀਂਤਾਂ {
                *ਮੁਕਤ += ਢੇਰ_2.size;
            }
        }
    }
}

ਸਥਿਰ vk::PhysicalDeviceType ggml_backend_vk_get_device_type(ਪੂਰਨ device_idx) {
    GGML_ASSERT(device_idx >= 0 && device_idx < (ਪੂਰਨ) vk_instance.device_indices.size());

    vk::PhysicalDevice device = vk_instance.instance.enumeratePhysicalDevices()[vk_instance.device_indices[device_idx]];

    vk::PhysicalDeviceProperties2 props = {};
    device.getProperties2(&props);

    ਵਾਪਸ props.properties.deviceType;
}

ਸਥਿਰ std::string ggml_backend_vk_get_device_pci_id(ਪੂਰਨ device_idx) {
    GGML_ASSERT(device_idx >= 0 && device_idx < (ਪੂਰਨ) vk_instance.device_indices.size());

    vk::PhysicalDevice device = vk_instance.instance.enumeratePhysicalDevices()[vk_instance.device_indices[device_idx]];

    ਅਚਲ std::vector<vk::ExtensionProperties> ext_props = device.enumerateDeviceExtensionProperties();

    ਤਾਰਕਿਕ ext_support = ਝੂਠ;

    ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ& properties : ext_props) {
        ਜੇ (strcmp("VK_EXT_pci_bus_info", properties.extensionName) == 0) {
            ext_support = ਸੱਚ;
            ਰੋਕੋ;
        }
    }

    ਜੇ (!ext_support) {
        ਵਾਪਸ "";
    }

    vk::PhysicalDeviceProperties2 props = {};
    vk::PhysicalDevicePCIBusInfoPropertiesEXT pci_bus_info = {};

    props.p_ਅਗਲਾ = &pci_bus_info;

    device.getProperties2(&props);

    ਅਚਲ uint32_t pci_domain = pci_bus_info.pciDomain;
    ਅਚਲ uint32_t pci_bus = pci_bus_info.pciBus;
    ਅਚਲ uint32_t pci_device = pci_bus_info.pciDevice;
    ਅਚਲ uint8_t pci_function = (uint8_t) pci_bus_info.pciFunction; // pci function is between 0 and 7, prevent printf overflow warning

    ਅੱਖਰ pci_bus_id[16] = {};
    snprintf(pci_bus_id, ਆਕਾਰ(pci_bus_id), "%04x:%02x:%02x.%x", pci_domain, pci_bus, pci_device, pci_function);

    ਵਾਪਸ std::string(pci_bus_id);
}

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_vk_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    ਵਾਪਸ ctx->name.c_str();
}

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_vk_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    ਵਾਪਸ ctx->description.c_str();
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_vk_device_get_ਸਮ੍ਰਿਤੀ(ggml_backend_dev_t device, size_t * ਮੁਕਤ, size_t * ਕੁੱਲ) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)device->context;
    ggml_backend_vk_get_device_ਸਮ੍ਰਿਤੀ(ctx->device, ਮੁਕਤ, ਕੁੱਲ);
}

ਸਥਿਰ ggml_backend_ਬਫਰ_type_t ggml_backend_vk_device_get_ਬਫਰ_type(ggml_backend_dev_t dev) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    ਵਾਪਸ ggml_backend_vk_ਬਫਰ_type(ctx->device);
}

ਸਥਿਰ ggml_backend_ਬਫਰ_type_t ggml_backend_vk_device_get_host_ਬਫਰ_type(ggml_backend_dev_t dev) {
    ਅਣਵਰਤਿਆ(dev);
    ਵਾਪਸ ggml_backend_vk_host_ਬਫਰ_type();
}

ਸਥਿਰ ਗਿਣਤੀ ggml_backend_dev_type ggml_backend_vk_device_get_type(ggml_backend_dev_t dev) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;

    ਵਾਪਸ ctx->is_integrated_gpu ? GGML_BACKEND_DEVICE_TYPE_IGPU : GGML_BACKEND_DEVICE_TYPE_GPU;
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_vk_device_get_props(ggml_backend_dev_t dev, ਬਣਤਰ ggml_backend_dev_props * props) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;

    props->name        = ggml_backend_vk_device_get_name(dev);
    props->description = ggml_backend_vk_device_get_description(dev);
    props->type        = ggml_backend_vk_device_get_type(dev);
    props->device_id   = ctx->pci_bus_id.ਖਾਲੀ_ਥਾਵਾਂ() ? ਖਾਲੀ_ਸੂਚਕ : ctx->pci_bus_id.c_str();
    ggml_backend_vk_device_get_ਸਮ੍ਰਿਤੀ(dev, &props->ਸਮ੍ਰਿਤੀ_free, &props->ਸਮ੍ਰਿਤੀ_ਕੁੱਲ);
    props->caps = {
        /* .async                 = */ ਸੱਚ,
        /* .host_buffer           = */ ਸੱਚ,
        /* .buffer_from_host_ptr  = */ ਝੂਠ,
        /* .events                = */ ਸੱਚ,
        /* .mmap_support          = */ !ctx->is_integrated_gpu,
    };
}

ਸਥਿਰ ggml_backend_t ggml_backend_vk_device_init(ggml_backend_dev_t dev, ਅਚਲ ਅੱਖਰ * params) {
    ਅਣਵਰਤਿਆ(params);
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    ਵਾਪਸ ggml_backend_vk_init(ctx->device);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_vk_device_supports_op(ggml_backend_dev_t dev, ਅਚਲ ggml_tensor * op) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    ਅਚਲ vk_device& device = ggml_vk_get_device(ctx->device);

    ਅਚਲ ਤਾਰਕਿਕ ਵਰਤਦਾ_bda = (op->op == GGML_OP_IM2_ਖਾਨਾ || op->op == GGML_OP_IM2_ਖਾਨਾ_3_D) &&
                          device->shader_int64 && device->ਬਫਰ_device_ਪਤਾ;

    ਸਵੈਚਾਲਿਤ ਅਚਲ & tensor_ਮਾਪ_supported = [&](size_t tensor_ਮਾਪ) {
        ਜੇ (tensor_ਮਾਪ > device->max_ਬਫਰ_ਮਾਪ) {
            ਵਾਪਸ ਝੂਠ;
        }
        // For im2col shaders using BDA, maxStorageBufferRange limit doesn't apply.
        // If shader64BitIndexing is enabled, maxStorageBufferRange limit doesn't apply.
        ਜੇ (!ਵਰਤਦਾ_bda && !device->shader_64b_indexing) {
            ਜੇ (tensor_ਮਾਪ > device->properties.limits.max_Storage_ਬਫਰ_Range) {
                ਵਾਪਸ ਝੂਠ;
            }
        }
        ਵਾਪਸ ਸੱਚ;
    };
    // reject any tensors larger than the max buffer size
    ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_SRC; i++) {
        ਜੇ (op->src[i] && !tensor_ਮਾਪ_supported(ggml_nbytes(op->src[i]))) {
            ਵਾਪਸ ਝੂਠ;
        }
    }
    ਜੇ (!tensor_ਮਾਪ_supported(ggml_nbytes(op))) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਚੋਣ (op->op) {
        ਹਾਲਤ GGML_OP_UNARY:
            ਚੋਣ (ggml_get_unary_op(op)) {
                ਹਾਲਤ GGML_UNARY_OP_EXP:
                ਹਾਲਤ GGML_UNARY_OP_EXPM1:
                ਹਾਲਤ GGML_UNARY_OP_ELU:
                ਹਾਲਤ GGML_UNARY_OP_GELU:
                ਹਾਲਤ GGML_UNARY_OP_GELU_ERF:
                ਹਾਲਤ GGML_UNARY_OP_GELU_ਤੇਜ਼:
                ਹਾਲਤ GGML_UNARY_OP_SILU:
                ਹਾਲਤ GGML_UNARY_OP_RELU:
                ਹਾਲਤ GGML_UNARY_OP_XIELU:
                ਹਾਲਤ GGML_UNARY_OP_NEG:
                ਹਾਲਤ GGML_UNARY_OP_TANH:
                ਹਾਲਤ GGML_UNARY_OP_SIGMOID:
                ਹਾਲਤ GGML_UNARY_OP_HARDSIGMOID:
                ਹਾਲਤ GGML_UNARY_OP_HARDSWISH:
                ਹਾਲਤ GGML_UNARY_OP_ABS:
                ਹਾਲਤ GGML_UNARY_OP_SOFTPLUS:
                ਹਾਲਤ GGML_UNARY_OP_ਕਦਮ:
                ਹਾਲਤ GGML_UNARY_OP_ਗੇੜਾ:
                ਹਾਲਤ GGML_UNARY_OP_CEIL:
                ਹਾਲਤ GGML_UNARY_OP_FLOOR:
                ਹਾਲਤ GGML_UNARY_OP_TRUNC:
                ਹਾਲਤ GGML_UNARY_OP_SGN:
                    ਵਾਪਸ (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                           (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                           (op->src[0]->type == op->type);
                ਮੂਲ:
                    ਵਾਪਸ ਝੂਠ;
            }
        ਹਾਲਤ GGML_OP_GLU:
            ਚੋਣ (ggml_get_glu_op(op)) {
                ਹਾਲਤ GGML_GLU_OP_GEGLU:
                ਹਾਲਤ GGML_GLU_OP_REGLU:
                ਹਾਲਤ GGML_GLU_OP_SWIGLU:
                ਹਾਲਤ GGML_GLU_OP_SWIGLU_OAI:
                ਹਾਲਤ GGML_GLU_OP_GEGLU_ERF:
                ਹਾਲਤ GGML_GLU_OP_GEGLU_ਤੇਜ਼:
                ਹਾਲਤ GGML_GLU_OP_SWIGLU_CLAMP:
                    ਵਾਪਸ (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                           (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                           (op->src[0]->type == op->type) &&
                           (!op->src[1] || op->src[1]->type == op->src[0]->type);
                ਮੂਲ:
                    ਵਾਪਸ ਝੂਠ;
            }
        ਹਾਲਤ GGML_OP_MUL_MAT:
        ਹਾਲਤ GGML_OP_MUL_MAT_ID:
            {
                ggml_type src0_type = op->src[0]->type;
                ਜੇ (op->op == GGML_OP_MUL_MAT_ID) {
                    ਜੇ (!ggml_vk_use_mul_mat_vec_id(op)) {
                        ਅਚਲ ggml_tensor * ids = op->src[2];
                        // Shaders store row IDs as uint16_t slot and token indices.
                        ਜੇ (ids->ne[0] > int64_t(UINT16_MAX) + 1 || ids->ne[1] > int64_t(UINT16_MAX) + 1) {
                            ਵਾਪਸ ਝੂਠ;
                        }
                    }
                    ਜੇ (!device->mul_mat_id_s[src0_type] && !device->mul_mat_id_m[src0_type] && !device->mul_mat_id_l[src0_type]) {
                        // If there's not enough shared memory for row_ids and the result tile, fallback to CPU
                        ਵਾਪਸ ਝੂਠ;
                    }
                    ਜੇ (ggml_get_op_params_i32(op, 3) == GGML_PREC_F32) {
                        ਵਾਪਸ ਝੂਠ;
                    }
                }
                ਚੋਣ (src0_type) {
                    ਹਾਲਤ GGML_TYPE_F32:
                    ਹਾਲਤ GGML_TYPE_F16:
                    ਹਾਲਤ GGML_TYPE_BF16:
                    ਹਾਲਤ GGML_TYPE_Q1_0:
                    ਹਾਲਤ GGML_TYPE_Q2_0:
                    ਹਾਲਤ GGML_TYPE_Q4_0:
                    ਹਾਲਤ GGML_TYPE_Q4_1:
                    ਹਾਲਤ GGML_TYPE_Q5_0:
                    ਹਾਲਤ GGML_TYPE_Q5_1:
                    ਹਾਲਤ GGML_TYPE_Q8_0:
                    ਹਾਲਤ GGML_TYPE_Q2_K:
                    ਹਾਲਤ GGML_TYPE_Q3_K:
                    ਹਾਲਤ GGML_TYPE_Q4_K:
                    ਹਾਲਤ GGML_TYPE_Q5_K:
                    ਹਾਲਤ GGML_TYPE_Q6_K:
                    ਹਾਲਤ GGML_TYPE_IQ1_S:
                    ਹਾਲਤ GGML_TYPE_IQ1_M:
                    ਹਾਲਤ GGML_TYPE_IQ2_XXS:
                    ਹਾਲਤ GGML_TYPE_IQ2_XS:
                    ਹਾਲਤ GGML_TYPE_IQ2_S:
                    ਹਾਲਤ GGML_TYPE_IQ3_XXS:
                    ਹਾਲਤ GGML_TYPE_IQ3_S:
                    ਹਾਲਤ GGML_TYPE_IQ4_XS:
                    ਹਾਲਤ GGML_TYPE_IQ4_NL:
                    ਹਾਲਤ GGML_TYPE_MXFP4:
                    ਹਾਲਤ GGML_TYPE_NVFP4:
                    ਹਾਲਤ GGML_TYPE_TQ1_0:
                    ਹਾਲਤ GGML_TYPE_TQ2_0:
                        ਰੋਕੋ;
                    ਮੂਲ:
                        ਵਾਪਸ ਝੂਠ;
                }
                ਬਣਤਰ ggml_tensor * a;
                ਬਣਤਰ ggml_tensor * b;
                ਜੇ (op->op == GGML_OP_MUL_MAT) {
                    a = op->src[0];
                    b = op->src[1];
                } ਨਹੀਂਤਾਂ {
                    a = op->src[2];
                    b = op->src[1];
                }
                ਜੇ (a->ne[3] != b->ne[3]) {
                    ਵਾਪਸ ਝੂਠ;
                }
                ਜੇ (!(ggml_vk_dim01_contiguous(op->src[0]) || op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_BF16) ||
                    !(ggml_vk_dim01_contiguous(op->src[1]) || op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16)) {
                    ਵਾਪਸ ਝੂਠ;
                }
                ਜੇ (op->src[0]->type == GGML_TYPE_BF16 && op->src[1]->type == GGML_TYPE_F16) {
                    // We currently don't have a bf16 x f16 shader, or an fp16->bf16 copy shader.
                    // So don't support this combination for now.
                    ਵਾਪਸ ਝੂਠ;
                }
                ਜੇ (op->src[1]->type == GGML_TYPE_BF16 && op->src[0]->type != GGML_TYPE_BF16 && op->op != GGML_OP_MUL_MAT) {
                    // BF16 in src1 (widened to F32/F16) is only wired up for MUL_MAT
                    ਵਾਪਸ ਝੂਠ;
                }

                ਵਾਪਸ ਸੱਚ;
            }
        ਹਾਲਤ GGML_OP_FLASH_ATTN_EXT:
            {
                ਤਾਰਕਿਕ coopmat2 = device->coopmat2;
                uint32_t HSK = op->src[1]->ne[0];
                uint32_t HSV = op->src[2]->ne[0];
                ਜੇ ((HSK % 8) != 0 || (HSV % 8) != 0) {
                    ਵਾਪਸ ਝੂਠ;
                }
                ਜੇ (op->src[4] && op->src[4]->type != GGML_TYPE_F32) {
                    ਵਾਪਸ ਝੂਠ;
                }
                ਜੇ (op->src[0]->type != GGML_TYPE_F32) {
                    ਵਾਪਸ ਝੂਠ;
                }
                ਜੇ (op->type != GGML_TYPE_F32) {
                    ਵਾਪਸ ਝੂਠ;
                }
                ਜੇ (op->src[3] && op->src[3]->type != GGML_TYPE_F16) {
                    ਵਾਪਸ ਝੂਠ;
                }
                ਸਵੈਚਾਲਿਤ fa_kv_ok = [](ggml_type t) {
                    ਚੋਣ (t) {
                    ਹਾਲਤ GGML_TYPE_F32:
                    ਹਾਲਤ GGML_TYPE_F16:
                    ਹਾਲਤ GGML_TYPE_BF16:
                    ਹਾਲਤ GGML_TYPE_Q8_0:
                    ਹਾਲਤ GGML_TYPE_Q5_1:
                    ਹਾਲਤ GGML_TYPE_Q5_0:
                    ਹਾਲਤ GGML_TYPE_Q4_1:
                    ਹਾਲਤ GGML_TYPE_Q4_0:
                    ਹਾਲਤ GGML_TYPE_IQ4_NL:
                        ਵਾਪਸ ਸੱਚ;
                    ਮੂਲ:
                        ਵਾਪਸ ਝੂਠ;
                    }
                };
                ਜੇ (!fa_kv_ok(op->src[1]->type) || !fa_kv_ok(op->src[2]->type)) {
                    ਵਾਪਸ ਝੂਠ;
                }
                ਜੇ ((op->src[1]->type == GGML_TYPE_BF16) != (op->src[2]->type == GGML_TYPE_BF16)) {
                    ਵਾਪਸ ਝੂਠ;
                }
                ਜੇ (!coopmat2 && !(device->subgroup_shuffle && device->subgroup_vote)) {
                    // scalar/coopmat1 FA uses subgroupShuffle/subgroupAll
                    ਵਾਪਸ ਝੂਠ;
                }
                ਵਾਪਸ ਸੱਚ;
            }
        ਹਾਲਤ GGML_OP_GET_ਪੰਕਤੀਆਂ:
            {
                ਚੋਣ (op->src[0]->type) {
                    ਹਾਲਤ GGML_TYPE_F32:
                    ਹਾਲਤ GGML_TYPE_F16:
                    ਹਾਲਤ GGML_TYPE_BF16:
                    ਹਾਲਤ GGML_TYPE_Q1_0:
                    ਹਾਲਤ GGML_TYPE_Q2_0:
                    ਹਾਲਤ GGML_TYPE_Q4_0:
                    ਹਾਲਤ GGML_TYPE_Q4_1:
                    ਹਾਲਤ GGML_TYPE_Q5_0:
                    ਹਾਲਤ GGML_TYPE_Q5_1:
                    ਹਾਲਤ GGML_TYPE_Q8_0:
                    ਹਾਲਤ GGML_TYPE_Q2_K:
                    ਹਾਲਤ GGML_TYPE_Q3_K:
                    ਹਾਲਤ GGML_TYPE_Q4_K:
                    ਹਾਲਤ GGML_TYPE_Q5_K:
                    ਹਾਲਤ GGML_TYPE_Q6_K:
                    ਹਾਲਤ GGML_TYPE_IQ1_S:
                    ਹਾਲਤ GGML_TYPE_IQ1_M:
                    ਹਾਲਤ GGML_TYPE_IQ2_XXS:
                    ਹਾਲਤ GGML_TYPE_IQ2_XS:
                    ਹਾਲਤ GGML_TYPE_IQ2_S:
                    ਹਾਲਤ GGML_TYPE_IQ3_XXS:
                    ਹਾਲਤ GGML_TYPE_IQ3_S:
                    ਹਾਲਤ GGML_TYPE_IQ4_XS:
                    ਹਾਲਤ GGML_TYPE_IQ4_NL:
                    ਹਾਲਤ GGML_TYPE_MXFP4:
                    ਹਾਲਤ GGML_TYPE_NVFP4:
                    ਹਾਲਤ GGML_TYPE_TQ1_0:
                    ਹਾਲਤ GGML_TYPE_TQ2_0:
                    ਹਾਲਤ GGML_TYPE_I32:
                        ਵਾਪਸ ਸੱਚ;
                    ਮੂਲ:
                        ਵਾਪਸ ਝੂਠ;
                }
            }
        ਹਾਲਤ GGML_OP_GET_ਪੰਕਤੀਆਂ_ਪਿੱਠ:
            ਵਾਪਸ op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32;
        ਹਾਲਤ GGML_OP_ਸੈੱਟ_ਪੰਕਤੀਆਂ:
            {
                ਜੇ ((op->src[0]->type != GGML_TYPE_F32 && op->src[0]->type != GGML_TYPE_F16) ||
                    (op->src[1]->type != GGML_TYPE_I32 && op->src[1]->type != GGML_TYPE_I64)) {
                    ਵਾਪਸ ਝੂਠ;
                }
                ਚੋਣ (op->type) {
                    ਹਾਲਤ GGML_TYPE_F32:
                    ਹਾਲਤ GGML_TYPE_F16:
                    ਹਾਲਤ GGML_TYPE_BF16:
                    ਹਾਲਤ GGML_TYPE_Q1_0:
                    ਹਾਲਤ GGML_TYPE_Q2_0:
                    ਹਾਲਤ GGML_TYPE_Q4_0:
                    ਹਾਲਤ GGML_TYPE_Q4_1:
                    ਹਾਲਤ GGML_TYPE_Q5_0:
                    ਹਾਲਤ GGML_TYPE_Q5_1:
                    ਹਾਲਤ GGML_TYPE_Q8_0:
                    ਹਾਲਤ GGML_TYPE_IQ4_NL:
                        ਵਾਪਸ ਸੱਚ;
                    ਮੂਲ:
                        ਵਾਪਸ ਝੂਠ;
                }
            }
        ਹਾਲਤ GGML_OP_CONT:
        ਹਾਲਤ GGML_OP_CPY:
        ਹਾਲਤ GGML_OP_DUP:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1] != ਖਾਲੀ_ਸੂਚਕ ? op->src[1]->type : src0_type;

                ਜੇ (src0_type == GGML_TYPE_F32) {
                    ਚੋਣ (src1_type) {
                    ਹਾਲਤ GGML_TYPE_F32:
                    ਹਾਲਤ GGML_TYPE_F16:
                    ਹਾਲਤ GGML_TYPE_BF16:
                    ਹਾਲਤ GGML_TYPE_Q1_0:
                    ਹਾਲਤ GGML_TYPE_Q2_0:
                    ਹਾਲਤ GGML_TYPE_Q4_0:
                    ਹਾਲਤ GGML_TYPE_Q4_1:
                    ਹਾਲਤ GGML_TYPE_Q5_0:
                    ਹਾਲਤ GGML_TYPE_Q5_1:
                    ਹਾਲਤ GGML_TYPE_Q8_0:
                    ਹਾਲਤ GGML_TYPE_IQ4_NL:
                        ਵਾਪਸ ਸੱਚ;
                    ਮੂਲ:
                        ਰੋਕੋ;
                    }
                }
                ਜੇ (src1_type == GGML_TYPE_F32) {
                    ਚੋਣ (src0_type) {
                    ਹਾਲਤ GGML_TYPE_F16:
                    ਹਾਲਤ GGML_TYPE_BF16:
                    ਹਾਲਤ GGML_TYPE_Q1_0:
                    ਹਾਲਤ GGML_TYPE_Q2_0:
                    ਹਾਲਤ GGML_TYPE_Q4_0:
                    ਹਾਲਤ GGML_TYPE_Q4_1:
                    ਹਾਲਤ GGML_TYPE_Q5_0:
                    ਹਾਲਤ GGML_TYPE_Q5_1:
                    ਹਾਲਤ GGML_TYPE_Q8_0:
                    ਹਾਲਤ GGML_TYPE_IQ4_NL:
                        ਵਾਪਸ ਸੱਚ;
                    ਮੂਲ:
                        ਰੋਕੋ;
                    }
                }

                ਜੇ (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F16) {
                    ਵਾਪਸ ਸੱਚ;
                }

                ਜੇ (
                    (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_I32) ||
                    (src0_type == GGML_TYPE_I32 && src1_type == GGML_TYPE_F32)
                ) {
                    ਵਾਪਸ ਸੱਚ;
                }

                // We can handle copying from a type to the same type if it's
                // either not quantized or is quantized and contiguous.
                // We use f16 or f32 shaders to do the copy,
                // so the type/block size must be a multiple of 4.
                ਜੇ (src0_type == src1_type &&
                    (!ggml_is_quantized(src0_type) || (ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op))) &&
                    (ggml_type_ਮਾਪ(src0_type) % 2) == 0) {
                    ਵਾਪਸ ਸੱਚ;
                }
                ਵਾਪਸ ਝੂਠ;
            }
        ਹਾਲਤ GGML_OP_REPEAT:
            ਵਾਪਸ ggml_type_ਮਾਪ(op->type) == ggml_type_ਮਾਪ(op->src[0]->type) &&
                  (ggml_type_ਮਾਪ(op->type) == ਆਕਾਰ(ਦਸ਼ਮਲਵ) || ggml_type_ਮਾਪ(op->type) == 2);
        ਹਾਲਤ GGML_OP_REPEAT_ਪਿੱਠ:
            ਵਾਪਸ op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32;
        ਹਾਲਤ GGML_OP_ROPE:
            ਵਾਪਸ ggml_is_contiguous_ਪੰਕਤੀਆਂ(op) && ggml_is_contiguous_ਪੰਕਤੀਆਂ(op->src[0]);
        ਹਾਲਤ GGML_OP_ROPE_ਪਿੱਠ:
        ਹਾਲਤ GGML_OP_NONE:
        ਹਾਲਤ GGML_OP_RESHAPE:
        ਹਾਲਤ GGML_OP_VIEW:
        ਹਾਲਤ GGML_OP_ਕ੍ਰਮ_ਬਦਲੋ:
        ਹਾਲਤ GGML_OP_TRANSPOSE:
        ਹਾਲਤ GGML_OP_RMS_NORM:
            ਵਾਪਸ ਸੱਚ;
        ਹਾਲਤ GGML_OP_ਸਮੂਹ_NORM:
            ਵਾਪਸ ggml_is_contiguous(op->src[0]);
        ਹਾਲਤ GGML_OP_NORM:
        ਹਾਲਤ GGML_OP_L2_NORM:
            ਵਾਪਸ op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        ਹਾਲਤ GGML_OP_ਜੋੜੋ:
        ਹਾਲਤ GGML_OP_SUB:
        ਹਾਲਤ GGML_OP_MUL:
        ਹਾਲਤ GGML_OP_DIV:
            ਵਾਪਸ (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                   (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) &&
                   (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16);
        ਹਾਲਤ GGML_OP_ਜੋੜੋ_ID:
            ਵਾਪਸ op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 && op->src[2]->type == GGML_TYPE_I32 &&
                   op->type == GGML_TYPE_F32;
        ਹਾਲਤ GGML_OP_SILU_ਪਿੱਠ:
        ਹਾਲਤ GGML_OP_RMS_NORM_ਪਿੱਠ:
            ਵਾਪਸ ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        ਹਾਲਤ GGML_OP_SQR:
        ਹਾਲਤ GGML_OP_SQRT:
        ਹਾਲਤ GGML_OP_SIN:
        ਹਾਲਤ GGML_OP_COS:
        ਹਾਲਤ GGML_OP_CLAMP:
        ਹਾਲਤ GGML_OP_LEAKY_RELU:
            ਵਾਪਸ (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                   op->type == op->src[0]->type;
        ਹਾਲਤ GGML_OP_OPT_ਕਦਮ_ADAMW:
        ਹਾਲਤ GGML_OP_OPT_ਕਦਮ_SGD:
            ਵਾਪਸ ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        ਹਾਲਤ GGML_OP_ਬਾਹਰਲਾ_PROD:
            ਵਾਪਸ ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32
                && ggml_is_contiguous(op->src[1]) && op->src[1]->type == GGML_TYPE_F32
                && op->type == GGML_TYPE_F32;
        ਹਾਲਤ GGML_OP_LOG:
        ਹਾਲਤ GGML_OP_TRI:
        ਹਾਲਤ GGML_OP_DIAG:
            ਵਾਪਸ (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                   op->type == op->src[0]->type;
        ਹਾਲਤ GGML_OP_ARGSORT:
            {
                ਜੇ (!ggml_is_contiguous(op) || !ggml_is_contiguous(op->src[0])) {
                    ਵਾਪਸ ਝੂਠ;
                }
                // pipeline_argsort_large_f32 requires vulkan memory model.
                ਜੇ (device->vulkan_ਸਮ੍ਰਿਤੀ_model) {
                    ਵਾਪਸ ਸੱਚ;
                } ਨਹੀਂਤਾਂ {
                    ਵਾਪਸ op->ne[0] <= (1 << std::min(device->max_workgroup_ਮਾਪ_log2, ਅੰਕ_argsort_pipelines - 1));
                }
            }
        ਹਾਲਤ GGML_OP_ਸਿਖਰ_K:
            {
                ਜੇ (!ggml_is_contiguous(op) || !ggml_is_contiguous(op->src[0])) {
                    ਵਾਪਸ ਝੂਠ;
                }
                // large k falls back to radix-select
                ਅਚਲ uint32_t min_pipeline =
                    std::max((uint32_t) log2f(ਦਸ਼ਮਲਵ(op->ne[0])) + 1, device->subgroup_ਮਾਪ_log2);
                ਜੇ (min_pipeline < ਅੰਕ_topk_pipelines && device->pipeline_topk_f32[min_pipeline]) {
                    ਵਾਪਸ ਸੱਚ;
                }
                ਵਾਪਸ device->pipeline_topk_radix_f32 != ਖਾਲੀ_ਸੂਚਕ;
            }
        ਹਾਲਤ GGML_OP_UPSCALE:
            ਜੇ (op->op_params[0] & GGML_SCALE_FLAG_ANTIALIAS) {
                ਜੇ ((op->op_params[0] & 0xFF) != GGML_SCALE_ਬਹੁਲਕ_BILINEAR) {
                    ਵਾਪਸ ਝੂਠ;
                }
            }
            ਵਾਪਸ op->src[0]->type == GGML_TYPE_F32;
        ਹਾਲਤ GGML_OP_ACC:
            ਵਾਪਸ op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        ਹਾਲਤ GGML_OP_ਸੈੱਟ:
            ਵਾਪਸ op->src[0]->type == op->src[1]->type && op->src[0]->type == op->type &&
                   (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_I32);
        ਹਾਲਤ GGML_OP_CONCAT: {
            ਵਾਪਸ ggml_vk_concat_supported(op->src[0], op->src[1], op);
        }
        ਹਾਲਤ GGML_OP_ADD1:
            ਵਾਪਸ (op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32)
                || (op->src[0]->type == GGML_TYPE_F16 && op->src[1]->type == GGML_TYPE_F32)
                || (op->src[0]->type == GGML_TYPE_F16 && op->src[1]->type == GGML_TYPE_F16);
        ਹਾਲਤ GGML_OP_ARANGE:
            ਵਾਪਸ op->type == GGML_TYPE_F32;
        ਹਾਲਤ GGML_OP_FILL:
            ਵਾਪਸ op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16;
        ਹਾਲਤ GGML_OP_SCALE:
            ਵਾਪਸ ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        ਹਾਲਤ GGML_OP_PAD:
        ਹਾਲਤ GGML_OP_PAD_REFLECT_1D:
        ਹਾਲਤ GGML_OP_ROLL:
            ਵਾਪਸ op->src[0]->type == GGML_TYPE_F32;
        ਹਾਲਤ GGML_OP_DIAG_ਨਕਾਬ_INF:
            ਵਾਪਸ ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        ਹਾਲਤ GGML_OP_SOFT_MAX:
            ਵਾਪਸ ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32
                && (!op->src[1] || (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16));
        ਹਾਲਤ GGML_OP_SOFT_MAX_ਪਿੱਠ:
            ਵਾਪਸ ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32
                && ggml_is_contiguous(op->src[1]) && op->src[1]->type == GGML_TYPE_F32;
        ਹਾਲਤ GGML_OP_ਜੋੜ:
            ਵਾਪਸ op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguously_allocated(op->src[0]);
        ਹਾਲਤ GGML_OP_ਜੋੜ_ਪੰਕਤੀਆਂ:
        ਹਾਲਤ GGML_OP_ਔਸਤ:
            ਵਾਪਸ op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous_ਪੰਕਤੀਆਂ(op->src[0]);
        ਹਾਲਤ GGML_OP_CUMSUM:
            {
                ਜੇ (device->subgroup_arithmetic && device->subgroup_require_ਭਰਿਆ_support) {
                    ਵਾਪਸ op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous_ਪੰਕਤੀਆਂ(op->src[0]);
                }
                ਵਾਪਸ ਝੂਠ;
            }
        ਹਾਲਤ GGML_OP_DSV4_HC_COMB:
        ਹਾਲਤ GGML_OP_DSV4_HC_PRE:
        ਹਾਲਤ GGML_OP_DSV4_HC_POST:
            {
                ਜੇ (op->type != GGML_TYPE_F32) {
                    ਵਾਪਸ ਝੂਠ;
                }
                ਲਈ (uint32_t i = 0; i < GGML_MAX_SRC; ++i) {
                    ਜੇ (op->src[i] && op->src[i]->type != GGML_TYPE_F32) {
                        ਵਾਪਸ ਝੂਠ;
                    }
                }
                // hc is hardcoded to 4 in the shaders. ggml only constrains it
                // to 4 for COMB, so PRE/POST have to be checked here.
                ਜੇ (op->op == GGML_OP_DSV4_HC_PRE && op->src[0]->ne[1] != 4) {
                    ਵਾਪਸ ਝੂਠ;
                }
                ਜੇ (op->op == GGML_OP_DSV4_HC_POST && op->src[1]->ne[1] != 4) {
                    ਵਾਪਸ ਝੂਠ;
                }
                ਜੇ (op->op == GGML_OP_DSV4_HC_COMB) {
                    ਵਾਪਸ device->pipeline_dsv4_hc_comb_f32 != ਖਾਲੀ_ਸੂਚਕ;
                }
                ਵਾਪਸ ਸੱਚ;
            }
        ਹਾਲਤ GGML_OP_SOLVE_TRI:
            {
                ਜੇ (op->type != GGML_TYPE_F32 || op->src[0]->type != GGML_TYPE_F32) {
                    ਵਾਪਸ ਝੂਠ;
                }
                ਅਚਲ uint32_t N = op->src[0]->ne[0];
                ਅਚਲ uint32_t K = op->src[1]->ne[0];
                // K dimension limited to workgroup size
                ਜੇ (K > 1u << device->max_workgroup_ਮਾਪ_log2) {
                    ਵਾਪਸ ਝੂਠ;
                }
                ਅਚਲ uint32_t batch_N = device->properties.limits.max_Compute_Shared_ਸਮ੍ਰਿਤੀ_ਮਾਪ / ((N + K) * ਆਕਾਰ(ਦਸ਼ਮਲਵ));

                ਜੇ (batch_N == 0) {
                    ਵਾਪਸ ਝੂਠ;
                }
                ਵਾਪਸ ਸੱਚ;
            }
        ਹਾਲਤ GGML_OP_ARGMAX:
            ਵਾਪਸ ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        ਹਾਲਤ GGML_OP_CROSS_ENTROPY_LOSS:
            ਵਾਪਸ ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32
                && ggml_is_contiguous(op->src[1]) && op->src[1]->type == GGML_TYPE_F32
                && ggml_are_same_shape(op->src[0], op->src[1])
                && ggml_is_contiguous(op) && ggml_is_scalar(op) && op->type == GGML_TYPE_F32;
        ਹਾਲਤ GGML_OP_CROSS_ENTROPY_LOSS_ਪਿੱਠ:
            ਵਾਪਸ ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32 && ggml_is_scalar(op->src[0])
                && ggml_is_contiguous(op->src[1]) && op->src[1]->type == GGML_TYPE_F32
                && ggml_is_contiguous(op->src[2]) && op->src[2]->type == GGML_TYPE_F32
                && ggml_are_same_shape(op->src[1], op->src[2])
                && ggml_are_same_shape(op->src[1], op)
                && ggml_is_contiguous(op) && op->type == GGML_TYPE_F32;
        ਹਾਲਤ GGML_OP_ਗਿਣਤੀ_EQUAL:
            ਵਾਪਸ ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_I32
                && ggml_is_contiguous(op->src[1]) && op->src[1]->type == GGML_TYPE_I32;
        ਹਾਲਤ GGML_OP_IM2_ਖਾਨਾ:
            ਵਾਪਸ ggml_is_contiguous(op->src[1])
                && op->src[1]->type == GGML_TYPE_F32
                && (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16);
        ਹਾਲਤ GGML_OP_IM2_ਖਾਨਾ_3_D:
            ਵਾਪਸ op->src[1]->type == GGML_TYPE_F32
                && (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16);
        ਹਾਲਤ GGML_OP_TIMESTEP_EMBEDDING:
            ਵਾਪਸ op->src[0]->type == GGML_TYPE_F32;
        ਹਾਲਤ GGML_OP_CONV_2D_DW:
            ਵਾਪਸ (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16)
                && op->src[1]->type == GGML_TYPE_F32;
        ਹਾਲਤ GGML_OP_POOL_1D:
            ਵਾਪਸ ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        ਹਾਲਤ GGML_OP_POOL_2D:
            ਵਾਪਸ ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        ਹਾਲਤ GGML_OP_RWKV_WKV6:
        ਹਾਲਤ GGML_OP_RWKV_WKV7:
            ਵਾਪਸ ਸੱਚ; // all inputs are contiguous, see ggml.c
        ਹਾਲਤ GGML_OP_GATED_ਰੇਖਿਕ_ATTN:
            // the shader block size is hardcoded to head_size 64
            ਵਾਪਸ op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && op->src[0]->ne[0] == 64;
        ਹਾਲਤ GGML_OP_LIGHTNING_INDEXER:
            {
                ਅਚਲ ggml_tensor * q = op->src[0];
                ਅਚਲ ggml_tensor * k = op->src[1];
                ਅਚਲ ggml_tensor * w = op->src[2];
                ਅਚਲ ggml_tensor * m = op->src[3];

                // the q/w/m types and the shape relationships between q, k, w, m and dst
                // are already asserted in ggml_lightning_indexer()
                ਜੇ (!ggml_vk_lightning_indexer_k_type_supported(k->type) || !device->fp16) {
                    ਵਾਪਸ ਝੂਠ;
                }

                // the shader block size is hardcoded to head size 128
                ਜੇ (q->ne[0] != 128) {
                    ਵਾਪਸ ਝੂਠ;
                }

                // the shader indexes the buffers by element stride, and is dispatched
                // without allow_misalign
                ਲਈ (ਅਚਲ ggml_tensor * t : {q, k, w, m, op}) {
                    ਜੇ (t->nb[0] != ggml_type_ਮਾਪ(t->type) ||
                        (vk_tensor_ਵਿੱਥ(t) + t->view_offs) % device->properties.limits.min_Storage_ਬਫਰ_ਵਿੱਥ_Alignment != 0) {
                        ਵਾਪਸ ਝੂਠ;
                    }
                    // the strides get scaled down from bytes, so the division must be exact
                    ਲਈ (ਪੂਰਨ i = 1; i < GGML_MAX_DIMS; ++i) {
                        ਜੇ (t->nb[i] % ggml_type_ਮਾਪ(t->type) != 0) {
                            ਵਾਪਸ ਝੂਠ;
                        }
                    }
                }
                ਵਾਪਸ ਸੱਚ;
            }
        ਹਾਲਤ GGML_OP_GATED_DELTA_NET:
            {
                ਅਚਲ uint32_t S_v = op->src[2]->ne[0];
                ਜੇ (S_v != 16 && S_v != 32 && S_v != 64 && S_v != 128) {
                    ਵਾਪਸ ਝੂਠ;
                }
                ਲਈ (ਪੂਰਨ i = 0; i < 6; i++) {
                    ਜੇ (op->src[i] == ਖਾਲੀ_ਸੂਚਕ || op->src[i]->type != GGML_TYPE_F32) {
                        ਵਾਪਸ ਝੂਠ;
                    }
                }
                ਵਾਪਸ op->type == GGML_TYPE_F32;
            }
        ਹਾਲਤ GGML_OP_SSM_ਜਾਂਚ:
            {
                ਲਈ (ਪੂਰਨ i = 0; i < 6; i++) {
                    ਜੇ (op->src[i] && ggml_is_quantized(op->src[i]->type)) {
                        ਵਾਪਸ ਝੂਠ;
                    }
                }
                ਜੇ (op->src[6] && op->src[6]->type != GGML_TYPE_I32) {
                    ਵਾਪਸ ਝੂਠ;
                }
                ਜੇ (op->src[0]->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
                    ਵਾਪਸ ਝੂਠ;
                }

                ਅਚਲ uint32_t d_state = op->src[0]->ne[0];
                ਅਚਲ uint32_t ਸਿਰ_dim = op->src[0]->ne[1];

                ਤਾਰਕਿਕ is_mamba2 = (op->src[3] && op->src[3]->nb[1] == ਆਕਾਰ(ਦਸ਼ਮਲਵ));
                ਜੇ (!is_mamba2) {
                    ਵਾਪਸ ਝੂਠ;
                }

                ਜੇ ((d_state != 128 && d_state != 256) || ਸਿਰ_dim % 16 != 0) {
                    ਵਾਪਸ ਝੂਠ;
                }

                size_t shmem_ਮਾਪ = d_state * ਆਕਾਰ(ਦਸ਼ਮਲਵ);

                ਜੇ (shmem_ਮਾਪ > device->properties.limits.max_Compute_Shared_ਸਮ੍ਰਿਤੀ_ਮਾਪ) {
                    ਵਾਪਸ ਝੂਠ;
                }

                ਜੇ (!device->subgroup_basic) {
                    ਵਾਪਸ ਝੂਠ;
                }

                ਵਾਪਸ ਸੱਚ;
            }
        ਹਾਲਤ GGML_OP_SSM_CONV:
            ਵਾਪਸ op->src[0]->type == GGML_TYPE_F32;
        ਹਾਲਤ GGML_OP_CONV_TRANSPOSE_1D:
            ਵਾਪਸ op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32;
        ਹਾਲਤ GGML_OP_COL2IM_1D:
            ਵਾਪਸ (op->src[0]->type == GGML_TYPE_F32 ||
                    op->src[0]->type == GGML_TYPE_F16 ||
                    op->src[0]->type == GGML_TYPE_BF16) &&
                   op->type == op->src[0]->type &&
                   ggml_is_contiguous(op->src[0]) &&
                   ggml_is_contiguous(op);
        ਹਾਲਤ GGML_OP_CONV_2D:
        ਹਾਲਤ GGML_OP_CONV_TRANSPOSE_2D:
            {
                ਅਚਲ ਤਾਰਕਿਕ transpose = op->op == GGML_OP_CONV_TRANSPOSE_2D;
                ਅਚਲ int64_t ਬਾਹਰ = !transpose ? op->src[0]->ne[3] : op->src[0]->ne[2];
                ਅਚਲ int64_t ਅੰਦਰ  = !transpose ? op->src[0]->ne[2] : op->src[0]->ne[3];

                // Channel-contiguous format is not supported yet.
                ਵਾਪਸ ((op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                    (op->src[0]->nb[0] == ਆਕਾਰ(ਦਸ਼ਮਲਵ) || op->src[0]->nb[0] == ਆਕਾਰ(ggml_fp16_t) ) &&
                    op->src[1]->type == GGML_TYPE_F32 &&
                    op->type == GGML_TYPE_F32 &&
                    ਬਾਹਰ == op->ne[2] &&
                    ਅੰਦਰ == op->src[1]->ne[2] &&
                    ggml_is_contiguous(op->src[0]) &&
                    ggml_is_contiguous(op->src[1]) &&
                    ggml_is_contiguous(op));
            }
        ਹਾਲਤ GGML_OP_CONV_3D:
            ਵਾਪਸ (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                op->src[1]->type == GGML_TYPE_F32 &&
                op->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(op->src[0]) &&
                ggml_is_contiguous(op->src[1]) &&
                ggml_is_contiguous(op);
        ਮੂਲ:
            ਵਾਪਸ ਝੂਠ;
    }

    ਅਣਵਰਤਿਆ(dev);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_vk_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_ਬਫਰ_type_t buft) {
    ਜੇ (buft->iface.get_name != ggml_backend_vk_ਬਫਰ_type_name) {
        ਵਾਪਸ ਝੂਠ;
    }

    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    ggml_backend_vk_ਬਫਰ_type_context * buft_ctx = (ggml_backend_vk_ਬਫਰ_type_context *)buft->context;

    ਵਾਪਸ buft_ctx->device->idx == ctx->device;
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_vk_device_offload_op(ggml_backend_dev_t dev, ਅਚਲ ggml_tensor * op) {
    ggml_backend_vk_device_context * dev_ctx = (ggml_backend_vk_device_context *)dev->context;

    ਵਾਪਸ ggml_vk_get_op_batch_ਮਾਪ(op) >= dev_ctx->op_offload_min_batch_ਮਾਪ;
}

ਸਥਿਰ ggml_backend_event_t ggml_backend_vk_device_event_new(ggml_backend_dev_t dev) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    ਸਵੈਚਾਲਿਤ device = ggml_vk_get_device(ctx->device);

    vk_event *vkev = ਨਵਾਂ vk_event;
    ਜੇ (!vkev) {
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }

    // No events initially, they get created on demand
    vkev->has_event = ਝੂਠ;

    vk::SemaphoreTypeCreateInfo tci{ vk::SemaphoreType::eTimeline, 0 };
    vk::SemaphoreCreateInfo ci{};
    ci.ਸੈੱਟ_PNext(&tci);
    vkev->tl_semaphore = { device->device.createSemaphore(ci), 0 };

    ਵਾਪਸ ਨਵਾਂ ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ vkev,
    };
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_vk_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    ਸਵੈਚਾਲਿਤ device = ggml_vk_get_device(ctx->device);

    vk_event *vkev = (vk_event *)event->context;

    device->device.destroySemaphore(vkev->tl_semaphore.s);
    ਲਈ (ਸਵੈਚਾਲਿਤ& event : vkev->events_free) {
        device->device.destroyEvent(event);
    }
    ਲਈ (ਸਵੈਚਾਲਿਤ& event : vkev->events_submitted) {
        device->device.destroyEvent(event);
    }
    ਜੇ (vkev->has_event) {
        device->device.destroyEvent(vkev->event);
    }
    ਮਿਟਾਓ vkev;
    ਮਿਟਾਓ event;
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_vk_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    VK_LOG_DEBUG("ggml_backend_vk_device_event_synchronize(backend=" << dev << ", event=" << event << ")");
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    ਸਵੈਚਾਲਿਤ device = ggml_vk_get_device(ctx->device);
    vk_event *vkev = (vk_event *)event->context;

    // Only do something if the event has actually been used
    ਜੇ (vkev->has_event) {
        vk::Semaphore sem = vkev->tl_semaphore.s;
        uint64_t val = vkev->tl_semaphore.value;
        vk::SemaphoreWaitInfo swi{vk::SemaphoreWaitFlags{}, sem, val};
        VK_CHECK(device->device.waitSemaphores(swi, UINT64_MAX), "event_synchronize", device);

        // Reset and move submitted events
        ਲਈ (ਸਵੈਚਾਲਿਤ& event : vkev->events_submitted) {
            device->device.resetEvent(event);
        }
        vkev->events_free.insert(vkev->events_free.end(), vkev->events_submitted.begin(), vkev->events_submitted.end());
        vkev->events_submitted.clear();

        // Finished using current command buffer so we flag for reuse
        ਜੇ (vkev->cmd_ਬਫਰ) {
            // Only flag for reuse if it hasn't been reused already
            ਜੇ (vkev->cmd_ਬਫਰ_use_counter == vkev->cmd_ਬਫਰ->use_counter) {
                vkev->cmd_ਬਫਰ->ਵਿੱਚ_use = ਝੂਠ;
                vkev->cmd_ਬਫਰ->buf.reset();
            }
            vkev->cmd_ਬਫਰ = ਖਾਲੀ_ਸੂਚਕ;
        }
    }
}

ਸਥਿਰ ggml_backend_ਬਫਰ_t ggml_backend_vk_device_ਬਫਰ_from_host_ptr(ggml_backend_dev_t dev, ਖਾਲੀ * ptr, size_t size, size_t max_tensor_ਮਾਪ) {
    VK_LOG_DEBUG("ggml_backend_vk_device_buffer_from_host_ptr(backend=" << dev << ", ptr=" << ptr << ", size=" << size << ")");
    GGML_ਅਣਵਰਤਿਆ(max_tensor_ਮਾਪ);

    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    ਸਵੈਚਾਲਿਤ device = ggml_vk_get_device(ctx->device);

    vk_ਬਫਰ buf = ggml_vk_ਬਫਰ_from_host_ptr(device, ptr, size);

    ਜੇ (!buf) {
        ਵਾਪਸ {};
    }

    ggml_backend_vk_ਬਫਰ_context * bufctx = ਨਵਾਂ ggml_backend_vk_ਬਫਰ_context(device, std::move(buf), device->name);

    ggml_backend_ਬਫਰ_t ret = ggml_backend_ਬਫਰ_init(ggml_backend_vk_device_get_ਬਫਰ_type(dev), ggml_backend_vk_ਬਫਰ_interface, bufctx, size);

    ਵਾਪਸ ret;
}

ਸਥਿਰ ਅਚਲ ਬਣਤਰ ggml_backend_device_i ggml_backend_vk_device_i = {
    /* .get_name             = */ ggml_backend_vk_device_get_name,
    /* .get_description      = */ ggml_backend_vk_device_get_description,
    /* .get_memory           = */ ggml_backend_vk_device_get_ਸਮ੍ਰਿਤੀ,
    /* .get_type             = */ ggml_backend_vk_device_get_type,
    /* .get_props            = */ ggml_backend_vk_device_get_props,
    /* .init_backend         = */ ggml_backend_vk_device_init,
    /* .get_buffer_type      = */ ggml_backend_vk_device_get_ਬਫਰ_type,
    /* .get_host_buffer_type = */ ggml_backend_vk_device_get_host_ਬਫਰ_type,
    /* .buffer_from_host_ptr = */ ggml_backend_vk_device_ਬਫਰ_from_host_ptr,
    /* .supports_op          = */ ggml_backend_vk_device_supports_op,
    /* .supports_buft        = */ ggml_backend_vk_device_supports_buft,
    /* .offload_op           = */ ggml_backend_vk_device_offload_op,
    /* .event_new            = */ ggml_backend_vk_device_event_new,
    /* .event_free           = */ ggml_backend_vk_device_event_free,
    /* .event_synchronize    = */ ggml_backend_vk_device_event_synchronize,
};

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_vk_reg_get_name(ggml_backend_reg_t reg) {
    ਅਣਵਰਤਿਆ(reg);
    ਵਾਪਸ GGML_VK_NAME;
}

ਸਥਿਰ size_t ggml_backend_vk_reg_get_device_ਗਿਣਤੀ(ggml_backend_reg_t reg) {
    ਅਣਵਰਤਿਆ(reg);
    ਵਾਪਸ ggml_backend_vk_get_device_ਗਿਣਤੀ();
}

ਸਥਿਰ ggml_backend_dev_t ggml_backend_vk_reg_get_device(ggml_backend_reg_t reg, size_t device) {
    ਸਥਿਰ std::vector<ggml_backend_dev_t> devices;

    ਸਥਿਰ ਤਾਰਕਿਕ initialized = ਝੂਠ;

    {
        ਸਥਿਰ std::mutex mutex;
        std::lock_guard<std::mutex> ਤਾਲਾ(mutex);
        ਜੇ (!initialized) {
            ਅਚਲ ਪੂਰਨ min_batch_ਮਾਪ = getenv("GGML_OP_OFFLOAD_MIN_BATCH") ? atoi(getenv("GGML_OP_OFFLOAD_MIN_BATCH")) : 32;
            ਲਈ (ਪੂਰਨ i = 0; i < ggml_backend_vk_get_device_ਗਿਣਤੀ(); i++) {
                ggml_backend_vk_device_context * ctx = ਨਵਾਂ ggml_backend_vk_device_context;
                ਅੱਖਰ desc[256];
                ggml_backend_vk_get_device_description(i, desc, ਆਕਾਰ(desc));
                ctx->device = i;
                ctx->name = GGML_VK_NAME + std::to_string(i);
                ctx->description = desc;
                ctx->is_integrated_gpu = ggml_backend_vk_get_device_type(i) == vk::PhysicalDeviceType::eIntegratedGpu;
                ctx->pci_bus_id = ggml_backend_vk_get_device_pci_id(i);
                ctx->op_offload_min_batch_ਮਾਪ = min_batch_ਮਾਪ;
                devices.push_back(ਨਵਾਂ ggml_backend_device {
                    /* .iface   = */ ggml_backend_vk_device_i,
                    /* .reg     = */ reg,
                    /* .context = */ ctx,
                });
            }
            initialized = ਸੱਚ;
        }
    }

    GGML_ASSERT(device < devices.size());
    ਵਾਪਸ devices[device];
}

ਸਥਿਰ ਅਚਲ ਬਣਤਰ ggml_backend_reg_i ggml_backend_vk_reg_i = {
    /* .get_name         = */ ggml_backend_vk_reg_get_name,
    /* .get_device_count = */ ggml_backend_vk_reg_get_device_ਗਿਣਤੀ,
    /* .get_device       = */ ggml_backend_vk_reg_get_device,
    /* .get_proc_address = */ ਨਿਰਾ,
};

ggml_backend_reg_t ggml_backend_vk_reg() {
    ਸਥਿਰ ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_ਸੰਸਕਰਣ,
        /* .iface       = */ ggml_backend_vk_reg_i,
        /* .context     = */ ਖਾਲੀ_ਸੂਚਕ,
    };
    ਕੋਸ਼ਿਸ਼ {
        ggml_vk_instance_init();
        ਵਾਪਸ &reg;
    } ਫੜੋ (ਅਚਲ vk::SystemError& e) {
        VK_LOG_DEBUG("ggml_backend_vk_reg() -> Error: System error: " << e.what());
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    } ਫੜੋ (ਅਚਲ std::exception &e) {
        VK_LOG_DEBUG("ggml_backend_vk_reg() -> Error: " << e.what());
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    } ਫੜੋ (...) {
        VK_LOG_DEBUG("ggml_backend_vk_reg() -> Error: unknown exception during Vulkan init");
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }
}

ਤਾਰਕਿਕ ggml_vk_instance_layer_settings_ਉਪਲਬਧ() {
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_VULKAN_VALIDATE
    // Check if validation layer provides the extension
    ਅਚਲ std::string layer_name = "VK_LAYER_KHRONOS_validation";
    ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ& layer : vk::enumerateInstanceLayerProperties()) {
        ਜੇ (layer_name == layer.layerName.ਅੰਕੜੇ()) {
            ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ& ext : vk::enumerateInstanceExtensionProperties(layer_name)) {
                ਜੇ (strcmp("VK_EXT_layer_settings", ext.extensionName.ਅੰਕੜੇ()) == 0) {
                    ਵਾਪਸ ਸੱਚ;
                }
            }
        }
    }

    std::cerr << "ggml_vulkan: WARNING: Validation layer or layer extension VK_EXT_layer_settings not found." << std::ਸਤਰ_ਅੰਤ;
#ਜੇ_ਅੰਤ
    ਵਾਪਸ ਝੂਠ;
}

ਤਾਰਕਿਕ ggml_vk_instance_portability_enumeration_ext_ਉਪਲਬਧ(ਅਚਲ std::vector<vk::ExtensionProperties>& instance_extensions) {
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ __APPLE__
    // Check for portability enumeration extension for MoltenVK support
    ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ& properties : instance_extensions) {
        ਜੇ (strcmp("VK_KHR_portability_enumeration", properties.extensionName) == 0) {
            ਵਾਪਸ ਸੱਚ;
        }
    }
    std::cerr << "ggml_vulkan: WARNING: Instance extension VK_KHR_portability_enumeration not found." << std::ਸਤਰ_ਅੰਤ;
#ਜੇ_ਅੰਤ
    ਵਾਪਸ ਝੂਠ;

    ਅਣਵਰਤਿਆ(instance_extensions);
}

ਤਾਰਕਿਕ ggml_vk_instance_debug_utils_ext_ਉਪਲਬਧ(
    ਅਚਲ std::vector<vk::ExtensionProperties> & instance_extensions) {
    // Check for portability enumeration extension for MoltenVK support
    ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & properties : instance_extensions) {
        ਜੇ (strcmp("VK_EXT_debug_utils", properties.extensionName) == 0) {
            ਵਾਪਸ ਸੱਚ;
        }
    }

    std::cerr << "ggml_vulkan: WARNING: Instance extension VK_EXT_debug_utils not found." << std::ਸਤਰ_ਅੰਤ;
    ਵਾਪਸ ਝੂਠ;

    ਅਣਵਰਤਿਆ(instance_extensions);
}

ਤਾਰਕਿਕ ggml_vk_device_is_supported(ਅਚਲ vk::PhysicalDevice & vkdev) {
    VkPhysicalDeviceFeatures2 device_features2;
    device_features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;

    VkPhysicalDeviceVulkan11Features vk11_features;
    vk11_features.p_ਅਗਲਾ = ਖਾਲੀ_ਸੂਚਕ;
    vk11_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    device_features2.p_ਅਗਲਾ = &vk11_features;

    vkGetPhysicalDeviceFeatures2(vkdev, &device_features2);

    ਵਾਪਸ vk11_features.storage_Buffer16_ਦੋਆਧਾਰੀ_ਅੰਕ_Access;
}

ਤਾਰਕਿਕ ggml_vk_khr_cooperative_ਮੈਟ੍ਰਿਕਸ_support(ਅਚਲ vk::PhysicalDeviceProperties& props, ਅਚਲ vk::PhysicalDeviceDriverProperties& driver_props, vk_device_architecture arch) {
    ਚੋਣ (props.vendorID) {
    ਹਾਲਤ VK_VENDOR_ID_INTEL:
        // Only allowing Xe2/Xe3 GPU and integrated Xe GPUs at the moment since older hardware (ex. Arc A770) has performance regressions.
        ਵਾਪਸ (arch == vk_device_architecture::INTEL_XE2) ||
            (arch == vk_device_architecture::INTEL_XE1 && props.deviceType == vk::PhysicalDeviceType::eIntegratedGpu && driver_props.driverID == vk::DriverId::eIntelProprietaryWindows);
    ਹਾਲਤ VK_VENDOR_ID_AMD:
        ਜੇ (driver_props.driverID == vk::DriverId::eAmdProprietary || driver_props.driverID == vk::DriverId::e_Amd_Open_ਸਰੋਤ) {
            // Workaround for AMD proprietary driver reporting support on all GPUs
            ਵਾਪਸ arch == vk_device_architecture::AMD_RDNA3 || arch == vk_device_architecture::AMD_RDNA4;
        }
        ਵਾਪਸ ਸੱਚ;
    ਹਾਲਤ VK_VENDOR_ID_QUALCOMM:
        // Only allow Adreno GPUs with hardware matrix cores (Gen 6+).
        ਵਾਪਸ arch == vk_device_architecture::QUALCOMM_ADRENO;
    ਮੂਲ:
        ਵਾਪਸ ਸੱਚ;
    }
}

uint32_t ggml_vk_intel_shader_core_ਗਿਣਤੀ(ਅਚਲ vk::PhysicalDevice& vkdev) {
    VkPhysicalDeviceProperties2 props = vkdev.getProperties2();

    ਜੇ (props.properties.vendorID != VK_VENDOR_ID_INTEL) {
        ਵਾਪਸ 0;
    }

    ਅਚਲ uint32_t device_id = props.properties.deviceID;

    ਚੋਣ (device_id) {
    ਹਾਲਤ 0x56A6:  // A310
        ਵਾਪਸ 6;
    ਹਾਲਤ 0x5693:  // A370M
    ਹਾਲਤ 0x56A5:  // A380
    ਹਾਲਤ 0x56B1:  // Pro A40/A50
        ਵਾਪਸ 8;
    ਹਾਲਤ 0x5697:  // A530M
        ਵਾਪਸ 12;
    ਹਾਲਤ 0x5692:  // A550M
    ਹਾਲਤ 0x56B3:  // Pro A60
        ਵਾਪਸ 16;
    ਹਾਲਤ 0x56A2:  // A580
        ਵਾਪਸ 24;
    ਹਾਲਤ 0x5691:  // A730M
    ਹਾਲਤ 0x56A1:  // A750
        ਵਾਪਸ 28;
    ਹਾਲਤ 0x56A0:  // A770
    ਹਾਲਤ 0x5690:  // A770M
        ਵਾਪਸ 32;
    ਹਾਲਤ 0xE212:  // Pro B50
        ਵਾਪਸ 16;
    ਹਾਲਤ 0xE20C:  // B570
        ਵਾਪਸ 18;
    ਹਾਲਤ 0xE20B:  // B580
    ਹਾਲਤ 0xE211:  // Pro B60
    ਹਾਲਤ 0xE222:  // Pro B65
        ਵਾਪਸ 20;
    ਹਾਲਤ 0xE223:  // Pro B70
        ਵਾਪਸ 32;
    ਹਾਲਤ 0xB080:  // PTL Xe3 LPG 2x6 (12 subslices)
        ਵਾਪਸ 12;
    ਮੂਲ:
        ਵਾਪਸ 0;
    }
}

ਤਾਰਕਿਕ ggml_vk_intel_windows_driver_ਵਿੱਚ_range(uint32_t driver_ਸੰਸਕਰਣ, uint32_t ਹੇਠਲਾ_major, uint32_t ਹੇਠਲਾ_minor, uint32_t ਉੱਤਲਾ_major, uint32_t ਉੱਤਲਾ_minor) {
#ਜੇ defined(_WIN32)
    // Intel Windows encodes xxx.yyyy as [31:14].[13:0].
    ਅਚਲ uint32_t major = driver_ਸੰਸਕਰਣ >> 14;
    ਅਚਲ uint32_t minor = driver_ਸੰਸਕਰਣ & 0x3fff;

    ਅਚਲ ਤਾਰਕਿਕ ge_ਹੇਠਲਾ = major > ਹੇਠਲਾ_major || (major == ਹੇਠਲਾ_major && minor >= ਹੇਠਲਾ_minor);
    ਅਚਲ ਤਾਰਕਿਕ lt_ਉੱਤਲਾ = major < ਉੱਤਲਾ_major || (major == ਉੱਤਲਾ_major && minor < ਉੱਤਲਾ_minor);

    ਵਾਪਸ ge_ਹੇਠਲਾ && lt_ਉੱਤਲਾ;
#ਨਹੀਂਤਾਂ
    GGML_ਅਣਵਰਤਿਆ(driver_ਸੰਸਕਰਣ);
    GGML_ਅਣਵਰਤਿਆ(ਹੇਠਲਾ_major);
    GGML_ਅਣਵਰਤਿਆ(ਹੇਠਲਾ_minor);
    GGML_ਅਣਵਰਤਿਆ(ਉੱਤਲਾ_major);
    GGML_ਅਣਵਰਤਿਆ(ਉੱਤਲਾ_minor);
    ਵਾਪਸ ਸੱਚ;
#ਜੇ_ਅੰਤ
}

GGML_BACKEND_DL_IMPL(ggml_backend_vk_reg)


// out-of-lined header method definitions

ਖਾਲੀ vk_ਕਤਾਰ_handle_synchronized::submit(vk::ArrayProxy<ਅਚਲ vk::SubmitInfo> submits, vk::Fence fence) {
    // Workaround for NVIDIA driver bug
    std::unique_lock<std::mutex> device_guard;
    ਜੇ (device_submit_mutex) {
        device_guard = std::unique_lock<std::mutex>(*device_submit_mutex);
    }
    std::lock_guard<std::mutex> guard(mutex);
    ਕੋਸ਼ਿਸ਼ {
        queue.submit(submits, fence);
    } ਫੜੋ (vk::Device_ਗੁਆਚਾ_Error &) {
        ਜੇ (ਸਵੈਚਾਲਿਤ dev = device.ਤਾਲਾ()) {
            ggml_vk_print_device_ਗੁਆਚਾ_info(dev);
        }
        ਸੁੱਟੋ;
    }
}

ਖਾਲੀ vk_ਕਤਾਰ_handle_unsynchronized::submit(vk::ArrayProxy<ਅਚਲ vk::SubmitInfo> submits, vk::Fence fence) {
    // Workaround for NVIDIA driver bug
    std::unique_lock<std::mutex> device_guard;
    ਜੇ (device_submit_mutex) {
        device_guard = std::unique_lock<std::mutex>(*device_submit_mutex);
    }
    ਕੋਸ਼ਿਸ਼ {
        queue.submit(submits, fence);
    } ਫੜੋ (vk::Device_ਗੁਆਚਾ_Error &) {
        ਜੇ (ਸਵੈਚਾਲਿਤ dev = device.ਤਾਲਾ()) {
            ggml_vk_print_device_ਗੁਆਚਾ_info(dev);
        }
        ਸੁੱਟੋ;
    }
}

vk_device_struct::~vk_device_struct() {
    // the build failed before the device was created, nothing to destroy
    ਜੇ (device == VK_NULL_HANDLE) {
        ਵਾਪਸ;
    }

    VK_LOG_DEBUG("destroy device " << name);

    device.destroyFence(fence);

    ggml_vk_destroy_ਬਫਰ(sync_staging);

    ਜੇ (compute_ਕਤਾਰ) compute_ਕਤਾਰ->cmd_pool.destroy(device);
    ਜੇ (transfer_ਕਤਾਰ) transfer_ਕਤਾਰ->cmd_pool.destroy(device);

    // Explicitly clear to ensure queues drop their shared_ptrs to handles
    // before the Vulkan logical device instance is destroyed
    compute_ਕਤਾਰ.reset();
    transfer_ਕਤਾਰ.reset();

    ਲਈ (ਸਵੈਚਾਲਿਤ& pipeline : all_pipelines) {
        ਜੇ (pipeline.expired()) {
            ਜਾਰੀ;
        }

        vk_pipeline pl = pipeline.ਤਾਲਾ();
        ggml_vk_destroy_pipeline(device, pl);
    }
    all_pipelines.clear();

    device.destroy_Descriptor_ਸੈੱਟ_Layout(dsl);

    device.destroy();
}

ਖਾਲੀ vk_perf_logger::print_timings(ਤਾਰਕਿਕ force) {
    ਜੇ (timings.ਖਾਲੀ_ਥਾਵਾਂ()) {
        ਵਾਪਸ;
    }
    print_ਗਿਣਤੀ++;
    ਜੇ ((print_ਗਿਣਤੀ % vk_perf_logger_frequency) != 0 && !force) {
        ਵਾਪਸ;
    }
    print_ਗਿਣਤੀ = 0;
    uint64_t ਕੁੱਲ_all_op_times = 0;
    std::cerr << "----------------\nVulkan Timings:" << std::ਸਤਰ_ਅੰਤ;
    ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & t : timings) {
        uint64_t ਕੁੱਲ_op_times = 0;
        ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & ਸਮਾਂ : t.second) {
            ਕੁੱਲ_op_times += ਸਮਾਂ;
        }
        std::cerr << t.ਪਹਿਲਾ << ": " << t.second.size() << " x " << (ਕੁੱਲ_op_times / t.second.size() / 1000.0)
                  << " us = " << (ਕੁੱਲ_op_times / 1000.0) << " us";

        // If we have as many flops entries as timing entries for the op, then compute and log the flops/S.
        ਸਵੈਚਾਲਿਤ it = flops.find(t.ਪਹਿਲਾ);
        ਜੇ (it != flops.end() && (it->second).size() == t.second.size()) {
            uint64_t ਕੁੱਲ_op_flops = 0;
            ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & elem : it->second) {
                ਕੁੱਲ_op_flops += elem;
            }
            std::cerr << " ("
                      << (ਦੁੱਗਣਾ(ਕੁੱਲ_op_flops) / (1000.0 * 1000.0 * 1000.0)) /
                             (ਦੁੱਗਣਾ(ਕੁੱਲ_op_times) / (1000.0 * 1000.0 * 1000.0))
                      << " GFLOPS/s)";
        }

        ਕੁੱਲ_all_op_times += ਕੁੱਲ_op_times;

        std::cerr << std::ਸਤਰ_ਅੰਤ;
    }

    ਜੇ (timings.size() > 0) {
        std::cerr << "Total time: " << ਕੁੱਲ_all_op_times / 1000.0 << " us." << std::ਸਤਰ_ਅੰਤ;
    }

    timings.clear();
    flops.clear();
}

std::string vk_perf_logger::get_ਗੰਢ_fusion_name(ਅਚਲ ggml_tensor * ਗੰਢ, ਅਚਲ ਅੱਖਰ *fusion_name, uint64_t *n_flops) {
    *n_flops = ggml_vk_get_ਗੰਢ_flops(ਗੰਢ);
    std::string fusion_str;
    ਜੇ (fusion_name) {
        fusion_str = fusion_name + std::string(" ");
    }
    ਜੇ (ਗੰਢ->op == GGML_OP_UNARY) {
        ਵਾਪਸ fusion_str + ggml_unary_op_name(ggml_get_unary_op(ਗੰਢ));
    }
    ਜੇ (ਗੰਢ->op == GGML_OP_MUL_MAT || ਗੰਢ->op == GGML_OP_MUL_MAT_ID) {
        ਅਚਲ uint64_t m     = ਗੰਢ->ne[0];
        ਅਚਲ uint64_t n     = ਗੰਢ->ne[1];
        ਅਚਲ uint64_t k     = ਗੰਢ->src[1]->ne[0];
        ਅਚਲ uint64_t batch = ਗੰਢ->ne[2] * ਗੰਢ->ne[3];
        std::string    name  = ggml_op_name(ਗੰਢ->op);
        ਜੇ ((ਗੰਢ->op == GGML_OP_MUL_MAT && n <= mul_mat_vec_max_ਖਾਨੇ_ਗਿਣਤੀ) ||
            (ਗੰਢ->op == GGML_OP_MUL_MAT_ID && ਗੰਢ->src[2]->ne[1] == 1)) {
            name += "_VEC";
        }
        name += " ";
        name += ggml_type_name(ਗੰਢ->src[0]->type);
        name += " m=" + std::to_string(m) + " n=" + std::to_string(n) + " k=" + std::to_string(k);
        ਜੇ (ਗੰਢ->op == GGML_OP_MUL_MAT_ID) {
            name += " n_expert=" + std::to_string(ਗੰਢ->src[0]->ne[2]);
        }
        ਜੇ (batch > 1) {
            name += " batch=" + std::to_string(batch);
        }
        ਵਾਪਸ fusion_str + name;
    }
    ਜੇ (ਗੰਢ->op == GGML_OP_CONV_2D || ਗੰਢ->op == GGML_OP_CONV_TRANSPOSE_2D) {
        std::string   name    = ggml_op_name(ਗੰਢ->op);
        ਅਚਲ ggml_tensor * knl = ਗੰਢ->src[0];
        uint64_t      Cout    = ਗੰਢ->ne[2];
        uint64_t      ਮਾਪ_K  = ਗੰਢ->src[1]->ne[2] * knl->ne[0] * knl->ne[1];
        uint64_t      ਮਾਪ_N_2  = ਗੰਢ->ne[3] * ਗੰਢ->ne[0] * ਗੰਢ->ne[1];
        name += " M=Cout=" + std::to_string(Cout) + ", K=Cin*KW*KH=" + std::to_string(ਮਾਪ_K) +
                ", N=N*OW*OH=" + std::to_string(ਮਾਪ_N_2);
        ਵਾਪਸ fusion_str + name;
    }
    ਜੇ (ਗੰਢ->op == GGML_OP_RMS_NORM) {
        std::string   name    = ggml_op_name(ਗੰਢ->op);
        name += "(" + std::to_string(ਗੰਢ->ne[0]) + "," + std::to_string(ਗੰਢ->ne[1]) + "," + std::to_string(ਗੰਢ->ne[2]) + "," + std::to_string(ਗੰਢ->ne[3]) + ")";
        ਵਾਪਸ fusion_str + name;
    }
    ਜੇ (ਗੰਢ->op == GGML_OP_FLASH_ATTN_EXT) {
        ਅਚਲ ggml_tensor * dst = ਗੰਢ;
        ਅਚਲ ggml_tensor * q = ਗੰਢ->src[0];
        ਅਚਲ ggml_tensor * k = ਗੰਢ->src[1];
        ਅਚਲ ggml_tensor * v = ਗੰਢ->src[2];
        ਅਚਲ ggml_tensor * m = ਗੰਢ->src[3];
        std::stringstream name;
        name << fusion_str;
        name << ggml_op_name(ਗੰਢ->op) <<
            " dst(" << dst->ne[0] << "," << dst->ne[1] << "," << dst->ne[2] << "," << dst->ne[3] << "), " <<
            " q(" << q->ne[0] << "," << q->ne[1] << "," << q->ne[2] << "," << q->ne[3] << "), " <<
            " k(" << k->ne[0] << "," << k->ne[1] << "," << k->ne[2] << "," << k->ne[3] << "), " <<
            " v(" << v->ne[0] << "," << v->ne[1] << "," << v->ne[2] << "," << v->ne[3] << "), " <<
            " m(" << (m?m->ne[0]:0) << "," << (m?m->ne[1]:0) << "," << (m?m->ne[2]:0) << "," << (m?m->ne[3]:0) << ")";
        ਵਾਪਸ name.str();
    }
    ਜੇ (ਗੰਢ->op == GGML_OP_ਸਿਖਰ_K) {
        std::stringstream name;
        name << fusion_str;
        name << ggml_op_name(ਗੰਢ->op) <<
            " K=" << ਗੰਢ->ne[0] <<
            " (" << ਗੰਢ->src[0]->ne[0] << "," << ਗੰਢ->src[0]->ne[1] << "," << ਗੰਢ->src[0]->ne[2] << "," << ਗੰਢ->src[0]->ne[3] << ")";
        ਵਾਪਸ name.str();
    }
    ਵਾਪਸ fusion_str + ggml_op_name(ਗੰਢ->op);
}

ggml_backend_vk_ਬਫਰ_context::~ggml_backend_vk_ਬਫਰ_context() {
    ggml_vk_destroy_ਬਫਰ(dev_ਬਫਰ);
}

ggml_vk_debug_label::ggml_vk_debug_label(vk_context & ctx, ਅਚਲ std::string & pipeline_name, uint32_t wg0, uint32_t wg1, uint32_t wg2) {
    ਜੇ (!vk_instance.debug_utils_support || ctx->s == ਖਾਲੀ_ਸੂਚਕ) {
        ਵਾਪਸ;
    }
    begin(ctx, pipeline_name + " (" + std::to_string(wg0) + "," + std::to_string(wg1) + "," + std::to_string(wg2) + ")");
}

ggml_vk_debug_label::ggml_vk_debug_label(vk_context & ctx, ਅਚਲ ggml_cgraph * cgraph, ਪੂਰਨ ਗੰਢ_idx, ਪੂਰਨ n_fused) {
    ਜੇ (!vk_instance.debug_utils_support || ctx->s == ਖਾਲੀ_ਸੂਚਕ) {
        ਵਾਪਸ;
    }
    std::string name = ggml_op_name(cgraph->nodes[ਗੰਢ_idx]->op);
    ਲਈ (ਪੂਰਨ i = 1; i <= n_fused; i++) {
        name += "+";
        name += ggml_op_name(cgraph->nodes[ਗੰਢ_idx + i]->op);
    }
    name += " ";
    name += cgraph->nodes[ਗੰਢ_idx]->name;
    begin(ctx, name);
}

ggml_vk_debug_label::ggml_vk_debug_label(vk_ਕਤਾਰ_handle * handle, ਅਚਲ ਅੱਖਰ * name) {
    ਜੇ (!vk_instance.debug_utils_support || handle == ਖਾਲੀ_ਸੂਚਕ) {
        ਵਾਪਸ;
    }
    vk::DebugUtilsLabelEXT label = {};
    label.pLabelName = name;
    label.color = std::array<ਦਸ਼ਮਲਵ, 4>{1.0f, 1.0f, 1.0f, 1.0f};

    qhandle = handle;
    std::lock_guard<vk_ਕਤਾਰ_handle> guard(*qhandle);
    vk_instance.pfn_vk_ਕਤਾਰ_Begin_Debug_Utils_Label_EXT(qhandle->queue, reinterpret_cast<VkDebugUtilsLabelEXT *>(&label));
}

ਖਾਲੀ ggml_vk_debug_label::close() {
    ਜੇ (subctx != ਖਾਲੀ_ਸੂਚਕ) {
        // close on the current command buffer, which may differ from the one begin used
        ਜੇ (subctx->s != ਖਾਲੀ_ਸੂਚਕ) {
            vk_instance.pfn_vk_Cmd_ਸਿਰਾ_Debug_Utils_Label_EXT(subctx->s->ਬਫਰ->buf);
        }
        subctx->debug_labels.pop_back();
        subctx = ਖਾਲੀ_ਸੂਚਕ;
    }
    ਜੇ (qhandle != ਖਾਲੀ_ਸੂਚਕ) {
        std::lock_guard<vk_ਕਤਾਰ_handle> guard(*qhandle);
        vk_instance.pfn_vk_ਕਤਾਰ_ਸਿਰਾ_Debug_Utils_Label_EXT(qhandle->queue);
        qhandle = ਖਾਲੀ_ਸੂਚਕ;
    }
}

ਖਾਲੀ ggml_vk_debug_label::begin(vk_context & ctx, ਅਚਲ std::string & name) {
    ਜੇ (!vk_instance.debug_utils_support || ctx->s == ਖਾਲੀ_ਸੂਚਕ) {
        ਵਾਪਸ;
    }
    subctx = ctx.get();
    subctx->debug_labels.push_back(name);
    ggml_vk_cmd_label_begin(subctx->s->ਬਫਰ->buf, subctx->debug_labels.back().c_str());
}

