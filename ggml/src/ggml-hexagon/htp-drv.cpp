#अन्तर्भाव <filesystem>
#अन्तर्भाव <set>
#अन्तर्भाव <sstream>
#अन्तर्भाव <string>

#यदि_परिभाषितम् _WIN32
#   define WIN32_LEAN_AND_माध्यम्
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#   include <winevt.h>
#अन्यथा
#   include <dlfcn.h>
#   include <unistd.h>
#यदि_अन्तः

#निर्देशः clang diagnostic ignored "-Wgnu-anonymous-struct"
#निर्देशः clang diagnostic ignored "-Wmissing-prototypes"
#निर्देशः clang diagnostic ignored "-Wsign-compare"
#निर्देशः clang diagnostic ignored "-Wlanguage-extension-token"
#निर्देशः clang diagnostic ignored "-Wmicrosoft-enum-value"
#निर्देशः clang diagnostic ignored "-Wnested-anon-types"

#अन्तर्भाव "ggml-impl.h"
#अन्तर्भाव "htp-drv.h"
#अन्तर्भाव "libdl.h"

#अन्तर्भाव <domain.h>

//
// Driver API types
//

प्रकारनिर्देश शून्य * (*rpcmem_alloc_pfn_t)(पूर्णाङ्क heapid, uint32_t flags, पूर्णाङ्क size);
प्रकारनिर्देश शून्य * (*rpcmem_alloc2_pfn_t)(पूर्णाङ्क heapid, uint32_t flags, size_t size);
प्रकारनिर्देश शून्य   (*rpcmem_free_pfn_t)(शून्य * po);
प्रकारनिर्देश पूर्णाङ्क    (*rpcmem_गन्तव्यम्_fd_pfn_t)(शून्य * po);

प्रकारनिर्देश AEEResult (*dspqueue_create_pfn_t)(पूर्णाङ्क                 domain,
                                           uint32_t            flags,
                                           uint32_t            req_queue_आकार,
                                           uint32_t            resp_queue_आकार,
                                           dspqueue_callback_t packet_callback,
                                           dspqueue_callback_t error_callback,
                                           शून्य *              callback_context,
                                           dspqueue_t *        queue);
प्रकारनिर्देश AEEResult (*dspqueue_close_pfn_t)(dspqueue_t queue);
प्रकारनिर्देश AEEResult (*dspqueue_export_pfn_t)(dspqueue_t queue, uint64_t *queue_id);
प्रकारनिर्देश AEEResult (*dspqueue_write_pfn_t)(dspqueue_t queue, uint32_t flags,
                                          uint32_t अङ्कः_buffers,
                                          संरचना dspqueue_बफर *buffers,
                                          uint32_t सन्देशः_दीर्घता,
                                          ध्रुव uint8_t *सन्देशः,
                                          uint32_t timeout_us);
प्रकारनिर्देश AEEResult (*dspqueue_read_pfn_t)(dspqueue_t queue, uint32_t *flags,
                                         uint32_t max_buffers, uint32_t *अङ्कः_buffers,
                                         संरचना dspqueue_बफर *buffers,
                                         uint32_t max_सन्देशः_दीर्घता,
                                         uint32_t *सन्देशः_दीर्घता, uint8_t *सन्देशः,
                                         uint32_t timeout_us);
प्रकारनिर्देश AEEResult (*dspqueue_read_noblock_pfn_t)(dspqueue_t queue, uint32_t *flags,
                                                 uint32_t max_buffers, uint32_t *अङ्कः_buffers,
                                                 संरचना dspqueue_बफर *buffers,
                                                 uint32_t max_सन्देशः_दीर्घता,
                                                 uint32_t *सन्देशः_दीर्घता, uint8_t *सन्देशः);
प्रकारनिर्देश पूर्णाङ्क (*fastrpc_mmap_pfn_t)(पूर्णाङ्क domain, पूर्णाङ्क fd, शून्य *addr, पूर्णाङ्क विस्थापनम्, size_t दीर्घता, गणना fastrpc_map_flags flags);
प्रकारनिर्देश पूर्णाङ्क (*fastrpc_munmap_pfn_t)(पूर्णाङ्क domain, पूर्णाङ्क fd, शून्य *addr, size_t दीर्घता);

प्रकारनिर्देश पूर्णाङ्क (*remote_handle64_open_pfn_t)(ध्रुव वर्ण* name, remote_handle64 *ph);
प्रकारनिर्देश पूर्णाङ्क (*remote_handle64_invoke_pfn_t)(remote_handle64 h, uint32_t dwScalars, remote_arg *pra);
प्रकारनिर्देश पूर्णाङ्क (*remote_handle64_close_pfn_t)(remote_handle h);
प्रकारनिर्देश पूर्णाङ्क (*remote_handle_control_pfn_t)(uint32_t req, शून्य* दत्त, uint32_t datalen);
प्रकारनिर्देश पूर्णाङ्क (*remote_handle64_control_pfn_t)(remote_handle64 h, uint32_t req, शून्य* दत्त, uint32_t datalen);
प्रकारनिर्देश पूर्णाङ्क (*remote_session_control_pfn_t)(uint32_t req, शून्य *दत्त, uint32_t datalen);
प्रकारनिर्देश पूर्णाङ्क (*remote_system_request_pfn_t)(system_req_payload * req);

//
// Driver API pfns
//

rpcmem_alloc_pfn_t  rpcmem_alloc_pfn  = शून्यसूचकः;
rpcmem_alloc2_pfn_t rpcmem_alloc2_pfn = शून्यसूचकः;
rpcmem_free_pfn_t   rpcmem_free_pfn   = शून्यसूचकः;
rpcmem_गन्तव्यम्_fd_pfn_t  rpcmem_गन्तव्यम्_fd_pfn  = शून्यसूचकः;

fastrpc_mmap_pfn_t   fastrpc_mmap_pfn   = शून्यसूचकः;
fastrpc_munmap_pfn_t fastrpc_munmap_pfn = शून्यसूचकः;

dspqueue_create_pfn_t       dspqueue_create_pfn       = शून्यसूचकः;
dspqueue_close_pfn_t        dspqueue_close_pfn        = शून्यसूचकः;
dspqueue_export_pfn_t       dspqueue_export_pfn       = शून्यसूचकः;
dspqueue_write_pfn_t        dspqueue_write_pfn        = शून्यसूचकः;
dspqueue_read_pfn_t         dspqueue_read_pfn         = शून्यसूचकः;
dspqueue_read_noblock_pfn_t dspqueue_read_noblock_pfn = शून्यसूचकः;

remote_handle64_open_pfn_t    remote_handle64_open_pfn    = शून्यसूचकः;
remote_handle64_invoke_pfn_t  remote_handle64_invoke_pfn  = शून्यसूचकः;
remote_handle64_close_pfn_t   remote_handle64_close_pfn   = शून्यसूचकः;
remote_handle_control_pfn_t   remote_handle_control_pfn   = शून्यसूचकः;
remote_handle64_control_pfn_t remote_handle64_control_pfn = शून्यसूचकः;
remote_session_control_pfn_t  remote_session_control_pfn  = शून्यसूचकः;
remote_system_request_pfn_t   remote_system_request_pfn   = शून्यसूचकः;

//
// Driver API
//

शून्य * rpcmem_alloc(पूर्णाङ्क heapid, uint32_t flags, पूर्णाङ्क size) {
    प्रत्यागम rpcmem_alloc_pfn(heapid, flags, size);
}

शून्य * rpcmem_alloc2(पूर्णाङ्क heapid, uint32_t flags, size_t size) {
    यदि (rpcmem_alloc2_pfn) {
        प्रत्यागम rpcmem_alloc2_pfn(heapid, flags, size);
    } अन्यथा {
        GGML_LOG_INFO("ggml-hex: rpcmem_alloc2 not found, falling back to rpcmem_alloc\n");
        प्रत्यागम rpcmem_alloc_pfn(heapid, flags, size);
    }
}

शून्य rpcmem_free(शून्य * po) {
    प्रत्यागम rpcmem_free_pfn(po);
}

पूर्णाङ्क rpcmem_गन्तव्यम्_fd(शून्य * po) {
    प्रत्यागम rpcmem_गन्तव्यम्_fd_pfn(po);
}

HTPDRV_API पूर्णाङ्क fastrpc_mmap(पूर्णाङ्क domain, पूर्णाङ्क fd, शून्य * addr, पूर्णाङ्क विस्थापनम्, size_t दीर्घता, गणना fastrpc_map_flags flags) {
    प्रत्यागम fastrpc_mmap_pfn(domain, fd, addr, विस्थापनम्, दीर्घता, flags);
}

HTPDRV_API पूर्णाङ्क fastrpc_munmap(पूर्णाङ्क domain, पूर्णाङ्क fd, शून्य * addr, size_t दीर्घता) {
    प्रत्यागम fastrpc_munmap_pfn(domain, fd, addr, दीर्घता);
}

AEEResult dspqueue_create(पूर्णाङ्क                 domain,
                          uint32_t            flags,
                          uint32_t            req_queue_आकार,
                          uint32_t            resp_queue_आकार,
                          dspqueue_callback_t packet_callback,
                          dspqueue_callback_t error_callback,
                          शून्य *              callback_context,
                          dspqueue_t *        queue) {
    प्रत्यागम dspqueue_create_pfn(domain, flags, req_queue_आकार, resp_queue_आकार, packet_callback, error_callback,
                               callback_context, queue);
}

AEEResult dspqueue_close(dspqueue_t queue) {
    प्रत्यागम dspqueue_close_pfn(queue);
}

AEEResult dspqueue_export(dspqueue_t queue, uint64_t * queue_id) {
    प्रत्यागम dspqueue_export_pfn(queue, queue_id);
}

AEEResult dspqueue_write(dspqueue_t               queue,
                         uint32_t                 flags,
                         uint32_t                 अङ्कः_buffers,
                         संरचना dspqueue_बफर * buffers,
                         uint32_t                 सन्देशः_दीर्घता,
                         ध्रुव uint8_t *          सन्देशः,
                         uint32_t                 timeout_us) {
    प्रत्यागम dspqueue_write_pfn(queue, flags, अङ्कः_buffers, buffers, सन्देशः_दीर्घता, सन्देशः, timeout_us);
}

AEEResult dspqueue_read(dspqueue_t               queue,
                        uint32_t *               flags,
                        uint32_t                 max_buffers,
                        uint32_t *               अङ्कः_buffers,
                        संरचना dspqueue_बफर * buffers,
                        uint32_t                 max_सन्देशः_दीर्घता,
                        uint32_t *               सन्देशः_दीर्घता,
                        uint8_t *                सन्देशः,
                        uint32_t                 timeout_us) {
#यदि_परिभाषितम् _WIN32
    यदि (timeout_us == 0) {
        प्रत्यागम dspqueue_read_noblock_pfn(queue, flags, max_buffers, अङ्कः_buffers, buffers, max_सन्देशः_दीर्घता,
                                     सन्देशः_दीर्घता, सन्देशः);
    }
#यदि_अन्तः
    प्रत्यागम dspqueue_read_pfn(queue, flags, max_buffers, अङ्कः_buffers, buffers, max_सन्देशः_दीर्घता, सन्देशः_दीर्घता,
                             सन्देशः, timeout_us);
}

HTPDRV_API पूर्णाङ्क remote_handle64_open(ध्रुव वर्ण * name, remote_handle64 * ph) {
    प्रत्यागम remote_handle64_open_pfn(name, ph);
}

HTPDRV_API पूर्णाङ्क remote_handle64_invoke(remote_handle64 h, uint32_t dwScalars, remote_arg * pra) {
    प्रत्यागम remote_handle64_invoke_pfn(h, dwScalars, pra);
}

HTPDRV_API पूर्णाङ्क remote_handle64_close(remote_handle64 h) {
    प्रत्यागम remote_handle64_close_pfn(h);
}

HTPDRV_API पूर्णाङ्क remote_handle_control(uint32_t req, शून्य * दत्त, uint32_t datalen) {
    प्रत्यागम remote_handle_control_pfn(req, दत्त, datalen);
}

HTPDRV_API पूर्णाङ्क remote_handle64_control(remote_handle64 h, uint32_t req, शून्य * दत्त, uint32_t datalen) {
    प्रत्यागम remote_handle64_control_pfn(h, req, दत्त, datalen);
}

HTPDRV_API पूर्णाङ्क remote_session_control(uint32_t req, शून्य * दत्त, uint32_t datalen) {
    प्रत्यागम remote_session_control_pfn(req, दत्त, datalen);
}

HTPDRV_API पूर्णाङ्क remote_system_request(system_req_payload * req) {
    यदि (!remote_system_request_pfn) {
        प्रत्यागम AEE_EUNSUPPORTEDAPI;
    }
    प्रत्यागम remote_system_request_pfn(req);
}

#यदि_परिभाषितम् _WIN32

स्थिर std::string wstr_गन्तव्यम्_str(std::wstring_view wstr) {
    std::string परिणाम;
    यदि (wstr.रिक्तानि()) {
        प्रत्यागम परिणाम;
    }
    स्वचालित अष्टकानि_needed = Wide_Char_गन्तव्यम्_Multi_Byte(CP_UTF8, WC_ERR_INVALID_CHARS,
                                            wstr.दत्त(), (पूर्णाङ्क) wstr.size(),
                                            शून्यसूचकः, 0, शून्यसूचकः, शून्यसूचकः);
    यदि (अष्टकानि_needed == 0) {
        GGML_LOG_ERROR("ggml-hex: WideCharToMultiByte failed. Error %lu\n", Get_चरमः_Error());
        क्षिप std::runtime_error("Invalid wstring input");
    }

    परिणाम.resize(अष्टकानि_needed, '\0');
    पूर्णाङ्क अष्टकानि_written = Wide_Char_गन्तव्यम्_Multi_Byte(CP_UTF8, WC_ERR_INVALID_CHARS,
                                            wstr.दत्त(), (पूर्णाङ्क) wstr.size(),
                                            परिणाम.दत्त(), अष्टकानि_needed,
                                            शून्यसूचकः, शून्यसूचकः);
    यदि (अष्टकानि_written == 0) {
        GGML_LOG_ERROR("ggml-hex: WideCharToMultiByte failed. Error %lu\n", Get_चरमः_Error());
        क्षिप std::runtime_error("Wstring conversion failed");
    }
    प्रत्यागम परिणाम;
}

स्थिर std::string get_driver_मार्गः() {
    std::wstring serviceName = L"qcnspmcdm";
    std::string परिणाम;

    // Get a handle to the SCM database.
    SC_HANDLE schSCManager = OpenSCManagerW(रिक्त, रिक्त, STANDARD_RIGHTS_READ);
    यदि (शून्यसूचकः == schSCManager) {
        GGML_LOG_ERROR("ggml-hex: Failed to open SCManager. Error: %lu\n", Get_चरमः_Error());
        प्रत्यागम परिणाम;
    }

    // Get a handle to the service.
    SC_HANDLE schService = OpenServiceW(schSCManager,           // SCM database
                                        serviceName.c_str(),    // name of service
                                        SERVICE_QUERY_CONFIG);  // need query config access

    यदि (शून्यसूचकः == schService) {
        GGML_LOG_ERROR("ggml-hex: Failed to open qcnspmcdm service. Error: %lu\n", Get_चरमः_Error());
        CloseServiceHandle(schSCManager);
        प्रत्यागम परिणाम;
    }

    // Store the size of buffer used as an output.
    DWORD बफर_आकार_2;
    यदि (!QueryServiceConfigW(schService, रिक्त, 0, &बफर_आकार_2) &&
        (Get_चरमः_Error() != ERROR_INSUFFICIENT_बफर)) {
        GGML_LOG_ERROR("ggml-hex: Failed to query service config. Error: %lu\n", Get_चरमः_Error());
        CloseServiceHandle(schService);
        CloseServiceHandle(schSCManager);
        प्रत्यागम परिणाम;
    }
    // Get the configuration of the service.
    LPQUERY_SERVICE_CONFIGW serviceConfig =
        static_cast<LPQUERY_SERVICE_CONFIGW>(LocalAlloc(LMEM_FIXED, बफर_आकार_2));
    यदि (!QueryServiceConfigW(schService, serviceConfig, बफर_आकार_2, &बफर_आकार_2)) {
        fprintf(stderr, "ggml-hex: Failed to query service config. Error: %lu\n", Get_चरमः_Error());
        LocalFree(serviceConfig);
        CloseServiceHandle(schService);
        CloseServiceHandle(schSCManager);
        प्रत्यागम परिणाम;
    }

    // Read the driver file path get its parent directory
    std::wstring driver_मार्गः = std::wstring(serviceConfig->lp_द्विमान_मार्गः_Name);
    driver_मार्गः = driver_मार्गः.substr(0, driver_मार्गः.find_last_of(L"\\"));

    // Clean up resources
    LocalFree(serviceConfig);
    CloseServiceHandle(schService);
    CloseServiceHandle(schSCManager);

    // Driver path would contain invalid path string, like:
    // \SystemRoot\System32\DriverStore\FileRepository\qcadsprpc8280.inf_arm64_c2b9460c9a072f37
    // "\SystemRoot" should be replace with a correct one (e.g. C:\Windows)
    ध्रुव std::wstring system_मूलम्_Placeholder = L"\\SystemRoot";
    यदि (0 != driver_मार्गः.compare(0, system_मूलम्_Placeholder.दीर्घता(), system_मूलम्_Placeholder)) {
        GGML_LOG_ERROR("ggml-hex: String pattern not found in driver path.\n");
        प्रत्यागम परिणाम;
    }

    // Replace \SystemRoot with an absolute path from system ENV windir
    ध्रुव std::wstring system_मूलम्_Env = L"windir";

    // Query the number of wide characters this variable requires
    DWORD अङ्कः_शब्दाः = GetEnvironmentVariableW(system_मूलम्_Env.c_str(), रिक्त, 0);
    यदि (अङ्कः_शब्दाः == 0) {
        GGML_LOG_ERROR("ggml-hex: Failed get systemRoot environment variable\n");
        प्रत्यागम परिणाम;
    }

    // Query the actual system root name from environment variable
    std::vector<wchar_t> system_मूलम्(अङ्कः_शब्दाः + 1);
    अङ्कः_शब्दाः = GetEnvironmentVariableW(system_मूलम्_Env.c_str(), system_मूलम्.दत्त(), अङ्कः_शब्दाः + 1);
    यदि (अङ्कः_शब्दाः == 0) {
        GGML_LOG_ERROR("ggml-hex: Failed to read windir environment variable\n");
        प्रत्यागम परिणाम;
    }
    driver_मार्गः.replace(0, system_मूलम्_Placeholder.दीर्घता(), std::wstring(system_मूलम्.दत्त()));

    प्रत्यागम wstr_गन्तव्यम्_str(driver_मार्गः);
}

#यदि_अन्तः

उपयुज्य dl_handle_ptr = std::unique_ptr<dl_handle, dl_handle_deleter>;

पूर्णाङ्क htpdrv_init() {
    स्थिर dl_handle_ptr lib_cdsp_rpc_handle = शून्यसूचकः;
    स्थिर तार्किक initialized = असत्य;
#यदि_परिभाषितम् _WIN32
    std::string drv_मार्गः = get_driver_मार्गः() + "\\" + "libcdsprpc.dll";
#अन्यथा
    std::string drv_मार्गः = "libcdsprpc.so";
#यदि_अन्तः
    यदि (initialized) {
        GGML_LOG_INFO("ggml-hex: Driver already loaded\n");
        प्रत्यागम AEE_SUCCESS;
    }
    GGML_LOG_INFO("ggml-hex: Loading driver %s\n", drv_मार्गः.c_str());

    fs::path path{ drv_मार्गः.c_str() };
    dl_handle_ptr handle { dl_आरोपय_library(path) };
    यदि (!handle) {
        GGML_LOG_ERROR("ggml-hex: failed to load %s: %s\n", path.u8string().c_str(), dl_error());
        प्रत्यागम AEE_EUNABLETOLOAD;
    }

#परिभाषय dlsym(drv, type, pfn, चिह्नम्, ignore)                               \
    कुरु {                                                                    \
        pfn = (type) dl_get_sym(drv, #symbol);                              \
        यदि (!ignore && शून्यसूचकः == pfn) {                                    \
            GGML_LOG_ERROR("ggml-hex: failed to dlsym %s\n", #symbol);      \
            प्रत्यागम AEE_EUNABLETOLOAD;                                       \
        }                                                                   \
    } यावत् (0)

    dlsym(handle.get(), rpcmem_alloc_pfn_t, rpcmem_alloc_pfn, rpcmem_alloc, असत्य);
    dlsym(handle.get(), rpcmem_alloc2_pfn_t, rpcmem_alloc2_pfn, rpcmem_alloc2, सत्य);
    dlsym(handle.get(), rpcmem_free_pfn_t, rpcmem_free_pfn, rpcmem_free, असत्य);
    dlsym(handle.get(), rpcmem_गन्तव्यम्_fd_pfn_t, rpcmem_गन्तव्यम्_fd_pfn, rpcmem_गन्तव्यम्_fd, असत्य);
    dlsym(handle.get(), fastrpc_mmap_pfn_t, fastrpc_mmap_pfn, fastrpc_mmap, असत्य);
    dlsym(handle.get(), fastrpc_munmap_pfn_t, fastrpc_munmap_pfn, fastrpc_munmap, असत्य);
    dlsym(handle.get(), dspqueue_create_pfn_t, dspqueue_create_pfn, dspqueue_create, असत्य);
    dlsym(handle.get(), dspqueue_close_pfn_t, dspqueue_close_pfn, dspqueue_close, असत्य);
    dlsym(handle.get(), dspqueue_export_pfn_t, dspqueue_export_pfn, dspqueue_export, असत्य);
    dlsym(handle.get(), dspqueue_write_pfn_t, dspqueue_write_pfn, dspqueue_write, असत्य);
    dlsym(handle.get(), dspqueue_read_pfn_t, dspqueue_read_pfn, dspqueue_read, असत्य);
    dlsym(handle.get(), dspqueue_read_noblock_pfn_t, dspqueue_read_noblock_pfn, dspqueue_read_noblock, असत्य);
    dlsym(handle.get(), remote_handle64_open_pfn_t, remote_handle64_open_pfn, remote_handle64_open, असत्य);
    dlsym(handle.get(), remote_handle64_invoke_pfn_t, remote_handle64_invoke_pfn, remote_handle64_invoke, असत्य);
    dlsym(handle.get(), remote_handle_control_pfn_t, remote_handle_control_pfn, remote_handle_control, असत्य);
    dlsym(handle.get(), remote_handle64_control_pfn_t, remote_handle64_control_pfn, remote_handle64_control, असत्य);
    dlsym(handle.get(), remote_session_control_pfn_t, remote_session_control_pfn, remote_session_control, असत्य);
    dlsym(handle.get(), remote_handle64_close_pfn_t, remote_handle64_close_pfn, remote_handle64_close, असत्य);
    dlsym(handle.get(), remote_system_request_pfn_t, remote_system_request_pfn, remote_system_request, सत्य);

    lib_cdsp_rpc_handle = std::move(handle);
    initialized         = सत्य;

    प्रत्यागम AEE_SUCCESS;
}

domain * htpdrv_get_domain(पूर्णाङ्क domain_id) {
    पूर्णाङ्क i    = 0;
    पूर्णाङ्क size = परिमाण(supported_domains) / परिमाण(domain);

    प्रति (i = 0; i < size; i++) {
        यदि (supported_domains[i].id == domain_id) {
            प्रत्यागम &supported_domains[i];
        }
    }

    प्रत्यागम रिक्त;
}

पूर्णाङ्क htpdrv_get_arch(पूर्णाङ्क domain, पूर्णाङ्क * arch) {
    यदि (!remote_handle_control_pfn) {
        GGML_LOG_ERROR("ggml-hex: remote_handle_control is not supported on this device\n");
        प्रत्यागम AEE_EUNSUPPORTEDAPI;
    }

    संरचना remote_dsp_capability arch_ver;
    arch_ver.domain       = (uint32_t) domain;
    arch_ver.attribute_ID = ARCH_VER;
    arch_ver.capability   = (uint32_t) 0;

    पूर्णाङ्क err = remote_handle_control(DSPRPC_GET_DSP_INFO, &arch_ver, परिमाण(arch_ver));
    यदि ((err & 0xff) == (AEE_EUNSUPPORTEDAPI & 0xff)) {
        GGML_LOG_ERROR("ggml-hex: FastRPC capability API is not supported on this device\n");
        प्रत्यागम AEE_EUNSUPPORTEDAPI;
    }

    यदि (err != AEE_SUCCESS) {
        GGML_LOG_ERROR("ggml-hex: FastRPC capability query failed (err %d)\n", err);
        प्रत्यागम err;
    }

    uint32_t val = arch_ver.capability & 0xff;
    *arch = (पूर्णाङ्क) ((val >> 4) * 10 + (val & 0x0f));
    प्रत्यागम 0;
}
