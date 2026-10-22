#निर्देशः once

#यदि_परिभाषितम् __cplusplus
बाह्य "C" {
#यदि_अन्तः

#यदि_परिभाषितम् _WIN32
#    pragma clang diagnostic ignored "-Wignored-attributes"
#यदि_अन्तः

#अन्तर्भाव <AEEStdErr.h>
#अन्तर्भाव <rpcmem.h>
#अन्तर्भाव <remote.h>
#अन्तर्भाव <dspqueue.h>

#यदि defined(_WIN32) && !defined(__MINGW32__)
#    ifdef GGML_BACKEND_BUILD
#        define HTPDRV_API __declspec(dllexport) बाह्य
#    अन्यथा
#        define HTPDRV_API __declspec(dllimport) बाह्य
#    endif
#अन्यथा
#    define HTPDRV_API __attribute__ ((visibility ("default"))) बाह्य
#यदि_अन्तः

/* Offset to differentiate HLOS and Hexagon error codes.
   Stores the value of AEE_EOFFSET for Hexagon. */
#यदि_अपरिभाषितम् DSP_विस्थापनम्
#    define DSP_विस्थापनम् 0x80000400
#यदि_अन्तः

/* Errno for connection reset by peer. */
#यदि_अपरिभाषितम् ECONNRESET
#    ifdef __hexagon__
#        define ECONNRESET 104
#    endif
#यदि_अन्तः

/* Abstraction of different OS specific sleep APIs.
   SLEEP accepts input in seconds. */
#यदि_अपरिभाषितम् SLEEP
#    ifdef __hexagon__
#        define SLEEP(x)                      \
            { /* Do nothing for simulator. */ \
            }
#    अन्यथा
#        ifdef _WIN32
#            define SLEEP(x) Sleep(1000 * x) /* Sleep accepts input in milliseconds. */
#        अन्यथा
#            define SLEEP(x) sleep(x)        /* sleep accepts input in seconds. */
#        endif
#    endif
#यदि_अन्तः

/* Include windows specific header files. */
#यदि_परिभाषितम् _WIN32
#    include <windows.h>
#    include <sysinfoapi.h>
#    define _CRT_SECURE_NO_WARNINGS         1
#    define _WINSOCK_DEPRECATED_NO_WARNINGS 1
#यदि_अन्तः

/* Includes and defines for all HLOS except windows */
#यदि !defined(__hexagon__) && !defined(_WIN32)
#    include "unistd.h"

#    include <sys/time.h>
#यदि_अन्तः

/* Includes and defines for Hexagon and all HLOS except Windows. */
#यदि !defined(_WIN32)
/* Weak reference to remote symbol for compilation. */
#    pragma weak remote_session_control
#    pragma weak remote_handle_control
#    pragma weak remote_handle64_control
#    pragma weak fastrpc_mmap
#    pragma weak fastrpc_munmap
#    pragma weak rpcmem_alloc2
#यदि_अन्तः

#यदि !defined(_WIN32)
#    pragma weak remote_system_request
#यदि_अन्तः

#यदि_परिभाषितम् _WIN32
#     define DSPQUEUE_TIMEOUT DSPQUEUE_TIMEOUT_NONE
#अन्यथा
#     define DSPQUEUE_TIMEOUT 1000000
#यदि_अन्तः

/**
 * htpdrv_init API: driver interface entry point
 *
 * @return      Return AEE error codes as defined in Hexagon SDK.
 */
HTPDRV_API पूर्णाङ्क htpdrv_init(शून्य);

/**
 * htpdrv_get_domain API: get domain struct from domain value.
 *
 * @param[in]  domain value of a domain
 * @return     Returns domain struct of the domain if it is supported or else
 *             returns NULL.
 *
 */
HTPDRV_API domain * htpdrv_get_domain(पूर्णाङ्क domain_id);

/**
 * htpdrv_get_arch API: query the Hexagon processor architecture version information
 *
 * @param[in]   domain_id value of a domain
 * @param[out]  Arch version (73, 75, ...)
 * @return      0 if query is successful.
 *              non-zero if error, return value points to the error.
 *
 */
HTPDRV_API पूर्णाङ्क htpdrv_get_arch(पूर्णाङ्क domain, पूर्णाङ्क * arch);

HTPDRV_API पूर्णाङ्क remote_system_request(system_req_payload * req);

#यदि_परिभाषितम् __cplusplus
}
#यदि_अन्तः
