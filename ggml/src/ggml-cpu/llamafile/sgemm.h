#సూచన once
#అన్తర్భావ <stdint.h>
#అన్తర్భావ <stdbool.h>

#ఒకవేళ defined(__VXE__) || defined(__VXE2__)
#అన్తర్భావ <vecintrin.h>
#అయితే_ముగింపు

#నిర్వచితమైతే _MSC_VER
#నిర్వచించు NOINLINE __declspec(noinline)
#లేకపోతే
#నిర్వచించు NOINLINE __attribute__((__noinline__))
#అయితే_ముగింపు

#నిర్వచితమైతే __cplusplus
బాహ్య "C" {
#అయితే_ముగింపు

bool llamafile_sgemm(స్థిరరాశి నిర్మాణం ggml_compute_params * params, int64_t, int64_t, int64_t,
                     స్థిరరాశి శూన్య *, int64_t, స్థిరరాశి శూన్య *, int64_t, శూన్య *, int64_t,
                     పూర్ణాంకం, పూర్ణాంకం, పూర్ణాంకం);

#నిర్వచితమైతే __cplusplus
}
#అయితే_ముగింపు
