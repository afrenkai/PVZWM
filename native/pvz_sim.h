#ifndef PVZWM_PVZ_SIM_H
#define PVZWM_PVZ_SIM_H

#include <stdint.h>

#include "pvz_core.h"

#if defined(_WIN32)
#define PVZ_API __declspec(dllexport)
#else
#define PVZ_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PvzBatch PvzBatch;

PVZ_API int32_t pvz_action_count(void);
PVZ_API int32_t pvz_observation_size(void);
PVZ_API PvzBatch* pvz_batch_create(int32_t num_envs, const PvzConfig* config);
PVZ_API void pvz_batch_destroy(PvzBatch* batch);
PVZ_API int32_t pvz_batch_reset(
    PvzBatch* batch,
    const uint64_t* seeds,
    float* observations,
    uint8_t* legal_masks
);
PVZ_API int32_t pvz_batch_step(
    PvzBatch* batch,
    const int32_t* actions,
    float* observations,
    float* rewards,
    uint8_t* terminals,
    uint8_t* truncations,
    uint8_t* legal_masks
);

#ifdef __cplusplus
}
#endif

#endif
