#include "pvz_sim.h"

#include <stddef.h>
#include <stdlib.h>

struct PvzBatch {
    PvzConfig config;
    int32_t num_envs;
    PvzState* states;
};

int32_t pvz_action_count(void) {
    return PVZ_ACTION_COUNT;
}

int32_t pvz_observation_size(void) {
    return PVZ_OBSERVATION_SIZE;
}

PvzBatch* pvz_batch_create(int32_t num_envs, const PvzConfig* config) {
    PvzBatch* batch;
    if (
        num_envs <= 0
        || config == NULL
        || config->tick_seconds <= 0.0f
        || config->max_time <= 0.0f
        || config->passive_sun_interval <= 0.0f
        || config->spawn_interval <= 0.0f
        || config->max_zombies <= 0
        || config->max_zombies > PVZ_MAX_ZOMBIES
    ) {
        return NULL;
    }
    batch = (PvzBatch*)calloc(1, sizeof(PvzBatch));
    if (batch == NULL) {
        return NULL;
    }
    batch->states = (PvzState*)calloc((size_t)num_envs, sizeof(PvzState));
    if (batch->states == NULL) {
        free(batch);
        return NULL;
    }
    batch->config = *config;
    batch->num_envs = num_envs;
    return batch;
}

void pvz_batch_destroy(PvzBatch* batch) {
    if (batch != NULL) {
        free(batch->states);
        free(batch);
    }
}

int32_t pvz_batch_reset(
    PvzBatch* batch,
    const uint64_t* seeds,
    float* observations,
    uint8_t* legal_masks
) {
    int32_t index;
    if (
        batch == NULL
        || seeds == NULL
        || observations == NULL
        || legal_masks == NULL
    ) {
        return -1;
    }
    for (index = 0; index < batch->num_envs; ++index) {
        pvz_reset_state(&batch->states[index], &batch->config, seeds[index]);
        pvz_write_observation(
            &batch->states[index],
            &batch->config,
            observations + (size_t)index * PVZ_OBSERVATION_SIZE
        );
        pvz_write_legal_mask(
            &batch->states[index],
            legal_masks + (size_t)index * PVZ_ACTION_COUNT
        );
    }
    return 0;
}

int32_t pvz_batch_step(
    PvzBatch* batch,
    const int32_t* actions,
    float* observations,
    float* rewards,
    uint8_t* terminals,
    uint8_t* truncations,
    uint8_t* legal_masks
) {
    int32_t index;
    if (
        batch == NULL
        || actions == NULL
        || observations == NULL
        || rewards == NULL
        || terminals == NULL
        || truncations == NULL
        || legal_masks == NULL
    ) {
        return -1;
    }
    for (index = 0; index < batch->num_envs; ++index) {
        PvzState* state = &batch->states[index];
        if (
            state->won
            || state->lost
            || state->truncated
            || !pvz_is_action_legal(state, actions[index])
        ) {
            return index + 1;
        }
    }

#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (index = 0; index < batch->num_envs; ++index) {
        PvzState* state = &batch->states[index];
        rewards[index] = pvz_step_state(state, &batch->config, actions[index]);
        terminals[index] = (uint8_t)(state->won || state->lost);
        truncations[index] = state->truncated;
        pvz_write_observation(
            state,
            &batch->config,
            observations + (size_t)index * PVZ_OBSERVATION_SIZE
        );
        pvz_write_legal_mask(
            state,
            legal_masks + (size_t)index * PVZ_ACTION_COUNT
        );
    }
    return 0;
}
