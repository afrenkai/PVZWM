#ifndef PVZWM_PVZ_CORE_H
#define PVZWM_PVZ_CORE_H

#include <stdint.h>

#define PVZ_ROWS 5
#define PVZ_COLS 9
#define PVZ_CELLS (PVZ_ROWS * PVZ_COLS)
#define PVZ_MAX_ZOMBIES 64
#define PVZ_ACTION_COUNT (1 + 4 * PVZ_CELLS)
#define PVZ_OBSERVATION_SIZE (4 * PVZ_CELLS + 4 + PVZ_ROWS + 3)
#define PVZ_NEVER 3.402823466e+38F

enum PvzActionKind {
    PVZ_WAIT = 0,
    PVZ_PLANT_SUNFLOWER = 1,
    PVZ_PLANT_PEASHOOTER = 2,
    PVZ_PLANT_WALLNUT = 3,
    PVZ_SHOVEL = 4,
};

enum PvzPlantKind {
    PVZ_EMPTY = 0,
    PVZ_SUNFLOWER = 1,
    PVZ_PEASHOOTER = 2,
    PVZ_WALLNUT = 3,
};

typedef struct PvzConfig {
    float tick_seconds;
    float max_time;
    float starting_sun;
    float passive_sun_interval;
    float spawn_interval;
    float final_wave_time;
    int32_t max_zombies;
} PvzConfig;

typedef struct PvzZombie {
    int32_t row;
    float x;
    float hp;
    float speed;
    float damage_per_second;
} PvzZombie;

typedef struct PvzState {
    uint8_t plants[PVZ_CELLS];
    float plant_hp[PVZ_CELLS];
    float sunflower_ready_at[PVZ_CELLS];
    float sun;
    float time;
    float next_passive_sun;
    float next_spawn;
    float seed_ready_at[3];
    PvzZombie zombies[PVZ_MAX_ZOMBIES];
    int32_t zombie_count;
    uint8_t mowers[PVZ_ROWS];
    int32_t kills;
    uint8_t won;
    uint8_t lost;
    uint8_t truncated;
    uint64_t rng_state;
} PvzState;

static inline float pvz_minf(float a, float b) {
    return a < b ? a : b;
}

static inline float pvz_maxf(float a, float b) {
    return a > b ? a : b;
}

static inline float pvz_clipf(float value, float low, float high) {
    return pvz_minf(pvz_maxf(value, low), high);
}

static inline uint64_t pvz_mix_seed(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    value ^= value >> 31;
    return value == 0 ? 0x9e3779b97f4a7c15ULL : value;
}

static inline uint64_t pvz_random_u64(PvzState* state) {
    uint64_t value = state->rng_state;
    value ^= value >> 12;
    value ^= value << 25;
    value ^= value >> 27;
    state->rng_state = value;
    return value * 0x2545f4914f6cdd1dULL;
}

static inline float pvz_random_unit(PvzState* state) {
    return (float)(pvz_random_u64(state) >> 40) * (1.0f / 16777216.0f);
}

static inline void pvz_reset_state(
    PvzState* state,
    const PvzConfig* config,
    uint64_t seed
) {
    int32_t index;
    for (index = 0; index < PVZ_CELLS; ++index) {
        state->plants[index] = PVZ_EMPTY;
        state->plant_hp[index] = 0.0f;
        state->sunflower_ready_at[index] = PVZ_NEVER;
    }
    state->sun = config->starting_sun;
    state->time = 0.0f;
    state->next_passive_sun = config->passive_sun_interval;
    state->next_spawn = config->spawn_interval;
    for (index = 0; index < 3; ++index) {
        state->seed_ready_at[index] = 0.0f;
    }
    state->zombie_count = 0;
    for (index = 0; index < PVZ_ROWS; ++index) {
        state->mowers[index] = 1;
    }
    state->kills = 0;
    state->won = 0;
    state->lost = 0;
    state->truncated = 0;
    state->rng_state = pvz_mix_seed(seed);
}

static inline void pvz_decode_action(
    int32_t action,
    int32_t* kind,
    int32_t* row,
    int32_t* col
) {
    int32_t offset;
    int32_t cell;
    if (action == 0) {
        *kind = PVZ_WAIT;
        *row = 0;
        *col = 0;
        return;
    }
    offset = action - 1;
    *kind = 1 + offset / PVZ_CELLS;
    cell = offset % PVZ_CELLS;
    *row = cell / PVZ_COLS;
    *col = cell % PVZ_COLS;
}

static inline float pvz_plant_cost(int32_t kind) {
    return kind == PVZ_PLANT_PEASHOOTER ? 100.0f : 50.0f;
}

static inline float pvz_recharge_seconds(int32_t kind) {
    return kind == PVZ_PLANT_WALLNUT ? 30.0f : 7.5f;
}

static inline int32_t pvz_is_action_legal(
    const PvzState* state,
    int32_t action
) {
    int32_t kind;
    int32_t row;
    int32_t col;
    int32_t cell;
    if (action < 0 || action >= PVZ_ACTION_COUNT) {
        return 0;
    }
    if (action == 0) {
        return 1;
    }
    pvz_decode_action(action, &kind, &row, &col);
    cell = row * PVZ_COLS + col;
    if (kind == PVZ_SHOVEL) {
        return state->plants[cell] != PVZ_EMPTY;
    }
    return state->plants[cell] == PVZ_EMPTY
        && state->sun >= pvz_plant_cost(kind)
        && state->time >= state->seed_ready_at[kind - 1];
}

static inline void pvz_write_legal_mask(
    const PvzState* state,
    uint8_t* mask
) {
    int32_t action;
    for (action = 0; action < PVZ_ACTION_COUNT; ++action) {
        mask[action] = (uint8_t)pvz_is_action_legal(state, action);
    }
}

static inline void pvz_write_observation(
    const PvzState* state,
    const PvzConfig* config,
    float* observation
) {
    int32_t index;
    int32_t zombie;
    int32_t col;
    float zombie_counts[PVZ_CELLS];
    float zombie_hp[PVZ_CELLS];

    for (index = 0; index < PVZ_CELLS; ++index) {
        zombie_counts[index] = 0.0f;
        zombie_hp[index] = 0.0f;
        observation[index] = (float)state->plants[index] / 3.0f;
        observation[PVZ_CELLS + index] =
            pvz_clipf(state->plant_hp[index] / 4000.0f, 0.0f, 1.0f);
    }
    for (zombie = 0; zombie < state->zombie_count; ++zombie) {
        col = (int32_t)state->zombies[zombie].x;
        if (col < 0) {
            col = 0;
        } else if (col >= PVZ_COLS) {
            col = PVZ_COLS - 1;
        }
        index = state->zombies[zombie].row * PVZ_COLS + col;
        zombie_counts[index] += 1.0f;
        zombie_hp[index] += pvz_maxf(state->zombies[zombie].hp, 0.0f);
    }
    for (index = 0; index < PVZ_CELLS; ++index) {
        observation[2 * PVZ_CELLS + index] =
            pvz_clipf(zombie_counts[index] / 4.0f, 0.0f, 1.0f);
        observation[3 * PVZ_CELLS + index] =
            pvz_clipf(zombie_hp[index] / 500.0f, 0.0f, 1.0f);
    }
    index = 4 * PVZ_CELLS;
    observation[index++] = pvz_minf(state->sun / 1000.0f, 1.0f);
    observation[index++] = pvz_minf(state->time / config->max_time, 1.0f);
    observation[index++] = (float)state->won;
    observation[index++] = (float)state->lost;
    for (col = 0; col < PVZ_ROWS; ++col) {
        observation[index++] = (float)state->mowers[col];
    }
    for (col = 0; col < 3; ++col) {
        float remaining = pvz_maxf(state->seed_ready_at[col] - state->time, 0.0f);
        observation[index++] = remaining / (col == 2 ? 30.0f : 7.5f);
    }
}

static inline float pvz_apply_action(PvzState* state, int32_t action) {
    int32_t kind;
    int32_t row;
    int32_t col;
    int32_t cell;
    float max_hp;
    if (action == 0) {
        return 0.0f;
    }
    pvz_decode_action(action, &kind, &row, &col);
    cell = row * PVZ_COLS + col;
    if (kind == PVZ_SHOVEL) {
        state->plants[cell] = PVZ_EMPTY;
        state->plant_hp[cell] = 0.0f;
        state->sunflower_ready_at[cell] = PVZ_NEVER;
        return -0.02f;
    }
    max_hp = kind == PVZ_WALLNUT ? 4000.0f : 300.0f;
    state->sun -= pvz_plant_cost(kind);
    state->plants[cell] = (uint8_t)kind;
    state->plant_hp[cell] = max_hp;
    state->seed_ready_at[kind - 1] = state->time + pvz_recharge_seconds(kind);
    if (kind == PVZ_SUNFLOWER) {
        state->sunflower_ready_at[cell] = state->time + 7.0f;
    }
    return 0.01f;
}

static inline void pvz_advance_economy(
    PvzState* state,
    const PvzConfig* config
) {
    int32_t index;
    float next_time = state->time + config->tick_seconds;
    if (next_time >= state->next_passive_sun) {
        state->sun += 25.0f;
        state->next_passive_sun += config->passive_sun_interval;
    }
    for (index = 0; index < PVZ_CELLS; ++index) {
        if (next_time < state->sunflower_ready_at[index]) {
            continue;
        }
        if (state->plants[index] == PVZ_SUNFLOWER) {
            state->sun += 25.0f;
            state->sunflower_ready_at[index] += 24.0f;
        } else {
            state->sunflower_ready_at[index] = PVZ_NEVER;
        }
    }
    if (
        next_time >= state->next_spawn
        && state->time < config->final_wave_time
        && state->zombie_count < config->max_zombies
        && state->zombie_count < PVZ_MAX_ZOMBIES
    ) {
        PvzZombie* zombie = &state->zombies[state->zombie_count++];
        float difficulty = 1.0f + 0.8f * state->time / config->max_time;
        float jitter;
        zombie->row = (int32_t)(pvz_random_u64(state) % PVZ_ROWS);
        zombie->x = (float)PVZ_COLS + 0.2f;
        zombie->hp = 120.0f * difficulty;
        zombie->speed = 0.18f;
        zombie->damage_per_second = 30.0f;
        jitter = 0.82f + 0.36f * pvz_random_unit(state);
        state->next_spawn += config->spawn_interval * jitter;
    } else if (
        state->time >= config->final_wave_time
        && state->zombie_count == 0
    ) {
        state->next_spawn = PVZ_NEVER;
    }
}

static inline float pvz_advance_combat(
    PvzState* state,
    const PvzConfig* config
) {
    int32_t row;
    int32_t col;
    int32_t index;
    int32_t zombie_index;
    int32_t target_index;
    int32_t shooter_count;
    int32_t survivor_count;
    float target_x;
    float reward = 0.0f;
    float dt = config->tick_seconds;

    for (row = 0; row < PVZ_ROWS; ++row) {
        shooter_count = 0;
        for (col = 0; col < PVZ_COLS; ++col) {
            if (state->plants[row * PVZ_COLS + col] == PVZ_PEASHOOTER) {
                ++shooter_count;
            }
        }
        target_index = -1;
        target_x = PVZ_NEVER;
        for (zombie_index = 0; zombie_index < state->zombie_count; ++zombie_index) {
            PvzZombie* zombie = &state->zombies[zombie_index];
            if (zombie->row == row && zombie->x >= 0.0f && zombie->x < target_x) {
                target_index = zombie_index;
                target_x = zombie->x;
            }
        }
        if (shooter_count > 0 && target_index >= 0) {
            state->zombies[target_index].hp -=
                (20.0f / 1.425f) * (float)shooter_count * dt;
        }
    }

    for (zombie_index = 0; zombie_index < state->zombie_count; ++zombie_index) {
        PvzZombie* zombie = &state->zombies[zombie_index];
        col = (int32_t)zombie->x;
        if (col < 0) {
            col = 0;
        } else if (col >= PVZ_COLS) {
            col = PVZ_COLS - 1;
        }
        index = zombie->row * PVZ_COLS + col;
        if (
            zombie->x >= 0.0f
            && zombie->x < (float)PVZ_COLS
            && state->plants[index] != PVZ_EMPTY
        ) {
            state->plant_hp[index] -= zombie->damage_per_second * dt;
            if (state->plant_hp[index] <= 0.0f) {
                state->plants[index] = PVZ_EMPTY;
                state->plant_hp[index] = 0.0f;
                state->sunflower_ready_at[index] = PVZ_NEVER;
                reward -= 0.2f;
            }
        } else {
            zombie->x -= zombie->speed * dt;
        }
        if (zombie->x < 0.0f && state->mowers[zombie->row]) {
            int32_t other;
            state->mowers[zombie->row] = 0;
            for (other = 0; other < state->zombie_count; ++other) {
                if (state->zombies[other].row == zombie->row) {
                    state->zombies[other].hp = 0.0f;
                }
            }
            reward -= 0.5f;
        }
    }

    survivor_count = 0;
    for (zombie_index = 0; zombie_index < state->zombie_count; ++zombie_index) {
        if (state->zombies[zombie_index].hp <= 0.0f) {
            ++state->kills;
            reward += 1.0f;
        } else {
            if (survivor_count != zombie_index) {
                state->zombies[survivor_count] = state->zombies[zombie_index];
            }
            ++survivor_count;
        }
    }
    state->zombie_count = survivor_count;
    return reward;
}

static inline float pvz_step_state(
    PvzState* state,
    const PvzConfig* config,
    int32_t action
) {
    int32_t zombie_index;
    float reward = pvz_apply_action(state, action);
    pvz_advance_economy(state, config);
    reward += pvz_advance_combat(state, config);
    state->time += config->tick_seconds;

    for (zombie_index = 0; zombie_index < state->zombie_count; ++zombie_index) {
        if (state->zombies[zombie_index].x < -0.4f) {
            state->lost = 1;
            reward -= 20.0f;
            break;
        }
    }
    if (
        state->time >= config->final_wave_time
        && state->zombie_count == 0
        && state->next_spawn == PVZ_NEVER
    ) {
        state->won = 1;
        reward += 20.0f;
    }
    if (state->time >= config->max_time && !state->won && !state->lost) {
        state->truncated = 1;
    }
    return reward - 0.002f;
}

#endif
