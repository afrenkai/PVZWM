#define PUF_BACKEND PUF_GPU

#include <assert.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>

typedef __nv_bfloat16 obs_t;
#include "pufferenv.h"

#define PVZ_ROWS 5
#define PVZ_COLS 9
#define PVZ_CELLS (PVZ_ROWS * PVZ_COLS)
#define PVZ_MAX_ZOMBIES 64
#define PVZ_ACTION_COUNT (1 + 4 * PVZ_CELLS)
#define PVZ_OBSERVATION_SIZE (4 * PVZ_CELLS + 4 + PVZ_ROWS + 3)
#define PVZ_NEVER 3.402823466e+38F
#define PVZ_THREADS_PER_ENV 8
#define PVZ_CUDA_BLOCK_SIZE 256
#define ACT_SIZES {PVZ_ACTION_COUNT}
#define NUM_ATNS 1
#define OBS_SIZE PVZ_OBSERVATION_SIZE

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
    float invalid_action_penalty;
    int max_zombies;
} PvzConfig;

typedef struct PvzZombie {
    int row;
    float x;
    float hp;
    float speed;
    float damage_per_second;
} PvzZombie;

typedef struct PvzState {
    unsigned char plants[PVZ_CELLS];
    float plant_hp[PVZ_CELLS];
    float sunflower_ready_at[PVZ_CELLS];
    float sun;
    float time;
    float next_passive_sun;
    float next_spawn;
    float seed_ready_at[3];
    PvzZombie zombies[PVZ_MAX_ZOMBIES];
    int zombie_count;
    unsigned char mowers[PVZ_ROWS];
    int kills;
    unsigned char won;
    unsigned char lost;
    unsigned char truncated;
    uint64_t rng_state;
} PvzState;

struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float wins;
    float losses;
    float truncations;
    float kills;
    float invalid_actions;
    float n;
};

struct Env {
    Log log;
    Agent agents[1];
    int num_agents;
    int tag;
    int boundary_reached;
    PvzState state;
    float episode_return;
    int episode_steps;
    int episode_invalid_actions;
};

__constant__ PvzConfig pvz_config;

struct PvzGpu {
    Env* envs;
    int num_envs;
    obs_t* observations;
    float* actions;
    float* rewards;
    float* terminals;
    cudaStream_t stream;
} pvz_gpu;

__device__ __forceinline__ float pvz_min(float first, float second) {
    return first < second ? first : second;
}

__device__ __forceinline__ float pvz_max(float first, float second) {
    return first > second ? first : second;
}

__device__ __forceinline__ float pvz_clip(float value, float low, float high) {
    return pvz_min(pvz_max(value, low), high);
}

__device__ __forceinline__ uint64_t pvz_random_u64(PvzState* state) {
    uint64_t value = state->rng_state;
    value ^= value >> 12;
    value ^= value << 25;
    value ^= value >> 27;
    state->rng_state = value;
    return value * 0x2545f4914f6cdd1dULL;
}

__device__ void pvz_reset_state(PvzState* state, uint64_t seed) {
    for (int cell = 0; cell < PVZ_CELLS; cell++) {
        state->plants[cell] = PVZ_EMPTY;
        state->plant_hp[cell] = 0.0f;
        state->sunflower_ready_at[cell] = PVZ_NEVER;
    }
    state->sun = pvz_config.starting_sun;
    state->time = 0.0f;
    state->next_passive_sun = pvz_config.passive_sun_interval;
    state->next_spawn = pvz_config.spawn_interval;
    for (int seed_index = 0; seed_index < 3; seed_index++) {
        state->seed_ready_at[seed_index] = 0.0f;
    }
    state->zombie_count = 0;
    for (int row = 0; row < PVZ_ROWS; row++) {
        state->mowers[row] = 1;
    }
    state->kills = 0;
    state->won = 0;
    state->lost = 0;
    state->truncated = 0;
    seed += 0x9e3779b97f4a7c15ULL;
    seed = (seed ^ (seed >> 30)) * 0xbf58476d1ce4e5b9ULL;
    seed = (seed ^ (seed >> 27)) * 0x94d049bb133111ebULL;
    seed ^= seed >> 31;
    state->rng_state = seed == 0 ? 0x9e3779b97f4a7c15ULL : seed;
}

__device__ __forceinline__ void pvz_store_observation(obs_t* destination, float value) {
    *destination = __float2bfloat16(value);
}

__device__ void pvz_write_observation(PvzState* state, obs_t* observation, int lane) {
    for (int cell = lane; cell < PVZ_CELLS; cell += PVZ_THREADS_PER_ENV) {
        float zombie_count = 0.0f;
        float zombie_hp = 0.0f;
        for (int zombie_index = 0;
            zombie_index < state->zombie_count;
            zombie_index++
        ) {
            PvzZombie* zombie = &state->zombies[zombie_index];
            int column = zombie->x;
            if (column < 0) {
                column = 0;
            } else if (column >= PVZ_COLS) {
                column = PVZ_COLS - 1;
            }
            int zombie_cell = zombie->row * PVZ_COLS + column;
            if (zombie_cell == cell) {
                zombie_count += 1.0f;
                zombie_hp += pvz_max(zombie->hp, 0.0f);
            }
        }
        pvz_store_observation(&observation[cell], state->plants[cell] / 3.0f);
        pvz_store_observation(
            &observation[PVZ_CELLS + cell],
            pvz_clip(state->plant_hp[cell] / 4000.0f, 0.0f, 1.0f));
        pvz_store_observation(
            &observation[2 * PVZ_CELLS + cell],
            pvz_clip(zombie_count / 4.0f, 0.0f, 1.0f));
        pvz_store_observation(
            &observation[3 * PVZ_CELLS + cell],
            pvz_clip(zombie_hp / 500.0f, 0.0f, 1.0f));
    }

    for (int tail = lane; tail < 12; tail += PVZ_THREADS_PER_ENV) {
        float value;
        if (tail == 0) {
            value = pvz_min(state->sun / 1000.0f, 1.0f);
        } else if (tail == 1) {
            value = pvz_min(state->time / pvz_config.max_time, 1.0f);
        } else if (tail == 2) {
            value = state->won;
        } else if (tail == 3) {
            value = state->lost;
        } else if (tail < 9) {
            value = state->mowers[tail - 4];
        } else {
            int seed_index = tail - 9;
            float remaining = pvz_max(
                state->seed_ready_at[seed_index] - state->time, 0.0f);
            value = remaining / (seed_index == 2 ? 30.0f : 7.5f);
        }
        pvz_store_observation(&observation[4 * PVZ_CELLS + tail], value);
    }
}

void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "wins", log->wins);
    dict_set(out, "losses", log->losses);
    dict_set(out, "truncations", log->truncations);
    dict_set(out, "kills", log->kills);
    dict_set(out, "invalid_actions", log->invalid_actions);
    dict_set(out, "n", log->n);
}

__global__ void pvz_reset_kernel(
    Env* envs, obs_t* observations, float* rewards,
    float* terminals, int num_envs
) {
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    int env_index = thread_index / PVZ_THREADS_PER_ENV;
    int lane = thread_index - env_index * PVZ_THREADS_PER_ENV;
    int active = env_index < num_envs;

    if (active && lane == 0) {
        Env* env = &envs[env_index];
        pvz_reset_state(&env->state, env->state.rng_state);
        env->episode_return = 0.0f;
        env->episode_steps = 0;
        env->episode_invalid_actions = 0;
        rewards[env_index] = 0.0f;
        terminals[env_index] = 0.0f;
    }
    __syncwarp();
    if (active) {
        pvz_write_observation(
            &envs[env_index].state,
            observations + (long)env_index * OBS_SIZE,
            lane);
    }
}

__global__ void pvz_step_kernel(
    Env* envs, float* actions, obs_t* observations,
    float* rewards, float* terminals, int num_envs
) {
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    int env_index = thread_index / PVZ_THREADS_PER_ENV;
    int lane = thread_index - env_index * PVZ_THREADS_PER_ENV;
    int active = env_index < num_envs;

    if (active && lane == 0) {
        Env* env = &envs[env_index];
        PvzState* state = &env->state;
        int requested_action = actions[(long)env_index * NUM_ATNS];
        int action = requested_action;
        int kind = PVZ_WAIT;
        int cell = 0;
        int legal = action >= 0 && action < PVZ_ACTION_COUNT;
        if (legal && action != PVZ_WAIT) {
            int offset = action - 1;
            kind = 1 + offset / PVZ_CELLS;
            cell = offset % PVZ_CELLS;
            if (kind == PVZ_SHOVEL) {
                legal = state->plants[cell] != PVZ_EMPTY;
            } else {
                float cost = kind == PVZ_PLANT_PEASHOOTER ? 100.0f : 50.0f;
                legal = state->plants[cell] == PVZ_EMPTY
                    && state->sun >= cost
                    && state->time >= state->seed_ready_at[kind - 1];
            }
        }
        if (!legal) {
            action = PVZ_WAIT;
            env->episode_invalid_actions++;
        }

        float reward = 0.0f;
        if (action != PVZ_WAIT) {
            if (kind == PVZ_SHOVEL) {
                state->plants[cell] = PVZ_EMPTY;
                state->plant_hp[cell] = 0.0f;
                state->sunflower_ready_at[cell] = PVZ_NEVER;
                reward -= 0.02f;
            } else {
                float cost = kind == PVZ_PLANT_PEASHOOTER ? 100.0f : 50.0f;
                float recharge = kind == PVZ_PLANT_WALLNUT ? 30.0f : 7.5f;
                state->sun -= cost;
                state->plants[cell] = kind;
                state->plant_hp[cell] = kind == PVZ_WALLNUT ? 4000.0f : 300.0f;
                state->seed_ready_at[kind - 1] = state->time + recharge;
                if (kind == PVZ_SUNFLOWER) {
                    state->sunflower_ready_at[cell] = state->time + 7.0f;
                }
                reward += 0.01f;
            }
        }

        float next_time = state->time + pvz_config.tick_seconds;
        if (next_time >= state->next_passive_sun) {
            state->sun += 25.0f;
            state->next_passive_sun += pvz_config.passive_sun_interval;
        }
        for (int plant_cell = 0; plant_cell < PVZ_CELLS; plant_cell++) {
            if (next_time < state->sunflower_ready_at[plant_cell]) {
                continue;
            }
            if (state->plants[plant_cell] == PVZ_SUNFLOWER) {
                state->sun += 25.0f;
                state->sunflower_ready_at[plant_cell] += 24.0f;
            } else {
                state->sunflower_ready_at[plant_cell] = PVZ_NEVER;
            }
        }
        if (next_time >= state->next_spawn
            && state->time < pvz_config.final_wave_time
            && state->zombie_count < pvz_config.max_zombies
        ) {
            PvzZombie* zombie = &state->zombies[state->zombie_count++];
            float difficulty = 1.0f + 0.8f * state->time / pvz_config.max_time;
            zombie->row = pvz_random_u64(state) % PVZ_ROWS;
            zombie->x = PVZ_COLS + 0.2f;
            zombie->hp = 120.0f * difficulty;
            zombie->speed = 0.18f;
            zombie->damage_per_second = 30.0f;
            float random_unit = (pvz_random_u64(state) >> 40)
                * (1.0f / 16777216.0f);
            state->next_spawn += pvz_config.spawn_interval
                * (0.82f + 0.36f * random_unit);
        } else if (state->time >= pvz_config.final_wave_time
            && state->zombie_count == 0
        ) {
            state->next_spawn = PVZ_NEVER;
        }

        for (int lawn_row = 0; lawn_row < PVZ_ROWS; lawn_row++) {
            int shooter_count = 0;
            for (int lawn_column = 0; lawn_column < PVZ_COLS; lawn_column++) {
                int plant_cell = lawn_row * PVZ_COLS + lawn_column;
                shooter_count += state->plants[plant_cell] == PVZ_PEASHOOTER;
            }
            int target_index = -1;
            float target_x = PVZ_NEVER;
            for (int zombie_index = 0;
                zombie_index < state->zombie_count;
                zombie_index++
            ) {
                PvzZombie* zombie = &state->zombies[zombie_index];
                if (zombie->row == lawn_row && zombie->x >= 0.0f
                    && zombie->x < target_x
                ) {
                    target_index = zombie_index;
                    target_x = zombie->x;
                }
            }
            if (shooter_count > 0 && target_index >= 0) {
                state->zombies[target_index].hp -= (20.0f / 1.425f)
                    * shooter_count * pvz_config.tick_seconds;
            }
        }

        for (int zombie_index = 0;
            zombie_index < state->zombie_count;
            zombie_index++
        ) {
            PvzZombie* zombie = &state->zombies[zombie_index];
            int lawn_column = zombie->x;
            lawn_column = lawn_column < 0 ? 0 : lawn_column;
            lawn_column = lawn_column >= PVZ_COLS ? PVZ_COLS - 1 : lawn_column;
            int plant_cell = zombie->row * PVZ_COLS + lawn_column;
            if (zombie->x >= 0.0f && zombie->x < PVZ_COLS
                && state->plants[plant_cell] != PVZ_EMPTY
            ) {
                state->plant_hp[plant_cell] -= zombie->damage_per_second
                    * pvz_config.tick_seconds;
                if (state->plant_hp[plant_cell] <= 0.0f) {
                    state->plants[plant_cell] = PVZ_EMPTY;
                    state->plant_hp[plant_cell] = 0.0f;
                    state->sunflower_ready_at[plant_cell] = PVZ_NEVER;
                    reward -= 0.2f;
                }
            } else {
                zombie->x -= zombie->speed * pvz_config.tick_seconds;
            }
            if (zombie->x < 0.0f && state->mowers[zombie->row]) {
                state->mowers[zombie->row] = 0;
                for (int other_index = 0;
                    other_index < state->zombie_count;
                    other_index++
                ) {
                    if (state->zombies[other_index].row == zombie->row) {
                        state->zombies[other_index].hp = 0.0f;
                    }
                }
                reward -= 0.5f;
            }
        }

        int survivor_count = 0;
        for (int zombie_index = 0;
            zombie_index < state->zombie_count;
            zombie_index++
        ) {
            if (state->zombies[zombie_index].hp <= 0.0f) {
                state->kills++;
                reward += 1.0f;
                continue;
            }
            if (survivor_count != zombie_index) {
                state->zombies[survivor_count] = state->zombies[zombie_index];
            }
            survivor_count++;
        }
        state->zombie_count = survivor_count;
        state->time += pvz_config.tick_seconds;

        for (int zombie_index = 0;
            zombie_index < state->zombie_count;
            zombie_index++
        ) {
            if (state->zombies[zombie_index].x < -0.4f) {
                state->lost = 1;
                reward -= 20.0f;
                break;
            }
        }
        if (state->time >= pvz_config.final_wave_time
            && state->zombie_count == 0
            && state->next_spawn == PVZ_NEVER
        ) {
            state->won = 1;
            reward += 20.0f;
        }
        if (state->time >= pvz_config.max_time && !state->won && !state->lost) {
            state->truncated = 1;
        }
        reward -= 0.002f;
        if (!legal) {
            reward += pvz_config.invalid_action_penalty;
        }

        env->episode_return += reward;
        env->episode_steps++;
        rewards[env_index] = reward;
        int terminal = state->won || state->lost || state->truncated;
        terminals[env_index] = terminal;
        if (terminal) {
            env->log.perf += state->won;
            env->log.score += env->episode_return;
            env->log.episode_return += env->episode_return;
            env->log.episode_length += env->episode_steps;
            env->log.wins += state->won;
            env->log.losses += state->lost;
            env->log.truncations += state->truncated;
            env->log.kills += state->kills;
            env->log.invalid_actions += env->episode_invalid_actions;
            env->log.n += 1.0f;
            uint64_t next_seed = pvz_random_u64(state);
            pvz_reset_state(state, next_seed);
            env->episode_return = 0.0f;
            env->episode_steps = 0;
            env->episode_invalid_actions = 0;
        }
    }
    __syncwarp();
    if (active) {
        pvz_write_observation(
            &envs[env_index].state,
            observations + (long)env_index * OBS_SIZE,
            lane);
    }
}

Env* puf_vec_create(
    int total_agents, Dict* env_kwargs, obs_t* observations,
    float* actions, float* rewards, float* terminals
) {
    assert(total_agents > 0);
    PvzConfig config = {
        .tick_seconds = dict_get(env_kwargs, "tick_seconds"),
        .max_time = dict_get(env_kwargs, "max_time"),
        .starting_sun = dict_get(env_kwargs, "starting_sun"),
        .passive_sun_interval = dict_get(env_kwargs, "passive_sun_interval"),
        .spawn_interval = dict_get(env_kwargs, "spawn_interval"),
        .final_wave_time = dict_get(env_kwargs, "final_wave_time"),
        .invalid_action_penalty = dict_get(env_kwargs, "invalid_action_penalty"),
        .max_zombies = dict_get(env_kwargs, "max_zombies"),
    };
    assert(config.tick_seconds > 0.0f);
    assert(config.max_time > 0.0f);
    assert(config.passive_sun_interval > 0.0f);
    assert(config.spawn_interval > 0.0f);
    assert(config.final_wave_time >= 0.0f);
    assert(config.max_zombies > 0 && config.max_zombies <= PVZ_MAX_ZOMBIES);
    assert(cudaMemcpyToSymbol(pvz_config, &config, sizeof(config)) == cudaSuccess);

    uint64_t seed = dict_get(env_kwargs, "seed");
    Env* host_envs = (Env*)calloc(total_agents, sizeof(Env));
    assert(host_envs != NULL);
    for (int env_index = 0; env_index < total_agents; env_index++) {
        host_envs[env_index].num_agents = 1;
        host_envs[env_index].agents[0].policy = 0;
        host_envs[env_index].state.rng_state = seed + env_index;
    }

    Env* envs = NULL;
    assert(cudaMalloc((void**)&envs, total_agents * sizeof(Env)) == cudaSuccess);
    assert(cudaMemcpy(
        envs, host_envs, total_agents * sizeof(Env), cudaMemcpyHostToDevice
    ) == cudaSuccess);
    free(host_envs);

    pvz_gpu.envs = envs;
    pvz_gpu.num_envs = total_agents;
    pvz_gpu.observations = observations;
    pvz_gpu.actions = actions;
    pvz_gpu.rewards = rewards;
    pvz_gpu.terminals = terminals;
    pvz_gpu.stream = 0;
    return envs;
}

void puf_bind_stream(cudaStream_t stream) {
    pvz_gpu.stream = stream;
}

void puf_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
}

void puf_reset(Env* envs) {
    int thread_count = pvz_gpu.num_envs * PVZ_THREADS_PER_ENV;
    int grid_size = (thread_count + PVZ_CUDA_BLOCK_SIZE - 1)
        / PVZ_CUDA_BLOCK_SIZE;
    pvz_reset_kernel<<<grid_size, PVZ_CUDA_BLOCK_SIZE, 0, pvz_gpu.stream>>>(
        pvz_gpu.envs,
        pvz_gpu.observations,
        pvz_gpu.rewards,
        pvz_gpu.terminals,
        pvz_gpu.num_envs);
    assert(cudaGetLastError() == cudaSuccess);
}

void puf_step(Env* envs) {
    int thread_count = pvz_gpu.num_envs * PVZ_THREADS_PER_ENV;
    int grid_size = (thread_count + PVZ_CUDA_BLOCK_SIZE - 1)
        / PVZ_CUDA_BLOCK_SIZE;
    pvz_step_kernel<<<grid_size, PVZ_CUDA_BLOCK_SIZE, 0, pvz_gpu.stream>>>(
        pvz_gpu.envs,
        pvz_gpu.actions,
        pvz_gpu.observations,
        pvz_gpu.rewards,
        pvz_gpu.terminals,
        pvz_gpu.num_envs);
    assert(cudaGetLastError() == cudaSuccess);
}

void puf_render(Env* envs) {
}

void puf_close(Env* envs) {
    assert(cudaFree(pvz_gpu.envs) == cudaSuccess);
    pvz_gpu.envs = NULL;
}
