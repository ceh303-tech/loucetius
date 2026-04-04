/* Locetius Core Solver Library - Linux Version
 * 
 * This is the obfuscated Linux implementation of the proprietary
 * Spectral Swarm Engine solver.
 * 
 * API Version: 1.0
 * Platform: Linux x86-64
 * CUDA Support: 12.1+
 */

#ifndef LIBLOCETIUS_CORE_H
#define LIBLOCETIUS_CORE_H

#ifdef __cplusplus
extern "C" {
#endif

/* Solver configuration structure */
typedef struct {
    int num_variables;
    int num_swarms;
    int annealing_steps;
    float thermal_melt;
    float density_bias;
    int max_iterations;
    int phase3_enforce;
    int early_stopping_patience;
} SwarmConfig;

/* Result structure */
typedef struct {
    float *best_solution;
    float best_energy;
    float consensus;
    int num_iterations;
} SolverResult;

/* Version string */
const char* LOUCETIUS_get_version(void);

/* Solve QUBO problem */
SolverResult* LOUCETIUS_solve_qubo(
    float *Q_data,
    int *Q_row,
    int *Q_col,
    int Q_nnz,
    SwarmConfig config
);

/* Free result memory */
void LOUCETIUS_free_result(SolverResult *result);

#ifdef __cplusplus
}
#endif

#endif /* LIBLOCETIUS_CORE_H */
