#include <stdio.h>
#include "scene_defines.h"

int main() {
    printf(
"(\n"
"    num_targets: %d,\n"
"    num_runs: %d,\n"
"    num_subscenes: %d,\n"
"    impulse_len: %d,\n"
")\n", NUM_TARGETS, NUM_RUNS, NUM_SUBSCENES, IMPULSE_LEN);
}
