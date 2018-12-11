#include "src/distfunc.cl"

#define MAX_DIST 10000.0
#define MAX_BOUNCES 10
#define SPEED_OF_SOUND 343.0
#define SAMPLE_RATE 44100.0
#define DIFFUSE true

#define NUM_TARGETS 2
#define NUM_RUNS 8
#define NUM_SUBSCENES 32
#define IMPULSE_LEN 65536

#define SOURCE (double3)(0.02385768, 0, 4.917285)
#define SOURCE_THETA_MIN 0
#define SOURCE_THETA_MAX PI
#define SOURCE_PHI_MIN 0
#define SOURCE_PHI_MAX (2.0*PI)

#define DEBUG_ORIGIN (double3)(0.0,0.0,4.9)
#define DEBUG_DIR (double3)(0.0,-1.0,0.0)
#define DEBUG_LIGHT_DIR (double3)(0.4,1,0.9)
#define DEBUG_MAX_DIST 100.0

Target sdf_target(double3 p, int subscene_id) {
    Target d = INF_TARGET;
    double a = (double)subscene_id/NUM_SUBSCENES*PI*2.0;
    d = target_min(d, target(sdf_sphere(p-(double3)(sin(a)*4.0,     0.0, cos(a)*4.0    ), 0.5), 0, 1));
    d = target_min(d, target(sdf_sphere(p-(double3)(sin(a+0.1)*4.0, 0.0, cos(a+0.1)*4.0), 0.5), 1, 1));
    return d;
}

double sdf_scene(double3 p, int subscene_id) {
    double d = INFINITY;

    d = min(d, -sdf_box(p, (double3)(5.0,5.0,5.0)));
    d = min(d, sdf_sphere(p, 4.0));
    d = min(d, sdf_target(p, subscene_id).dist);

    return d;
}

