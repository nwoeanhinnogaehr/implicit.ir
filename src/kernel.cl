#pragma OPENCL EXTENSION cl_khr_fp64: enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable

#include "src/scene.cl"

void atomic_add_double(global double *addr, double val) {
    union {
        double d;
        ulong l;
    } old, new;
    do {
        old.d = *addr;
        new.d = old.d + val;
    } while (atom_cmpxchg((volatile global ulong *) addr, old.l, new.l) != old.l);
}

double3 reflect(double3 i, double3 n) {
    // from http://developer.download.nvidia.com/cg/reflect.html
    return i - 2.0 * n * dot(n, i);
}

double3 grad_sdf_scene(double3 p, int subscene_id) {
    // tetrahedron method
    const double eps = 0.00000001;
    const double2 c = (double2)(1,-1);
    return normalize(c.xxx*sdf_scene(p + c.xxx*eps, subscene_id) +
                     c.xyy*sdf_scene(p + c.xyy*eps, subscene_id) +
                     c.yxy*sdf_scene(p + c.yxy*eps, subscene_id) +
                     c.yyx*sdf_scene(p + c.yyx*eps, subscene_id));
}

double hash(double seed) {
    return fmod(sin(seed)*43758.5458957601829733, 1);
}

double3 cosineDirection(double seed, double3 normal) {
    // inspired by http://www.amietia.com/lambertnotangent.html
    double u = hash(125.32956735898 + seed)*2.0 - 1.0;
    double v = hash(14.593679587463 + seed)*2.0*PI;
    return normalize(normal + (double3)(sqrt(1.0 - u*u) * (double2)(cos(v), sin(v)), u) );
}

double intersect(double3 p, double3 dir, double eps, double maxd, int subscene_id) {
    double total = 0.0;
    double dist = maxd;
    do {
        dist = sdf_scene(p, subscene_id);
        p += dist * dir;
        total += dist;
    } while (dist > eps && total < maxd);
    return total;
}

// (0 < theta < PI, 0 < phi < 2PI, rad)
double3 from_spherical(double3 sph) {
    return (double3)(sin(sph.x)*cos(sph.y), sin(sph.x)*sin(sph.y), cos(sph.x))*sph.z;
}

void path_trace(
        double3 source, // start position
        double3 dir, // direction (should be unit)
        int subscene_id, // [0,NUM_SUBSCENES)
        int run, // [0,NUM_RUNS)
        Target *min_target, // output: closest target
        double *total_dist, // output: total distance
        double3 *intersect_pos, // output: target intersection position
        double3 *intersect_dir, // output: target intersection direction
        int *closest_bounce) { // output: number of bounces at target intersection point
    const double eps = 0.0000001;
    Target min_target_ = INF_TARGET;
    double total = 0.0;
    double dist;
    double3 p = source;
    int bounce_count = 0;
    //*intersect_pos = 0.0;
    //*intersect_dir = 0.0;
    for (int i = 0; i < MAX_BOUNCES; i++) {
        bounce_count++;
        // sphere tracing
        do {
            dist = sdf_scene(p, subscene_id);
            p += dist * dir;
            Target t = sdf_target(p, subscene_id);
            if (t.dist < min_target_.dist) {
                *closest_bounce = bounce_count;
                min_target_ = t;
                *intersect_dir = dir;
                *intersect_pos = p;
            }
            total += dist;
        } while (dist > eps && total < MAX_DIST);

        if (1/pow(total+1.0, 2.0) < 0.0001) {
            break; // contribution to impulse response is negligible
        }
        if (min_target_.dist < eps) {
            // hit target
            break;
        }
        // step back from surface before calculating gradient
        p-=0.000001*dir;

#if DIFFUSE
        dir = cosineDirection(hash(((i+1)*(run+1))*173.483297)
                +hash(p.x*179+100)
                +hash(p.y*199+200)
                +hash(p.z*211+300)
                +hash(400+dir.x+dir.y*153+dir.z*4997)
                +hash(500+total*357)
                +hash(600+min_target_.dist*811),
                grad_sdf_scene(p, subscene_id));
#else
        dir = reflect(dir, grad_sdf_scene(p, subscene_id));
#endif
    }
    *min_target = min_target_;
    *total_dist = total;
}

kernel void trace(
        int run,
        int subscene_id,
        global Target* g_min_target,
        global double* g_total_dist,
        global double3* g_intersect_pos,
        global double3* g_intersect_dir,
        global int *g_closest_bounce) {
    int id0 = get_global_id(0);
    int id1 = get_global_id(1);
    int id0res = get_global_size(0);
    int id1res = get_global_size(1);
    int idx = id0*id0res + id1;
    int width = (int)sqrt((float)id0res*id1res);
    int x = idx%width;
    int y = idx/width;
    double2 uv = (double2)((double)x / (double)width, (double)y / (double)width);
    uv += (double2)(hash(uv.y+run) / (double)width, hash(uv.x+run) / (double)width);

    // ray origin and direction
    double3 origin = SOURCE;
    double3 dir = from_spherical((double3)(
                uv.x*(SOURCE_THETA_MAX-SOURCE_THETA_MIN)+SOURCE_THETA_MIN,
                uv.y*(SOURCE_PHI_MAX-SOURCE_PHI_MIN)+SOURCE_PHI_MIN,
                1.0)); // 1.0 produces a unit vector

    Target min_target;
    double total_dist;
    double3 intersect_pos, intersect_dir;
    int closest_bounce;

    path_trace(origin, dir, subscene_id, run, &min_target, &total_dist, &intersect_pos, &intersect_dir, &closest_bounce);

    g_min_target[idx] = min_target;
    g_total_dist[idx] = total_dist;
    g_intersect_pos[idx] = intersect_pos;
    g_intersect_dir[idx] = intersect_dir;
    g_closest_bounce[idx] = closest_bounce;
}

kernel void scene_render(
        int subscene_id,
        global double3 *image) {
    int id0 = get_global_id(0);
    int id1 = get_global_id(1);
    int id0res = get_global_size(0);
    int id1res = get_global_size(1);
    int idx = id0*id0res + id1;
    int width = (int)sqrt((float)id0res*id1res);
    int x = idx%width;
    int y = idx/width;
    double2 uv = (double2)((double)x / (double)width, (double)y / (double)width);

    double3 origin = DEBUG_ORIGIN;
    double3 dir = normalize((double3)(uv*2.0-1.0, 1.0));

    const double maxd = DEBUG_MAX_DIST;
    double dist = intersect(origin, dir, 0.001, maxd, subscene_id);
    if (dist < maxd) {
        // found intersection, do shading
        double3 pt = origin + dir*(dist-0.01);
        double3 normal = grad_sdf_scene(pt, subscene_id);
        double3 light = normalize(DEBUG_LIGHT_DIR);
        double shade = max(0.0, -dot(light, normal));
        image[idx] = shade;
    } else {
        // no intersection
        image[idx] = 0.0;
    }
}

kernel void debug_render(
        global double3 *image,
        global const Target* g_min_target,
        global const double* g_total_dist,
        global const double3* g_intersect_pos,
        global const double3* g_intersect_dir,
        global const int *g_closest_bounce) {
    int id0 = get_global_id(0);
    int id1 = get_global_id(1);
    int id0res = get_global_size(0);
    int id1res = get_global_size(1);
    int idx = id0*id0res + id1;

    // multiple options for debug output, for experimentation

    //image[idx] += g_intersect_dir[idx];
    //image[idx] += g_intersect_dir[idx] / (1.0+g_total_dist[idx]);
    //image[idx] += (double3)(g_min_target[idx].dist, g_total_dist[idx], g_closest_bounce[idx]);
    //image[idx] += g_min_target[idx].dist / (1.0+g_total_dist[idx]);
    image[idx] += (g_intersect_dir[idx]*0.5+0.5) * g_min_target[idx].dist;
}

kernel void gen_impulse_response(
        global double *impulse,
        int impulse_len,
        global const Target* g_min_target,
        global const double* g_total_dist,
        global const double3* g_intersect_pos,
        global const double3* g_intersect_dir,
        global const int *g_closest_bounce) {
    int id0 = get_global_id(0);
    int id1 = get_global_id(1);
    int id0res = get_global_size(0);
    int id1res = get_global_size(1);
    int idx = id0*id0res + id1;

    Target target = g_min_target[idx];
    if (target.dist < 0.00001) {
        // this ray intersected a target
        double dist = g_total_dist[idx];
        double impulse_time = dist / SPEED_OF_SOUND * SAMPLE_RATE; // time in samples
        int impulse_idx = ((int)impulse_time) * NUM_TARGETS;
        double idx_fract = fmod(impulse_time, 1.0);
        if (impulse_idx+2*NUM_TARGETS < impulse_len) {
            double sample = pow(-1.0, g_closest_bounce[idx]) / pow(dist+1.0, 2.0); // attenuation
            double output = sample * max(0.0, dot(g_intersect_dir[idx], (double3)(target.axisx, target.axisy, target.axisz))); // angular filter
            // lerp between adjacent bins
            atomic_add_double(&impulse[impulse_idx+target.id], output*(1.0-idx_fract));
            atomic_add_double(&impulse[impulse_idx+target.id+NUM_TARGETS], output*idx_fract);
        }
    }
}
