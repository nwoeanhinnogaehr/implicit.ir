
#define PI 3.1415926535897932384626433832795

// need to split axis up into components because otherwise the driver gives CL_INVALID_COMMAND_QUEUE :(
typedef struct {
    double axisx;
    double axisy;
    double axisz;
    double dist;
    int id;
} Target;

#define INF_TARGET target(INFINITY, -1, 0)

Target target(double dist, int id, double3 axis) {
    Target t;
    t.dist = dist;
    t.id = id;
    t.axisx = axis.x;
    t.axisy = axis.y;
    t.axisz = axis.z;
    return t;
}

Target target_min(Target a, Target b) {
    if (a.dist < b.dist)
        return a;
    return b;
}

double sdf_sphere(double3 p, double rad) {
    return length(p) - rad;
}

double sdf_cylinder(double3 p, double3 c) {
    return length(p.xz - c.xy) - c.z;
}

double sdf_box(double3 p, double3 b) {
    double3 d = fabs(p) - b;
    return min(max(d.x, max(d.y, d.z)), 0.0) + length(max(d, 0.0));
}

double sdf_torus(double3 p, double2 t) {
    double2 q = (double2)(length(p.xz) - t.x, p.y);
    return length(q) - t.y;
}
