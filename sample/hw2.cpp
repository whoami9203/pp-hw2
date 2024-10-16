#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <mpi.h>
#include <lodepng.h>

#define GLM_FORCE_SWIZZLE  // vec3.xyz(), vec3.xyx() ...ect, these are called "Swizzle".
// https://glm.g-truc.net/0.9.1/api/a00002.html
//
#include <glm/glm.hpp>
// for the usage of glm functions
// please refer to the document: http://glm.g-truc.net/0.9.9/api/a00143.html
// or you can search on google with typing "glsl xxx"
// xxx is function name (eg. glsl clamp, glsl smoothstep)

#define pi 3.1415926535897932384626433832795

typedef glm::dvec2 vec2;  // doube precision 2D vector (x, y) or (u, v)
typedef glm::dvec3 vec3;  // 3D vector (x, y, z) or (r, g, b)
typedef glm::dvec4 vec4;  // 4D vector (x, y, z, w)
typedef glm::dmat3 mat3;  // 3x3 matrix

unsigned int num_threads;  // number of thread
unsigned int width;        // image width
unsigned int height;       // image height
vec2 iResolution;          // just for convenience of calculation

int AA = 2;  // anti-aliasing

double power = 8.0;           // the power of the mandelbulb equation
double md_iter = 24;          // the iteration count of the mandelbulb
double ray_step = 10000;      // maximum step of ray marching
double shadow_step = 1500;    // maximum step of shadow casting
double step_limiter = 0.2;    // the limit of each step length
double ray_multiplier = 0.1;  // prevent over-shooting, lower value for higher quality
double bailout = 2.0;         // escape radius
double eps = 0.0005;          // precision
double FOV = 1.5;             // fov ~66deg
double far_plane = 100.;      // scene depth

vec3 camera_pos;  // camera position in 3D space (x, y, z)
vec3 target_pos;  // target position in 3D space (x, y, z)

unsigned char* raw_image;  // 1D image
unsigned char** image;     // 2D image
unsigned char* final_image;
int batch_size = 32;
vec3 ro;    // ray (camera) origin
vec3 ta;    // target position
vec3 cf;    // forward vector
vec3 cs;    // right (side) vector
vec3 cu;    // up vector

// save raw_image to PNG file
void write_png(const char* filename) {
    unsigned error = lodepng_encode32_file(filename, final_image, width, height);

    if (error) printf("png error %u: %s\n", error, lodepng_error_text(error));
}

// mandelbulb distance function (DE)
// v = v^8 + c
// p: current position
// trap: for orbit trap coloring : https://en.wikipedia.org/wiki/Orbit_trap
// return: minimum distance to the mandelbulb surface
double md(vec3 p, double& trap) {
    vec3 v = p;
    double dr = 1.;             // |v'|
    double r = glm::length(v);  // r = |v| = sqrt(x^2 + y^2 + z^2)
    trap = r;

    for (int i = 0; i < md_iter; ++i) {
        double theta = glm::atan(v.y, v.x) * power;
        double phi = glm::asin(v.z / r) * power;
        dr = power * glm::pow(r, power - 1.) * dr + 1.;
        v = p + glm::pow(r, power) *
                    vec3(cos(theta) * cos(phi), cos(phi) * sin(theta), -sin(phi));  // update vk+1

        // orbit trap for coloring
        trap = glm::min(trap, r);

        r = glm::length(v);      // update r
        if (r > bailout) break;  // if escaped
    }
    return 0.5 * log(r) * r / dr;  // mandelbulb's DE function
}

// scene mapping
double map(vec3 p, double& trap) {
    // vec2 rt = vec2(cos(pi / 2.), sin(pi / 2.));
    // vec3 rp = mat3(1., 0., 0., 0., rt.x, -rt.y, 0., rt.y, rt.x) *
    //           p;  // rotation matrix, rotate 90 deg (pi/2) along the X-axis
    vec3 rp = vec3(p.x, -p.z, p.y);
    return md(rp, trap);
}

// dummy function
// becase we dont need to know the ordit trap or the object ID when we are calculating the surface
// normal
double map(vec3 p) {
    double dmy;  // dummy
    // int dmy2;    // dummy2
    vec3 rp = vec3(p.x, -p.z, p.y); // rotation matrix, rotate 90 deg (pi/2) along the X-axis
    return md(rp, dmy);
}

// simple palette function (borrowed from Inigo Quilez)
// see: https://www.shadertoy.com/view/ll2GD3
vec3 pal(double t, vec3 a, vec3 b, vec3 c, vec3 d) {
    return a + b * glm::cos(2. * pi * (c * t + d));
}

// second march: cast shadow
// also borrowed from Inigo Quilez
// see: http://www.iquilezles.org/www/articles/rmshadows/rmshadows.htm
double softshadow(vec3 ro, vec3 rd, double k) {
    double res = 1.0;
    double t = 0.;  // total distance
    for (int i = 0; i < shadow_step; ++i) {
        double h = map(ro + rd * t);
        res = glm::min(
            res, k * h / t);  // closer to the objects, k*h/t terms will produce darker shadow
        if (res < 0.02) return 0.02;
        t += glm::clamp(h, .001, step_limiter);  // move ray
    }
    return glm::clamp(res, .02, 1.);
}

// use gradient to calc surface normal
vec3 calcNor(vec3 p) {
    vec2 e = vec2(eps, 0.);
    return normalize(vec3(map(p + e.xyy()) - map(p - e.xyy()),  // dx
        map(p + e.yxy()) - map(p - e.yxy()),                    // dy
        map(p + e.yyx()) - map(p - e.yyx())                     // dz
        ));
}

// first march: find object's surface
double trace(vec3 ro, vec3 rd, double& trap) {
    double t = 0;    // total distance
    double len = 0;  // current distance

    for (int i = 0; i < ray_step; ++i) {
        len = map(ro + rd * t, trap);  // get minimum distance from current ray position to the object's surface
        if (glm::abs(len) < eps || t > far_plane) break;
        t += len * ray_multiplier;
    }
    return t < far_plane
               ? t
               : -1.;  // if exceeds the far plane then return -1 which means the ray missed a shot
}

void process_rows(int start_row, int end_row){
    //---start rendering
    #pragma omp parallel for schedule(dynamic)
    for (int i = start_row; i < end_row; ++i) {
        //#pragma omp simd
        for (int j = 0; j < width; ++j) {
            vec4 fcol(0.);  // final color (RGBA 0 ~ 1)
            
            // anti aliasing
            for (int m = 0; m < AA; ++m) {
                for (int n = 0; n < AA; ++n) {
                    vec2 p = vec2(j, i) + vec2(m, n) / (double)AA;

                    //---convert screen space coordinate to (-ap~ap, -1~1)
                    // ap = aspect ratio = width/height
                    vec2 uv = (-iResolution.xy() + 2. * p) / iResolution.y;
                    uv.y *= -1;  // flip upside down
                    //---
                    
                    vec3 rd = glm::normalize(uv.x * cs + uv.y * cu + FOV * cf);  // ray direction
                    //---

                    //---marching
                    double trap;  // orbit trap
                    // int objID;    // the object id intersected with
                    double d = trace(ro, rd, trap);
                    //---

                    //---lighting
                    vec3 col(0.);                          // color
                    vec3 sd = glm::normalize(camera_pos);  // sun direction (directional light)
                    vec3 sc = vec3(1., .9, .717);          // light color
                    //---

                    //---coloring
                    if (d < 0.) {        // miss (hit sky)
                        col = vec3(0.);  // sky color (black)
                    } else {
                        vec3 pos = ro + rd * d;              // hit position
                        vec3 nr = calcNor(pos);              // get surface normal
                        vec3 hal = glm::normalize(sd - rd);  // blinn-phong lighting model (vector
                                                             // h)
                        // for more info:
                        // https://en.wikipedia.org/wiki/Blinn%E2%80%93Phong_shading_model

                        // use orbit trap to get the color
                        col = pal(trap - .4, vec3(.5), vec3(.5), vec3(1.),
                            vec3(.0, .1, .2));  // diffuse color
                        vec3 ambc = vec3(0.3);  // ambient color
                        double gloss = 32.;     // specular gloss

                        // simple blinn phong lighting model
                        double amb =
                            (0.7 + 0.3 * nr.y) *
                            (0.2 + 0.8 * glm::clamp(0.05 * log(trap), 0.0, 1.0));  // self occlution
                        double sdw = softshadow(pos + .001 * nr, sd, 16.);         // shadow
                        double dif = glm::clamp(glm::dot(sd, nr), 0., 1.) * sdw;   // diffuse
                        double spe = glm::pow(glm::clamp(glm::dot(nr, hal), 0., 1.), gloss) *
                                     dif;  // self shadow

                        vec3 lin(0.);
                        lin += ambc * (.05 + .95 * amb);  // ambient color * ambient
                        lin += sc * dif * 0.8;            // diffuse * light color * light intensity
                        col *= lin;

                        col = glm::pow(col, vec3(.7, .9, 1.));  // fake SSS (subsurface scattering)
                        col += spe * 0.8;                       // specular
                    }
                    //---

                    col = glm::clamp(glm::pow(col, vec3(.4545)), 0., 1.);  // gamma correction
                    fcol += vec4(col, 1.);
                }
            }

            fcol /= (double)(AA * AA);
            // convert double (0~1) to unsigned char (0~255)
            fcol *= 255.0;
            image[i - start_row][4 * j + 0] = (unsigned char)fcol.r;  // r
            image[i - start_row][4 * j + 1] = (unsigned char)fcol.g;  // g
            image[i - start_row][4 * j + 2] = (unsigned char)fcol.b;  // b
            image[i - start_row][4 * j + 3] = 255;                    // a
        }
    }
}

int main(int argc, char** argv) {
    // ./source [num_threads] [x1] [y1] [z1] [x2] [y2] [z2] [width] [height] [filename]
    // num_threads: number of threads per process
    // x1 y1 z1: camera position in 3D space
    // x2 y2 z2: target position in 3D space
    // width height: image size
    // filename: filename
    assert(argc == 11);

    auto start_all = std::chrono::high_resolution_clock::now();

    MPI_Init(&argc, &argv);  // Initialize MPI

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get process ID (rank)
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get number of processes

    //---init arguments
    num_threads = atoi(argv[1]);
    camera_pos = vec3(atof(argv[2]), atof(argv[3]), atof(argv[4]));
    target_pos = vec3(atof(argv[5]), atof(argv[6]), atof(argv[7]));
    width = atoi(argv[8]);
    height = atoi(argv[9]);

    double total_pixel = width * height;
    double current_pixel = 0;

    iResolution = vec2(width, height);
    //---

    //---create image
    raw_image = new unsigned char[width * batch_size * 4];
    image = new unsigned char*[batch_size];

    for (int i = 0; i < batch_size; ++i) {
        image[i] = raw_image + i * width * 4;
    }
    //---

    //--- Gather Results ---//
    int* sendcounts = nullptr;
    int* displs = nullptr;
    if (rank == 0) {
        // Only on the master process, allocate space for the final image
        final_image = new unsigned char[width * height * 4];  // Full image size

        // Allocate arrays to store the number of pixels sent by each process and the displacement
        sendcounts = new int[size];
        displs = new int[size];
    }

    ro = camera_pos;               // ray (camera) origin
    ta = target_pos;               // target position
    cf = glm::normalize(ta - ro);  // forward vector
    cs = glm::normalize(glm::cross(cf, vec3(0., 1., 0.)));  // right (side) vector
    cu = glm::normalize(glm::cross(cs, cf));          // up vector

    auto start = std::chrono::high_resolution_clock::now();

    //int rows_remaining = height;
    MPI_Request request;
    MPI_Status mpi_status;
    int flag;  // To check if the message has arrived
    int rows_per_process;
    int offset = 0;
    int current_row = 0;
    int rows_done = 0;
    int row_info[2];

    if (rank == 0){
        for (int p = 1; p < size; ++p) {
            rows_per_process = std::min(batch_size, (int)(height) - current_row);
            row_info[0] = current_row;
            row_info[1] = current_row + rows_per_process;
            current_row += rows_per_process;
            sendcounts[p] = rows_per_process * width * 4;
            displs[p] = offset;
            offset += sendcounts[p];

            if(rows_per_process <= 0)
                break;
            MPI_Send(row_info, 2, MPI_INT, p, 0, MPI_COMM_WORLD);
        }

        while (current_row < height){
            rows_per_process = std::min(batch_size << 1, (int)(height) - current_row);
            row_info[0] = current_row;
            row_info[1] = current_row + rows_per_process;

            current_row += rows_per_process;
            rows_done += rows_per_process;
            sendcounts[0] = rows_per_process * width * 4;
            displs[0] = offset;
            offset += sendcounts[0];

            // map image to final image
            int index = 0;
            for(int i=row_info[0]; i<row_info[1]; ++i){
                image[index++] = final_image + i * width * 4;
            }

            // Master does its own work on the batch
            process_rows(row_info[0], row_info[1]);

            // Check the results from other processes non-blocking
            for (int p = 1; p < size; ++p) {
                MPI_Irecv(&rows_per_process, 1, MPI_INT, p, 0, MPI_COMM_WORLD, &request);

                // Use MPI_Test to check if the message has arrived
                MPI_Test(&request, &flag, MPI_STATUS_IGNORE);

                // if recv now, then handle it
                if(flag){
                    MPI_Recv(final_image + displs[p], rows_per_process * width * 4, MPI_UNSIGNED_CHAR, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    rows_per_process = std::min(batch_size, (int)(height) - current_row);
                    row_info[0] = current_row;
                    row_info[1] = current_row + rows_per_process;

                    current_row += rows_per_process;
                    rows_done += rows_per_process;
                    sendcounts[p] = rows_per_process * width * 4;
                    displs[p] = offset;
                    offset += sendcounts[p];

                    if(current_row >= height)
                        break;
                    MPI_Send(row_info, 2, MPI_INT, p, 0, MPI_COMM_WORLD);
                }
            }
        }

        int sender_rank;
        while(rows_done < height){
            MPI_Recv(&rows_per_process, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &mpi_status);
            sender_rank = mpi_status.MPI_SOURCE;
            rows_done += rows_per_process;
            MPI_Recv(final_image + displs[sender_rank], rows_per_process * width * 4, MPI_UNSIGNED_CHAR, sender_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    else
    {
        while (current_row < height){
            current_row += batch_size * size;
            MPI_Recv(row_info, 2, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            rows_per_process = row_info[1] - row_info[0];
            process_rows(row_info[0], row_info[1]);

            // Send the results to master processes
            MPI_Send(&rows_per_process, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            MPI_Send(raw_image, rows_per_process * width * 4, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "rank: " << rank << " Main Program Time: " << elapsed_seconds.count() * 1000.0 << " ms" << std::endl;

    //--- Saving Image ---//
    if (rank == 0) {
        write_png(argv[10]);  // Write final image on process 0
        delete[] final_image;
        auto end_all = std::chrono::high_resolution_clock::now();
        elapsed_seconds = end_all - start_all;
        std::cout << "Total Program Time: " << elapsed_seconds.count() * 1000.0 << " ms" << std::endl;
    }

    //---saving image
    // write_png(argv[10]);
    //---

    //---finalize
    delete[] raw_image;
    delete[] image;
    //---

    MPI_Finalize();

    return 0;
}
