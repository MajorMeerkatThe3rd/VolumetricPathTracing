// smallpaint by karoly zsolnai - zsolnai@cg.tuwien.ac.at
// volumetric path tracing implementation by michael oppitz - e1227129@student.tuwien.ac.at
// compilation by: g++ smallpaint_fixed.cpp -O3 -std=gnu++0x -fopenmp
// uses >=gcc-4.5.0
// render, modify, create new scenes, tinker around, and most of all:
// have fun!
// 
// If you have problems/DLL hell with the compilation under Windows and
// MinGW, you can add these flags: -static-libgcc -static-libstdc++
//
// This program is used as an educational learning tool on the Rendering
// course at TU Wien. Course webpage:
// http://cg.tuwien.ac.at/courses/Rendering/

#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <ctime>
#include <vector>
#include <string>
#include <random>
#include <cstdint>
#include <algorithm>
#include <omp.h>

// Helpers for random number generation
std::mt19937 mersenneTwister;
std::uniform_real_distribution<double> uniform;

#define RND (2.0*uniform(mersenneTwister)-1.0)
#define RND2 (uniform(mersenneTwister))

#define PI 3.1415926536
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))
#define SIGMA_A .1
#define SIGMA_S .25

#define UNI 0
#define EXP 1

#define PL 0
#define RL 1

int sampling_type = 0;

const int width = 512, height = 512;
const double inf = 1e3;
const double eps = 1e-6;
using namespace std;

struct Vec {
	double x, y, z;
	Vec(double x0, double y0, double z0) { x = x0; y = y0; z = z0; }
	Vec(double w0) { x = w0; y = w0; z = w0; }
	Vec() { x = 0; y = 0; z = 0; }
	Vec operator+(const Vec &b) const { return Vec(x + b.x, y + b.y, z + b.z); }
	Vec operator-(const Vec &b) const { return Vec(x - b.x, y - b.y, z - b.z); }
	Vec operator*(double b) const { return Vec(x*b, y*b, z*b); }
	Vec operator*(const Vec &b) const { return Vec(x*b.x, y*b.y, z*b.z); }
	Vec operator/(double b) const { return Vec(x / b, y / b, z / b); }
	Vec mult(const Vec &b) const { return Vec(x*b.x, y*b.y, z*b.z); }
	Vec& norm() { return *this = *this * (1 / sqrt(x*x + y * y + z * z)); }
	double length() { return sqrt(x*x + y * y + z * z); }
	double dot(const Vec &b) const { return x * b.x + y * b.y + z * b.z; }
	Vec operator%(const Vec &b) const { return Vec(y*b.z - z * b.y, z*b.x - x * b.z, x*b.y - y * b.x); }
	//	double& operator[](size_t i) { return data[i]; }
	const double& operator[](size_t i) const { return i == 0 ? x : (i == 1 ? y : z); }
};

// given v1, set v2 and v3 so they form an orthonormal system
// (we assume v1 is already normalized)
void ons(const Vec& v1, Vec& v2, Vec& v3) {
	if (std::abs(v1.x) > std::abs(v1.y)) {
		// project to the y = 0 plane and construct a normalized orthogonal vector in this plane
		double invLen = 1.f / sqrtf(v1.x * v1.x + v1.z * v1.z);
		v2 = Vec(-v1.z * invLen, 0.0f, v1.x * invLen);
	}
	else {
		// project to the x = 0 plane and construct a normalized orthogonal vector in this plane
		double invLen = 1.0f / sqrtf(v1.y * v1.y + v1.z * v1.z);
		v2 = Vec(0.0f, v1.z * invLen, -v1.y * invLen);
	}
	v3 = v1 % v2;
}

// Rays have origin and direction.
// The direction vector should always be normalized.
struct Ray {
	Vec o, d;
	Ray(Vec o0 = 0, Vec d0 = 0) { o = o0, d = d0.norm(); }
};

// Input is the pixel offset, output is the appropriate coordinate
// on the image plane
Vec camcr(const double x, const double y) {
	double w = width;
	double h = height;
	double fovx = PI / 4;
	double fovy = (h / w) * fovx;
	return Vec(((2 * x - w) / w) * tan(fovx),
		-((2 * y - h) / h) * tan(fovy),
		-1.0);
}

Vec sampleSphere(double u1, double u2) {
	const double cos_theta = u1 * 2.0 - 1.0; // remap to -1 .. 1
	const double r = sqrt(1.0 - cos_theta*cos_theta);
	const double phi = 2 * PI * u2;
	return Vec(cos(phi)*r, sin(phi)*r, cos_theta);
}

double pfHenyeyGreenstein(Vec v1, Vec v2, double g) {
	double cos_theta = v1.dot(v2);
	double g_squ = g * g;
	return (1.0 / (4.0*PI)) * ((1.0 - g_squ) / pow(1.0 + g_squ - 2.0*g*cos_theta, 1.5));
}

void sampleUniform(double &x, double &pdf) {
	x = RND2 * inf;
	pdf = 1.0 / inf;
}

void sampleExponential(double &x, double &pdf) {
	double minU = exp(-(SIGMA_A+SIGMA_S) * inf);
	double a = RND2 * (1.0 - minU) + minU;

	x = -log(a) / (SIGMA_A + SIGMA_S);
	pdf = (SIGMA_A + SIGMA_S) * a / (1.0 - minU);
}

void sampleEquiAngular(const Ray& ray, const Vec& lightPos, double &distance, double& probability) {
	// Get closest point to light along the ray
	double tclose = (lightPos - ray.o).dot(ray.d);

	// Closest distance to light from ray
	double rayDist = (ray.o + ray.d * tclose - lightPos).length();

	// angles spanned by light, closest point and ray origin or end
	double thetaA = atan((0.0 - tclose) / rayDist);
	double thetaB = atan((inf - tclose) / rayDist);

	double u = RND2;
	double phi = (1 - u) * thetaA + u * thetaB;

	// sample ray position based on angle
	double t = rayDist * tan(phi); // relative to closest point
	distance = tclose + t;
	probability = rayDist / ((thetaB - thetaA)*(rayDist*rayDist + t * t)); //see: https://www.solidangle.com/research/s2011_equiangular_slides.pdf
}

double g = 0.0;
int lightType = 0;

void trace(Ray &ray, Ray light, int depth, Vec& clr) {

	if (PL == lightType) {

		// sample camera ray
		double camDist, cameraPdf;
		if (sampling_type == UNI) {
			sampleUniform(camDist, cameraPdf);
		} else if (sampling_type == EXP) {
			sampleExponential(camDist, cameraPdf);
		}
		Vec cameraSample = ray.o + ray.d * camDist;
		Vec lightSample = light.o;

		// connect to light and check shadow ray
		Vec lightVec = light.o - cameraSample;
		double d = lightVec.length();

		// accumulate particle response if not occluded
		double trans = exp(-(SIGMA_A + SIGMA_S) * (d + camDist));
		double geomTerm = 1.0 / (d*d);
		//double pf = pfHenyeyGreenstein(ray.d, lightVec.norm(), g);

		clr = clr + Vec(10.0)*(1.0/(4.0*PI))*geomTerm*trans*SIGMA_S / (cameraPdf);
	} else if (RL == lightType) {


		// sample light ray
		double lightDist, lightPdf;
		if (sampling_type == UNI) {
			sampleUniform(lightDist, lightPdf);
		} else if (sampling_type == EXP) {
			sampleExponential(lightDist, lightPdf);
		}
		Vec lightSample = light.o + light.d * lightDist;

		// sample camera ray
		double camDist, cameraPdf;
		if (sampling_type == UNI) {
			sampleUniform(camDist, cameraPdf);
		} else if (sampling_type == EXP) {
			sampleExponential(camDist, cameraPdf);
		} else {
			sampleEquiAngular(ray, lightSample, camDist, cameraPdf);
		}
		Vec cameraSample = ray.o + ray.d * camDist;

		// connect to light and check shadow ray
		Vec lightVec = lightSample - cameraSample;
		double d = lightVec.length();

		// accumulate particle response if not occluded
		double trans = exp(-(SIGMA_A + SIGMA_S) * (d + camDist + lightDist));
		double geomTerm = 1.0 / (d*d);
		double pf1 = pfHenyeyGreenstein(light.d*-1.0, lightVec.norm(), g);
		double pf2 = pfHenyeyGreenstein(ray.d, lightVec.norm(), g);

		clr = clr + Vec(100.0)*pf1*pf2*geomTerm*trans*SIGMA_S*SIGMA_S / (cameraPdf*lightPdf);
	}
	
}

int main() {
	srand(time(NULL));
	Vec **pix = new Vec*[width];
	for (int i = 0; i < width; i++) {
		pix[i] = new Vec[height];
	}

	const int spp = 50000;
	Ray light(Vec(0, 0, -2), Vec(0, 1, 0));

	///lightType = 0;
	lightType = 1;

	///sampling_type = 0;
	sampling_type = 1;
	
	g = -0.9;
	//g = 0.0;
	//g = 0.9;

	for (int s = 0; s < spp; s++) {
		light.d = sampleSphere(RND2, RND2);
		omp_set_num_threads(8);
	#pragma omp parallel for schedule(dynamic)
		for (int col = 0; col < width; col++) {
			fprintf(stdout, "\rRendering: %1.0fspp %8.2f%%", spp, (double)s / spp * 100);
			for (int row = 0; row < height; row++) {
				Vec color;
				Vec cam = camcr(col, row); // construct image plane coordinates
				cam.x = cam.x + RND / 700; // anti-aliasing for free
				cam.y = cam.y + RND / 700;
				Ray ray(Vec(0, 0, 0), cam.norm());
				trace(ray, light, 0, color);
				pix[col][row] = pix[col][row] + color / spp; // write the contributions
			}
		}
	}

	string name = "volpath_spp_" + to_string(spp) + (g > 0 ? "_g_+" : g == 0 ? "_g_" : "_g_-") + to_string(abs(g)).substr(0, 3) + "_t_" + to_string(sampling_type) + ".ppm";
	FILE *f = fopen(name.c_str(), "w");
	fprintf(f, "P3\n%d %d\n%d\n ", width, height, 255);
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			fprintf(f, "%d %d %d ", min((int)(pix[col][row].x * 255.0), 255), min((int)(pix[col][row].y * 255.0), 255), min((int)(pix[col][row].z * 255.0), 255));
		}
		fprintf(f, "\n");
	}
	fclose(f);
	return 0;
}
