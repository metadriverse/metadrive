
#include "pandabase.h"
#include "pnmImage.h"

#include <stdlib.h>
#include <time.h>
#include <random>
#include <iostream>
#include <fstream>
#include <functional>
#include <string>

using namespace std;

#define TWO_PI 6.28318530718f
#define square(x) ((x) * (x))

struct Vec {
  float x, y, z;
  inline float length_sq() const {
    return x*x + y*y + z*z;
  }
  inline float dist_sq(const Vec& other) const {
    return square(x - other.x) + square(y - other.y) + square(z - other.z);
  }

  inline float angle_signed() const {
    return atan2(y, x);
  }

  friend inline ostream& operator<<(ostream& stream, const Vec& v) {
    return stream << "Vec(" << v.x << ", " << v.y << ", " << v.z << ")";
  }
};

int compare_vec(const void * a, const void * b)
{
  return ((Vec*)a)->angle_signed() - ((Vec*)b)->angle_signed();
}

class PDGenerator
{

PUBLISHED:

  PDGenerator(int sample_count, int evaluations, int num_tries, int is_3d_)
  {
    is_3d = is_3d_;
    rnd_device = new random_device();
    rnd_generator = new mt19937(rnd_device->operator()());
    rng = new uniform_real_distribution<float>(-1.0f, 1.0f);
    randvec = is_3d ? &PDGenerator::randvec_3d : &PDGenerator::randvec_2d;

    cout << "Sample count = " << sample_count << endl;
    cout << "Evaluations = " << evaluations << endl;
    cout << "Num tries = " << num_tries << endl;

    Vec* sample_points = new Vec[sample_count];
    Vec* best_points = new Vec[sample_count];

    float best_score = -100000.0;

    for (size_t try_count = 0; try_count < num_tries; ++try_count) {
      // Generate the initial sample point
      sample_points[0] = randvec(this);

      // Fill the points
      for (size_t i = 1; i < sample_count; ++i) {
        Vec best_point;
        float best_point_min_dist = 0.0f;

        // Find the point which has the greatest minimum distance to all current points
        for (size_t k = 0; k < evaluations; ++k) {
          Vec point = randvec(this);
          float min_dist = 10000.0;

          // Get the minimum distance to all current points
          for (size_t j = 0; j < i; ++j) {
            min_dist = min(min_dist, point.dist_sq(sample_points[j]));
          }

          if (min_dist > best_point_min_dist) {
            best_point = point;
            best_point_min_dist = min_dist;
          }
        }

        sample_points[i] = best_point;
      }

      // Rate the points
      float score = 0.0;

      // Check for all points
      for (size_t i = 0; i < sample_count; ++i) {

        // Find minimum distance to all other points
        // float mindist = 10000.0;
        float mindist = 0.0;
        Vec point = sample_points[i];
        for (size_t j = 0; j < sample_count; ++j) {
          if (j == i) continue;
          // mindist = min(mindist, point.dist_sq(sample_points[j]));
          mindist += point.dist_sq(sample_points[j]);
        }
        score += mindist * mindist;
      }

      if (try_count % 20 == 0) {
        cout << "Try " << try_count << " has a score of " << score << endl;
      }

      if (score >= best_score) {
        memcpy(best_points, sample_points, sizeof(Vec) * sample_count);
        best_score = score;
      }

    }

    // Sort points based on their angle
    qsort(best_points, sample_count, sizeof(Vec), compare_vec);

    // write results
    string datatype = is_3d ? "vec3" : "vec2";
    ofstream output("disk.txt");

    output << "CONST_ARRAY " << datatype << " poisson_disk_" << (is_3d ? "3D" : "2D")
           << "_" << sample_count << "[" << sample_count << "] = " << datatype << "[](" << endl;

    // write out all points
    for (size_t i = 0; i < sample_count; ++i) {
      Vec v = best_points[i];
      output << "    " << datatype << "(" << v.x << ", " << v.y;
      if (is_3d) {
        output << ", " << v.z;
      }
      output << ")" << (i == sample_count - 1 ? "" : ",") << endl;
    }

    output << ");" << endl;

    output.close();

    // Plot the points
    PNMImage dest(256, 256);

    for (float rad = 0.0f; rad < TWO_PI; rad += 0.005f) {
      size_t x = int(sin(rad) * 66.0f) + 127;
      size_t y = int(cos(rad) * 66.0f) + 127;
      if ( int(rad / TWO_PI * 64.0f) % 2 == 0) {
        dest.set_xel(x, y, 0, 0.1, 0.1);
      }
    }

    for (size_t i = 0; i < sample_count; ++i) {
      Vec point = best_points[i];
      int px = int(point.x * 64.0) + 127;
      int py = int(point.y * 64.0) + 127;
      float f = float(i / float(sample_count)) * 0.8 + 0.2;
      dest.set_xel(px, py, f, f, f);
    }

    dest.write("result.png");
  }

  inline float frand() {
    return rng->operator()(*rnd_generator);
  }

  inline Vec randvec_3d() {
    float r = frand();

    // this is so wrong that its almost correct again
    float phi = frand() * TWO_PI * 0.5;
    float theta = (frand() * 0.5 + 0.5) * TWO_PI;
    float sin_phi = sin(phi);
    Vec v = { sin_phi * cos(theta) * r, sin_phi * sin(theta) * r,
              cos(phi) * r };
    return v;
  }

  inline Vec randvec_2d() {
    float r = frand() * 0.5 + 0.5;
    float phi = (frand() * 0.5 + 0.5) * TWO_PI;
    Vec v = { cos(phi) * r, sin(phi) * r, 0.0 };
    return v;
  }

  bool is_3d;

#ifndef INTERROGATE
  std::function<Vec(PDGenerator*)> randvec;
  std::random_device* rnd_device;
  mt19937* rnd_generator;
  uniform_real_distribution<float>* rng;
#endif

};
