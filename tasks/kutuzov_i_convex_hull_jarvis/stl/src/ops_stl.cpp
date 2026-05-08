#include "kutuzov_i_convex_hull_jarvis/stl/include/ops_stl.hpp"

#include <cmath>
#include <cstddef>
#include <thread>
#include <vector>

#include "kutuzov_i_convex_hull_jarvis/common/include/common.hpp"

namespace kutuzov_i_convex_hull_jarvis {

namespace {

struct BestCandidate {
  size_t index;
  double x, y;
};

inline unsigned GetNumThreads() {
  const unsigned hw = std::thread::hardware_concurrency();
  return (hw == 0) ? 1 : hw;
}

}  // anonymous namespace

KutuzovITestConvexHullSTL::KutuzovITestConvexHullSTL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

double KutuzovITestConvexHullSTL::DistanceSquared(double a_x, double a_y, double b_x, double b_y) {
  return (a_x - b_x) * (a_x - b_x) + (a_y - b_y) * (a_y - b_y);
}

double KutuzovITestConvexHullSTL::CrossProduct(double o_x, double o_y, double a_x, double a_y, double b_x, double b_y) {
  return (a_x - o_x) * (b_y - o_y) - (a_y - o_y) * (b_x - o_x);
}

size_t KutuzovITestConvexHullSTL::FindLeftmostPoint(const InType &input) {
  auto it = std::min_element(input.begin(), input.end(), [](const auto &a, const auto &b) {
    double ax = std::get<0>(a), ay = std::get<1>(a);
    double bx = std::get<0>(b), by = std::get<1>(b);
    return (ax < bx) || (ax == bx && ay < by);
  });
  return static_cast<size_t>(std::distance(input.begin(), it));
}

bool KutuzovITestConvexHullSTL::IsBetterPoint(double cross, double epsilon, double current_x, double current_y,
                                              double i_x, double i_y, double next_x, double next_y) {
  if (cross < -epsilon) {
    return true;
  }
  if (std::abs(cross) < epsilon) {
    return DistanceSquared(current_x, current_y, i_x, i_y) > DistanceSquared(current_x, current_y, next_x, next_y);
  }
  return false;
}

bool KutuzovITestConvexHullSTL::ValidationImpl() {
  return true;
}
bool KutuzovITestConvexHullSTL::PreProcessingImpl() {
  return true;
}

bool KutuzovITestConvexHullSTL::RunImpl() {
  const auto &points = GetInput();
  if (points.size() < 3) {
    GetOutput() = points;
    return true;
  }

  auto &output = GetOutput();
  output.clear();

  const double epsilon = 1e-9;
  const size_t n = points.size();

  const size_t leftmost_idx = FindLeftmostPoint(points);

  const unsigned num_threads = GetNumThreads();
  std::vector<BestCandidate> locals(num_threads);

  size_t current_idx = leftmost_idx;
  double current_x = std::get<0>(points[current_idx]);
  double current_y = std::get<1>(points[current_idx]);

  while (true) {
    output.push_back(points[current_idx]);

    for (unsigned i = 0; i < num_threads; ++i) {
      locals[i] = {current_idx, current_x, current_y};
    }

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    for (unsigned tid = 0; tid < num_threads; ++tid) {
      size_t start = (tid * n) / num_threads;
      size_t end = ((tid + 1) * n) / num_threads;
      if (start >= end) {
        continue;
      }

      threads.emplace_back([&, tid, start, end]() {
        BestCandidate &best = locals[tid];

        for (size_t i = start; i < end; ++i) {
          if (i == current_idx) {
            continue;
          }

          const auto &p = points[i];
          double px = std::get<0>(p);
          double py = std::get<1>(p);

          double cross = (best.x - current_x) * (py - current_y) - (best.y - current_y) * (px - current_x);

          if (cross < -epsilon ||
              (std::abs(cross) < epsilon &&
               (px - current_x) * (px - current_x) + (py - current_y) * (py - current_y) >
                   (best.x - current_x) * (best.x - current_x) + (best.y - current_y) * (best.y - current_y))) {
            best.index = i;
            best.x = px;
            best.y = py;
          }
        }
      });
    }

    for (auto &t : threads) {
      t.join();
    }

    size_t global_idx = locals[0].index;
    double global_x = locals[0].x;
    double global_y = locals[0].y;

    for (unsigned i = 1; i < num_threads; ++i) {
      if (locals[i].index == current_idx) {
        continue;
      }

      double cross =
          (global_x - current_x) * (locals[i].y - current_y) - (global_y - current_y) * (locals[i].x - current_x);

      if (cross < -epsilon ||
          (std::abs(cross) < epsilon && DistanceSquared(current_x, current_y, locals[i].x, locals[i].y) >
                                            DistanceSquared(current_x, current_y, global_x, global_y))) {
        global_idx = locals[i].index;
        global_x = locals[i].x;
        global_y = locals[i].y;
      }
    }

    current_idx = global_idx;
    current_x = global_x;
    current_y = global_y;

    if (current_idx == leftmost_idx) {
      break;
    }
  }

  return true;
}

bool KutuzovITestConvexHullSTL::PostProcessingImpl() {
  return true;
}

}  // namespace kutuzov_i_convex_hull_jarvis
