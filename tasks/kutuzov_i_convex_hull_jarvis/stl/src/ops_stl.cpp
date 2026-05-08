#include "kutuzov_i_convex_hull_jarvis/stl/include/ops_stl.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <execution>
#include <numeric>
#include <thread>
#include <vector>

#include "kutuzov_i_convex_hull_jarvis/common/include/common.hpp"

namespace kutuzov_i_convex_hull_jarvis {

namespace {

constexpr double kEpsilon = 1e-9;
constexpr size_t kMinPointsForHull = 3;

inline unsigned GetNumThreads() {
  const unsigned hw = std::thread::hardware_concurrency();
  return (hw == 0) ? 1 : hw;
}

inline unsigned GetThreadIndex() {
  static const unsigned kNumThreads = GetNumThreads();
  static std::atomic<unsigned> next_id{0};
  thread_local unsigned tid = next_id.fetch_add(1, std::memory_order_relaxed) % kNumThreads;
  return tid;
}

struct BestCandidate {
  size_t index;
  double x;
  double y;
};

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
  auto it = std::min_element(std::execution::par, input.begin(), input.end(), [](const auto &a, const auto &b) {
    const double ax = std::get<0>(a);
    const double ay = std::get<1>(a);
    const double bx = std::get<0>(b);
    const double by = std::get<1>(b);
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

  if (points.size() < kMinPointsForHull) {
    GetOutput() = points;
    return true;
  }

  auto &output = GetOutput();
  output.clear();

  const size_t leftmost_idx = FindLeftmostPoint(points);

  const unsigned num_threads = GetNumThreads();
  std::vector<BestCandidate> locals(num_threads);

  std::vector<size_t> indices(points.size());
  std::iota(indices.begin(), indices.end(), static_cast<size_t>(0));

  size_t current_idx = leftmost_idx;
  double current_x = std::get<0>(points[current_idx]);
  double current_y = std::get<1>(points[current_idx]);

  while (true) {
    output.push_back(points[current_idx]);

    for (unsigned i = 0; i < num_threads; ++i) {
      locals[i] = {current_idx, current_x, current_y};
    }

    std::for_each(std::execution::par, indices.begin(), indices.end(), [&](size_t i) {
      if (i == current_idx) {
        return;
      }

      const unsigned tid = GetThreadIndex();
      BestCandidate &best = locals[tid];

      const auto &p = points[i];
      const double px = std::get<0>(p);
      const double py = std::get<1>(p);

      const double cross = (best.x - current_x) * (py - current_y) - (best.y - current_y) * (px - current_x);

      if (cross < -kEpsilon ||
          (std::abs(cross) < kEpsilon &&
           (px - current_x) * (px - current_x) + (py - current_y) * (py - current_y) >
               (best.x - current_x) * (best.x - current_x) + (best.y - current_y) * (best.y - current_y))) {
        best.index = i;
        best.x = px;
        best.y = py;
      }
    });

    size_t global_idx = locals[0].index;
    double global_x = locals[0].x;
    double global_y = locals[0].y;

    for (unsigned i = 1; i < num_threads; ++i) {
      if (locals[i].index == current_idx) {
        continue;
      }

      const double cross =
          (global_x - current_x) * (locals[i].y - current_y) - (global_y - current_y) * (locals[i].x - current_x);

      if (cross < -kEpsilon ||
          (std::abs(cross) < kEpsilon && DistanceSquared(current_x, current_y, locals[i].x, locals[i].y) >
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
