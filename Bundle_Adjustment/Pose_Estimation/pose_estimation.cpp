#include <Eigen/Core>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <sophus/se3.hpp>
#include <vector>

void find_feature_matches(const cv::Mat &img1, const cv::Mat &img2,
                          std::vector<cv::KeyPoint> &keypoints1,
                          std::vector<cv::KeyPoint> &keypoints2,
                          std::vector<cv::DMatch> &matches);

cv::Point2d pixel2cam(const cv::Point2d &p, const cv::Mat &K);

typedef std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
    VecVector2d;
typedef std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
    VecVector3d;

void BAg2o(const VecVector2d &points2d, const VecVector3d &points3d,
           const cv::Mat &K, Sophus::SE3d &pose);

int main(int argc, char **argv) {
  if (argc != 5) {
    std::cout << "Provide img1 img2 depth1 depth2" << std::endl;
    return 1;
  }

  cv::Mat K =
      (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

  cv::Mat img1 = cv::imread(argv[1]);
  cv::Mat img2 = cv::imread(argv[2]);

  std::vector<cv::KeyPoint> keypoints1, keypoints2;
  std::vector<cv::DMatch> matches;
  find_feature_matches(img1, img2, keypoints1, keypoints2, matches);

  // 3d points
  cv::Mat depth_img1 = cv::imread(argv[3]);
  std::vector<cv::Point3d> pts3d;
  std::vector<cv::Point2d> pts2d;
  for (auto match : matches) {
    unsigned short d = depth_img1.ptr<unsigned short>(int(
        keypoints1[match.queryIdx].pt.y))[int(keypoints1[match.queryIdx].pt.x)];
    if (d == 0) {
      continue;
    }
    float dd = d / 5000.0;
    cv::Point2d p1 = pixel2cam(keypoints1[match.queryIdx].pt, K);
    pts3d.push_back(cv::Point3d(p1.x * dd, p1.y * dd, dd));
    pts2d.push_back(keypoints2[match.trainIdx].pt);
  }
  std::cout << "3D-2D paire : " << pts3d.size() << std::endl;

  // 3D-2D pose estimation using PnP
  std::cout << "**************************" << std::endl;
  std::cout << "Pose Estimation using OpenCV SolvePnP" << std::endl;
  cv::Mat R, r, t;
  cv::solvePnP(pts3d, pts2d, K, cv::Mat(), r, t);
  cv::Rodrigues(r, R);
  std::cout << "R matrix : " << std::endl << R << std::endl;
  std::cout << "t : " << std::endl << t << std::endl;

  // using g2o
  std::cout << "**************************" << std::endl;
  std::cout << "Using G2O" << std::endl;
  VecVector2d pts2d_eigen;
  VecVector3d pts3d_eigen;
  Sophus::SE3d pose;
  for (int i = 0; i < pts3d.size(); i++) {
    pts3d_eigen.push_back(Eigen::Vector3d(pts3d[i].x, pts3d[i].y, pts3d[i].z));
    pts2d_eigen.push_back(Eigen::Vector2d(pts2d[i].x, pts2d[i].y));
  }
  BAg2o(pts2d_eigen, pts3d_eigen, K, pose);
  std::cout << "Pose using g2o : " << pose.matrix() << std::endl;
}

void find_feature_matches(const cv::Mat &img1, const cv::Mat &img2,
                          std::vector<cv::KeyPoint> &keypoints1,
                          std::vector<cv::KeyPoint> &keypoints2,
                          std::vector<cv::DMatch> &matches) {
  cv::Mat descriptors1, descriptors2;
  std::vector<cv::DMatch> allmatches;
  cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
  cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
  cv::Ptr<cv::DescriptorMatcher> matcher =
      cv::DescriptorMatcher::create("BruteForce-Hamming");
  detector->detect(img1, keypoints1);
  detector->detect(img2, keypoints2);
  descriptor->compute(img1, keypoints1, descriptors1);
  descriptor->compute(img2, keypoints2, descriptors2);
  matcher->match(descriptors1, descriptors2, allmatches);
  auto minmax =
      std::minmax_element(allmatches.begin(), allmatches.end(),
                          [](const cv::DMatch &m1, const cv::DMatch &m2) {
                            return m1.distance < m2.distance;
                          });
  double mindist = minmax.first->distance;
  for (cv::DMatch match : allmatches) {
    if (match.distance <= std::max(2 * mindist, 30.0)) {
      matches.push_back(match);
    }
  }
}

cv::Point2d pixel2cam(const cv::Point2d &p, const cv::Mat &K) {
  return cv::Point2d((p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
                     (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1));
}

class PoseVertex : public g2o::BaseVertex<6, Sophus::SE3d> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  virtual void setToOriginImpl() override { _estimate = Sophus::SE3d(); }
  virtual void oplusImpl(const double *update) override {
    Eigen::Matrix<double, 6, 1> update_eigen;
    update_eigen << update[0], update[1], update[2], update[3], update[4],
        update[5];
    _estimate = Sophus::SE3d::exp(update_eigen) * _estimate;
  }
  virtual bool read(std::istream &in) override {}
  virtual bool write(std::ostream &out) const override {}
};

class EdgeProjection
    : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, PoseVertex> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  EdgeProjection(const Eigen::Vector3d &pos, const Eigen::Matrix3d &K)
      : _pos3d(pos), _K(K) {}
  virtual void computeError() override {
    const PoseVertex *v = static_cast<PoseVertex *>(_vertices[0]);
    Sophus::SE3d T = v->estimate();
    Eigen::Vector3d pixel_pos = _K * (T * _pos3d);
    pixel_pos /= pixel_pos[2];
    _error = _measurement - pixel_pos.head<2>();
  }
  virtual void linearizeOplus() override {
    const PoseVertex *v = static_cast<PoseVertex *>(_vertices[0]);
    Sophus::SE3d T = v->estimate();
    Eigen::Vector3d pos_cam = T * _pos3d;
    double fx = _K(0, 0);
    double fy = _K(1, 1);
    // double cx = _K(0, 2);
    // double cy = _K(1, 2);
    double X = pos_cam[0];
    double Y = pos_cam[1];
    double Z = pos_cam[2];
    double Z2 = Z * Z;
    _jacobianOplusXi << -fx / Z, 0, fx * X / Z2, fx * X * Y / Z2,
        -fx - fx * X * X / Z2, fx * Y / Z, 0, -fy / Z, fy * Y / (Z * Z),
        fy + fy * Y * Y / Z2, -fy * X * Y / Z2, -fy * X / Z;
  }
  virtual bool read(std::istream &in) override {}
  virtual bool write(std::ostream &out) const override {}

private:
  Eigen::Vector3d _pos3d;
  Eigen::Matrix3d _K;
};

void BAg2o(const VecVector2d &points2d, const VecVector3d &points3d,
           const cv::Mat &K, Sophus::SE3d &pose) {
  typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType;
  typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>
      LinearSolverType;
  g2o::OptimizationAlgorithmGaussNewton *solver =
      new g2o::OptimizationAlgorithmGaussNewton(
          g2o::make_unique<BlockSolverType>(
              g2o::make_unique<LinearSolverType>()));
  g2o::SparseOptimizer optimizer;
  optimizer.setAlgorithm(solver);
  PoseVertex *pose_vertex = new PoseVertex();
  pose_vertex->setId(0);
  pose_vertex->setEstimate(Sophus::SE3d());
  optimizer.addVertex(pose_vertex);
  Eigen::Matrix3d K_eigen;
  K_eigen << K.at<double>(0, 0), K.at<double>(0, 1), K.at<double>(0, 2),
      K.at<double>(1, 0), K.at<double>(1, 1), K.at<double>(1, 2),
      K.at<double>(2, 0), K.at<double>(2, 1), K.at<double>(2, 2);
  int index = 1;
  for (size_t i = 0; i < points2d.size(); ++i) {
    auto p2d = points2d[i];
    auto p3d = points3d[i];
    EdgeProjection *edge = new EdgeProjection(p3d, K_eigen);
    edge->setId(index);
    edge->setVertex(0, pose_vertex);
    edge->setMeasurement(p2d);
    edge->setInformation(Eigen::Matrix2d::Identity());
    optimizer.addEdge(edge);
    index++;
  }
  optimizer.setVerbose(false);
  optimizer.initializeOptimization();
  optimizer.optimize(10);
  pose = pose_vertex->estimate();
}