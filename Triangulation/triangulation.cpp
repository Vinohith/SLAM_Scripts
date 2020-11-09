#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

void find_feature_matches(const cv::Mat &img1, const cv::Mat &img2,
                          std::vector<cv::KeyPoint> &keypoints1,
                          std::vector<cv::KeyPoint> &keypoints2,
                          std::vector<cv::DMatch> &matches);

void pose_estimation_2d2d(std::vector<cv::KeyPoint> &keypoints1,
                          std::vector<cv::KeyPoint> &keypoints2,
                          std::vector<cv::DMatch> &matches, cv::Mat &K,
                          cv::Mat &R, cv::Mat &t);

void triangulate_points(std::vector<cv::KeyPoint> &keypoints1,
                        std::vector<cv::KeyPoint> &keypoints2,
                        std::vector<cv::DMatch> &matches, cv::Mat &K,
                        cv::Mat &R, cv::Mat &t,
                        std::vector<cv::Point3d> &points);

cv::Point2d pixel2cam(const cv::Point2d &p, const cv::Mat &K);

inline cv::Scalar get_color(float depth) {
  float up_th = 50, low_th = 10, th_range = up_th - low_th;
  if (depth > up_th) {
    depth = up_th;
  }
  if (depth < low_th) {
    depth = low_th;
  }
  return cv::Scalar(255 * depth / th_range, 0, 255 * (1 - depth / th_range));
}

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cout << "Pass img1 and img2." << std::endl;
    return 1;
  }
  cv::Mat img1 = cv::imread(argv[1]);
  cv::Mat img2 = cv::imread(argv[2]);

  std::vector<cv::KeyPoint> keypoints1, keypoints2;
  std::vector<cv::DMatch> matches;
  find_feature_matches(img1, img2, keypoints1, keypoints2, matches);

  std::cout << "Number of Matches : " << matches.size() << std::endl;

  cv::Mat R, t;
  cv::Mat K =
      (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
  pose_estimation_2d2d(keypoints1, keypoints2, matches, K, R, t);
  std::cout << "R = " << R << std::endl;
  std::cout << "t = " << t << std::endl;

  std::vector<cv::Point3d> points;
  triangulate_points(keypoints1, keypoints2, matches, K, R, t, points);

  // creates independent copy
  cv::Mat img1_plot = img1.clone();
  cv::Mat img2_plot = img2.clone();
  for (int i = 0; i < matches.size(); i++) {
    float depth1 = points[i].z;
    cv::Point2d pt_cam1 = pixel2cam(keypoints1[matches[i].queryIdx].pt, K);
    cv::circle(img1_plot, keypoints1[matches[i].queryIdx].pt, 2,
               get_color(depth1), 2);
    // p_t = R*p + t
    cv::Mat pt_trans =
        R * (cv::Mat_<double>(3, 1) << points[i].x, points[i].y, points[i].z) +
        t;
    float depth2 = pt_trans.at<float>(2, 0);
    cv::circle(img2_plot, keypoints2[matches[i].trainIdx].pt, 2,
               get_color(depth2), 2);
  }

  cv::imshow("img 1", img1_plot);
  cv::imshow("img 2", img2_plot);
  cv::waitKey(0);

  return 0;
}

void find_feature_matches(const cv::Mat &img1, const cv::Mat &img2,
                          std::vector<cv::KeyPoint> &keypoints1,
                          std::vector<cv::KeyPoint> &keypoints2,
                          std::vector<cv::DMatch> &matches) {

  cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
  cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
  cv::Ptr<cv::DescriptorMatcher> matcher =
      cv::DescriptorMatcher::create("BruteForce-Hamming");

  detector->detect(img1, keypoints1);
  detector->detect(img2, keypoints2);

  cv::Mat descriptors1, descriptors2;
  descriptor->compute(img1, keypoints1, descriptors1);
  descriptor->compute(img2, keypoints2, descriptors2);

  std::vector<cv::DMatch> allmatches;
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

  //   cv::Mat imgmatch;
  //   cv::drawMatches(img1, keypoints1, img2, keypoints2, matches, imgmatch);
  //   cv::imshow("Matches", imgmatch);
  //   cv::waitKey(0);
}

void pose_estimation_2d2d(std::vector<cv::KeyPoint> &keypoints1,
                          std::vector<cv::KeyPoint> &keypoints2,
                          std::vector<cv::DMatch> &matches, cv::Mat &K,
                          cv::Mat &R, cv::Mat &t) {
  std::vector<cv::Point2f> points1;
  std::vector<cv::Point2f> points2;
  for (auto match : matches) {
    points1.push_back(keypoints1[match.queryIdx].pt);
    points2.push_back(keypoints2[match.trainIdx].pt);
  }

  cv::Point2d principal_point(K.at<double>(0, 2), K.at<double>(1, 2));
  double focal_length = K.at<double>(1, 1);
  cv::Mat essential_matrix =
      cv::findEssentialMat(points1, points2, focal_length, principal_point);
  cv::recoverPose(essential_matrix, points1, points2, K, R, t);
}

void triangulate_points(std::vector<cv::KeyPoint> &keypoints1,
                        std::vector<cv::KeyPoint> &keypoints2,
                        std::vector<cv::DMatch> &matches, cv::Mat &K,
                        cv::Mat &R, cv::Mat &t,
                        std::vector<cv::Point3d> &points) {
  cv::Mat T1 = (cv::Mat_<float>(3, 4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);
  cv::Mat T2 = (cv::Mat_<float>(3, 4) << R.at<double>(0, 0), R.at<double>(0, 1),
                R.at<double>(0, 2), t.at<double>(0, 0), R.at<double>(1, 0),
                R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
                R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2),
                t.at<double>(2, 0));
  std::vector<cv::Point2f> pts1, pts2;
  for (cv::DMatch match : matches) {
    pts1.push_back(pixel2cam(keypoints1[match.queryIdx].pt, K));
    pts2.push_back(pixel2cam(keypoints2[match.trainIdx].pt, K));
  }
  cv::Mat pts4d;
  cv::triangulatePoints(T1, T2, pts1, pts2, pts4d);
  std::cout << pts4d.size() << std::endl;
  std::cout << pts4d.cols << std::endl;
  for (int i = 0; i < pts4d.cols; i++) {
    std::cout << "4D point : " << pts4d.col(i) << std::endl;
    cv::Mat x = pts4d.col(i);
    x /= x.at<float>(3, 0);
    std::cout << "3D point : " << x.at<float>(0, 0) << ", " << x.at<float>(1, 0)
              << ", " << x.at<float>(2, 0) << std::endl;
    cv::Point3d p(x.at<float>(0, 0), x.at<float>(1, 0), x.at<float>(2, 0));
    points.push_back(p);
  }
}

cv::Point2d pixel2cam(const cv::Point2d &p, const cv::Mat &K) {
  return cv::Point2d((p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
                     (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1));
}