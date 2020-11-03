#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <vector>

void find_feature_matches(const cv::Mat &img1, const cv::Mat &img2,
                          std::vector<cv::KeyPoint> &keypoints1,
                          std::vector<cv::KeyPoint> &keypoints2,
                          std::vector<cv::DMatch> &matches);

void pose_estimation_2d2d(std::vector<cv::KeyPoint> &keypoints1,
                          std::vector<cv::KeyPoint> &keypoints2,
                          std::vector<cv::DMatch> &matches, cv::Mat &K,
                          cv::Mat &R, cv::Mat &t);

cv::Point2d pixel2cam(const cv::Point2d &p, const cv::Mat &K);

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

  cv::Mat t_hat =
      (cv::Mat_<double>(3, 3) << 0, -t.at<double>(2, 0), t.at<double>(1, 0),
       t.at<double>(2, 0), 0, -t.at<double>(0, 0), -t.at<double>(1, 0),
       t.at<double>(0, 0), 0);
  std::cout << "t^R = " << t_hat * R << std::endl;
  for (cv::DMatch match : matches) {
    // std::cout << (keypoints1[match.queryIdx].pt.x - K.at<double>(0, 2)) /
    //                  K.at<double>(0, 0)
    //           << std::endl;
    cv::Point2d pt1 = pixel2cam(keypoints1[match.queryIdx].pt, K);
    cv::Mat y1 = (cv::Mat_<double>(3, 1) << pt1.x, pt1.y, 1);
    cv::Point2d pt2 = pixel2cam(keypoints2[match.trainIdx].pt, K);
    cv::Mat y2 = (cv::Mat_<double>(3, 1) << pt2.x, pt2.y, 1);
    cv::Mat d = y2.t() * t_hat * R * y1;
    std::cout << "epipolar constraint = " << d << std::endl;
  }
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

  cv::Mat imgmatch;
  cv::drawMatches(img1, keypoints1, img2, keypoints2, matches, imgmatch);
  cv::imshow("Matches", imgmatch);
  cv::waitKey(0);
}

void pose_estimation_2d2d(std::vector<cv::KeyPoint> &keypoints1,
                          std::vector<cv::KeyPoint> &keypoints2,
                          std::vector<cv::DMatch> &matches, cv::Mat &K,
                          cv::Mat &R, cv::Mat &t) {
  std::vector<cv::Point2f> points1;
  std::vector<cv::Point2f> points2;
  for (auto match : matches) {
    // std::cout << keypoints1[match.queryIdx].pt <<
    // keypoints2[match.trainIdx].pt
    //           << match.distance << std::endl;
    points1.push_back(keypoints1[match.queryIdx].pt);
    points2.push_back(keypoints2[match.trainIdx].pt);
  }

  cv::Point2d principal_point(K.at<double>(0, 2), K.at<double>(1, 2));
  // std::cout << principal_point << std::endl;
  double focal_length = K.at<double>(1, 1);
  cv::Mat essential_matrix =
      cv::findEssentialMat(points1, points2, focal_length, principal_point);
  // std::cout << essential_matrix << std::endl;
  cv::recoverPose(essential_matrix, points1, points2, K, R, t);
  // std::cout << R << std::endl;
  // std::cout << t << std::endl;
}

cv::Point2d pixel2cam(const cv::Point2d &p, const cv::Mat &K) {
  return cv::Point2d((p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
                     (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1));
}