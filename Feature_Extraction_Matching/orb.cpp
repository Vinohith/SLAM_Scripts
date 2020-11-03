#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <vector>

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cout << "Pass img1 and img2" << std::endl;
    return 1;
  }

  // read image
  cv::Mat img1 = cv::imread(argv[1]);
  cv::Mat img2 = cv::imread(argv[2]);
  assert(img1.data != nullptr && img2.data != nullptr);

  std::vector<cv::KeyPoint> keypoints1;
  std::vector<cv::KeyPoint> keypoints2;
  cv::Mat descriptor1;
  cv::Mat descriptor2;
  cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
  cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
  // auto takes type cv::Ptr<cv::DescriptorMatcher>
  auto matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

  // Detect corners
  detector->detect(img1, keypoints1);
  detector->detect(img2, keypoints2);

  // Find descriptors
  descriptor->compute(img1, keypoints1, descriptor1);
  descriptor->compute(img2, keypoints2, descriptor2);

  // Draw detected keypoints
  cv::Mat imgout;
  cv::drawKeypoints(img1, keypoints1, imgout);
  cv::imshow("Features", imgout);
  cv::waitKey(0);

  // Performing matching
  std::vector<cv::DMatch> matches;
  matcher->match(descriptor1, descriptor2, matches);
  // auto takes type std::pair<std::vector<cv::DMatch>::iterator,
  // std::vector<cv::DMatch>::iterator>
  auto minmax =
      std::minmax_element(matches.begin(), matches.end(),
                          [](const cv::DMatch &m1, const cv::DMatch &m2) {
                            return m1.distance < m2.distance;
                          });
  double mindist = minmax.first->distance;
  double maxdist = minmax.second->distance;

  // Finding good matches. When the distance between the descriptors is greater
  // than twice the minimum distance, it is considered that the matching is
  // wrong. But sometimes the minimum distance will be very small, set an
  // empirical value of 30 as the lower limit.
  std::vector<cv::DMatch> goodmatches;
  for (auto match : matches) {
    if (match.distance <= std::max(2 * mindist, 30.0)) {
      goodmatches.push_back(match);
    }
  }

  cv::Mat imgmatch;
  cv::Mat imggoodmatch;
  cv::drawMatches(img1, keypoints1, img2, keypoints2, matches, imgmatch);
  cv::drawMatches(img1, keypoints1, img2, keypoints2, goodmatches,
                  imggoodmatch);
  cv::imshow("matches", imgmatch);
  cv::imshow("good matches", imggoodmatch);
  cv::waitKey();

  return 0;
}