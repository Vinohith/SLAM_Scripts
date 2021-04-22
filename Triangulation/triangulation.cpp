#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <vector>

/*
Step 1 : Extract features from image1 and Image2 and then match them (matches are from Image1(queryIdx) to Image2(trainIdx)).
Step 2 : Estimate the 2D-2D pose between the two image frames (this produces the transformation of points in frame of Image1 
															   to frame of Image2), which is the relative pose T_c2_c1.
Step 3 : Triangulate the points using the matches (the output points will be in the in the reference frame of Image1)

Example : If the frame of Image1 is the world frame, then performing 2D-2D pose estimation results in Tcw = [R|t] for Image2
		  (i.e. the transformation of points in world frame to the frame of Image2 (current frame)). Because the triangulation
		  is happening between points in Image1 and Image2 using Image1 frame as the reference frame (world frame) and Image2 
		  frame (Tcw => can be considered as the pose of camera 2 in the world frame) the output triangulated points will be 
		  in the frame of Image2 (i.e. world frame). p_c = K * Tcw * P_w

		  If frame of Image1 is not the world frame, but some T_c1_w, then 2D-2D Pose estimation produces T_c2_c1, then,
		  T_c2_w = T_c2_c1 * T_c1_w (the matrices should be converted to 4x4 matrices).

In conclusion : 2D-2D Pose estimation produces the transformation of points in frame of Image1 to frame of Image2 (i.e. T_c2_c1
				transformation of points in camera 1 to camera 2. This can be considered as the pose of camera 2 in the frame
				of camera 1). p_c2 = K * T_c2_c1 * P_c1

				Triangulating the matches in Image1 and Image2 using the poses of camera 1 (in some reference frame) and camera 2 
				(in some ereference frame) results in producing the triangulated points in the reference frame (can be considered as
				the world frame). So, if pose of camera 1 = T_c1_w and pose of camera 2 = T_c2_w, then the triangulated points will
				be P_w, such that, p_c1 = K * T_c1_w * P_w and p_c2 = K * T_c2_w * P_w
*/

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
                        std::vector<cv::Point3f> &points);

cv::Point2f pixel2cam(const cv::Point2f &p, const cv::Mat &K);



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
      (cv::Mat_<float>(3, 3) << 707.0912, 0, 601.8873, 0, 707.0912, 183.1104, 0, 0, 1);
  pose_estimation_2d2d(keypoints1, keypoints2, matches, K, R, t);
  std::cout << "R = " << R << std::endl;
  std::cout << "t = " << t << std::endl;

  std::vector<cv::Point3f> points;
  triangulate_points(keypoints1, keypoints2, matches, K, R, t, points);

  // std::cout << points.size() << std::endl;

  for (int i=0; i < points.size(); i++){
  	cv::DMatch match = matches[i];
  	cv::Point2f pts1 = keypoints1[match.queryIdx].pt; 
  	cv::Point2f pts2 = keypoints2[match.trainIdx].pt;
  	cv::Mat P = (cv::Mat_<float>(4, 1) << points[i].x, points[i].y, points[i].z , (float) 1);
  	cv::Mat T1 = (cv::Mat_<float>(3, 4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);
  	cv::Mat T2 = (cv::Mat_<float>(3, 4) << R.at<double>(0, 0), R.at<double>(0, 1),
                R.at<double>(0, 2), t.at<double>(0, 0), R.at<double>(1, 0),
                R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
                R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2),
                t.at<double>(2, 0));
  	// cv::Mat T2 = (cv::Mat_<float>(3, 4) << 9.999910e-01, 1.048972e-03, -4.131348e-03, -9.374345e-02,
  	// 									  -1.058514e-03, 9.999968e-01, -2.308104e-03, -5.676064e-02,
  	// 									   4.128913e-03, 2.312456e-03, 9.999887e-01, 1.716275e+00);
  	std::cout << points[i] << std::endl;
  	// Pose 1
  	cv::Mat p1 = K * T1 * P;
  	p1 /= p1.at<float>(2, 0);
  	cv::Point2f pt_cam1(p1.at<float>(0, 0), p1.at<float>(1, 0));
  	std::cout << pts1 << "  " << pt_cam1 << std::endl;
  	// Pose 2
  	cv::Mat p2 = K * T2 * P;
  	p2 /= p2.at<float>(2, 0);
  	cv::Point2f pt_cam2(p2.at<float>(0, 0), p2.at<float>(1, 0));
  	std::cout << pts2 << "  " << pt_cam2 << std::endl;

  	std::cout << std::endl;
  }

  return 0;
}


void find_feature_matches(const cv::Mat &img1, const cv::Mat &img2,
                          std::vector<cv::KeyPoint> &keypoints1,
                          std::vector<cv::KeyPoint> &keypoints2,
                          std::vector<cv::DMatch> &matches) {

  cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create(2000);
  cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
  cv::Ptr<cv::DescriptorMatcher> matcher =
      cv::DescriptorMatcher::create("BruteForce-Hamming");
  // cv::Ptr<cv::FeatureDetector> detector =
  //     cv::GFTTDetector::create(1000, 0.001, 15.0);
  // cv::Ptr<cv::DescriptorExtractor> descriptor =
  //     cv::xfeatures2d::BriefDescriptorExtractor::create();
  // cv::Ptr<cv::DescriptorMatcher> matcher =
  //       cv::BFMatcher::create(cv::NORM_HAMMING);

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
  float mindist = minmax.first->distance;
  for (cv::DMatch match : allmatches) {
    if (match.distance <= std::max(2 * mindist, (float) 30.0)) {
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
    points1.push_back(keypoints1[match.queryIdx].pt);
    points2.push_back(keypoints2[match.trainIdx].pt);
  }

  cv::Point2f principal_point(K.at<float>(0, 2), K.at<float>(1, 2));
  float focal_length = K.at<float>(1, 1);
  cv::Mat essential_matrix =
      cv::findEssentialMat(points1, points2, focal_length, principal_point);
  cv::recoverPose(essential_matrix, points1, points2, K, R, t);
}

void triangulate_points(std::vector<cv::KeyPoint> &keypoints1,
                        std::vector<cv::KeyPoint> &keypoints2,
                        std::vector<cv::DMatch> &matches, cv::Mat &K,
                        cv::Mat &R, cv::Mat &t,
                        std::vector<cv::Point3f> &points) {
	cv::Mat T1 = (cv::Mat_<float>(3, 4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);
  	cv::Mat T2 = (cv::Mat_<float>(3, 4) << R.at<double>(0, 0), R.at<double>(0, 1),
                R.at<double>(0, 2), t.at<double>(0, 0), R.at<double>(1, 0),
                R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
                R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2),
                t.at<double>(2, 0));
  	// cv::Mat T2 = (cv::Mat_<float>(3, 4) << 9.999910e-01, 1.048972e-03, -4.131348e-03, -9.374345e-02,
  	// 									  -1.058514e-03, 9.999968e-01, -2.308104e-03, -5.676064e-02,
  	// 									   4.128913e-03, 2.312456e-03, 9.999887e-01, 1.716275e+00);
  std::vector<cv::Point2f> pts1, pts2;
  for (cv::DMatch match : matches) {
  	// std::cout << match.queryIdx << "  " << match.trainIdx << std::endl;
    pts1.push_back(pixel2cam(keypoints1[match.queryIdx].pt, K));
    pts2.push_back(pixel2cam(keypoints2[match.trainIdx].pt, K));
  }
  cv::Mat pts4d;
  cv::triangulatePoints(T1, T2, pts1, pts2, pts4d);
  std::cout << pts4d.size() << std::endl;
  std::cout << pts4d.cols << std::endl;
  for (int i = 0; i < pts4d.cols; i++) {
    // std::cout << "4D point : " << pts4d.col(i) << std::endl;
    cv::Mat x = pts4d.col(i);
    x /= x.at<float>(3, 0);
    // std::cout << "3D point : " << x.at<float>(0, 0) << ", " << x.at<float>(1, 0)
    //           << ", " << x.at<float>(2, 0) << std::endl;
    cv::Point3f p(x.at<float>(0, 0), x.at<float>(1, 0), x.at<float>(2, 0));
    points.push_back(p);
  }
}


cv::Point2f pixel2cam(const cv::Point2f &p, const cv::Mat &K) {
  return cv::Point2f((p.x - K.at<float>(0, 2)) / K.at<float>(0, 0),
                     (p.y - K.at<float>(1, 2)) / K.at<float>(1, 1));
}
