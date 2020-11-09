#include <DBoW3/DBoW3.h>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  DBoW3::Vocabulary vocab("../vocabulary.yml.gz");

  int num_images = std::stoi(argv[1]);
  std::vector<cv::Mat> images;
  for (int i = 0; i < num_images; i++) {
    std::string image_path = "../data/" + std::to_string(i + 1) + ".png";
    // std::cout << image_path << std::endl;
    images.push_back(cv::imread(image_path));
  }

  cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
  cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
  std::vector<cv::Mat> descriptors;
  for (cv::Mat &image : images) {
    std::vector<cv::KeyPoint> keypoint;
    cv::Mat descr;
    detector->detect(image, keypoint);
    descriptor->compute(image, keypoint, descr);
    descriptors.push_back(descr);
  }

  std::cout << "Comparing Images with other images" << std::endl;
  for (int i = 0; i < images.size(); i++) {
    DBoW3::BowVector v1;
    vocab.transform(descriptors[i], v1);
    for (int j = i; j < images.size(); j++) {
      DBoW3::BowVector v2;
      vocab.transform(descriptors[j], v2);
      double score = vocab.score(v1, v2);
      std::cout << "image " << i << " vs image " << j << " : " << score
                << std::endl;
    }
    std::cout << "*******************" << std::endl;
  }
  std::cout << std::endl;

  std::cout << "Comparing Images with database" << std::endl;
  DBoW3::Database db(vocab, false, 0);
  for (int i = 0; i < descriptors.size(); i++) {
    db.add(descriptors[i]);
  }
  std::cout << "database info: " << db << std::endl;
  for (int i = 0; i < descriptors.size(); i++) {
    DBoW3::QueryResults ret;
    db.query(descriptors[i], ret, 3);
    std::cout << "searching for image " << i << " returns " << ret << std::endl;
    std::cout << "*******************" << std::endl;
  }

  return 0;
}