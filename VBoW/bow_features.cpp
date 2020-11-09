#include <DBoW3/DBoW3.h>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <string>
#include <vector>

int main(int argc, char **argv) {
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

  std::cout << "Creating Vocabulary" << std::endl;
  DBoW3::Vocabulary vocab;
  vocab.create(descriptors);
  std::cout << "vocabulary info: " << vocab << std::endl;
  vocab.save("../vocabulary.yml.gz");

  return 0;
}