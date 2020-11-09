#include <DBoW3/DBoW3.h>
#include <iomanip>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>

std::string to_format(const int number) {
  std::stringstream ss;
  ss << std::setw(6) << std::setfill('0') << number;
  return ss.str();
}

inline bool file_exists(const std::string &name) {
  struct stat buffer;
  return (stat(name.c_str(), &buffer) == 0);
}

int main(int argc, char **argv) {
  int num_images = std::stoi(argv[1]);
  std::vector<cv::Mat> images;
  for (int i = 0; i < num_images; i++) {
    std::string image_path = "../data2/" + to_format(i + 1) + ".png";
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

  DBoW3::Vocabulary vocab;
  std::string vocabname = "../vocabulary_kitti.yml.gz";
  std::cout << file_exists(vocabname) << std::endl;

  if (file_exists(vocabname)) {
    vocab.load("../vocabulary_kitti.yml.gz");
  } else {
    std::cout << "Creating Vocabulary" << std::endl;
    vocab.create(descriptors);
    vocab.save("../vocabulary_kitti.yml.gz");
    std::cout << "Loading from if statement" << std::endl;
    vocab.load("../vocabulary_kitti.yml.gz");
  }

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