#include <iostream>
#include <yolo_v2_class.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <fstream>
#include <string>

const int NUM_LABELS = 80;

std::vector<std::string> read_names(const std::string& filename)
{
    std::ifstream in; in.open(filename);
    std::vector<std::string> names(NUM_LABELS);
    for(int i = 0; i < NUM_LABELS; i++)
        std::getline(in, names[i]);
    return names;
}

int main()
{
    auto names = read_names("/home/rohan/Downloads/darknet/data/coco.names");

    Detector obj("/home/rohan/Downloads/darknet/cfg/yolov3.cfg", "/home/rohan/Downloads/darknet/yolov3.weights");
    cv::VideoCapture cap("/home/rohan/Downloads/Compressed/darknet/arapaho/ultadanga.mp4");
    cv::Mat frame;
    cv::namedWindow("darknet", CV_WINDOW_OPENGL);
    while(true)
    {
        cap >> frame;
        std::vector<bbox_t> boxes = obj.detect(frame, 0.3);
        for(int i = 0; i < boxes.size(); i++)
        {
            std::cout << "Object " << (i + 1) << '/' << boxes.size() << '\n';
            std::cout << "obj-id = " << boxes[i].obj_id << '\n';
            std::cout << "track-id = " << boxes[i].track_id << '\n';
            std::cout << names[boxes[i].obj_id] << '\n';
            std::cout << boxes[i].x << ' ' << boxes[i].y << ' ' << boxes[i].w << ' ' << boxes[i].h << '\n';
            int leftTopX = boxes[i].x;
            int leftTopY = boxes[i].y;
            int rightBotX = boxes[i].x + boxes[i].w;
            int rightBotY = boxes[i].y + boxes[i].h;
            cv::rectangle(frame, cv::Point(leftTopX, leftTopY), cv::Point(rightBotX, rightBotY), CV_RGB(255, 0, 0), 1, 8, 0);
            cv::putText(frame, names[boxes[i].obj_id], cv::Point(leftTopX, leftTopY),
                cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(255, 0, 200), 1, CV_AA);
        }
        cv::imshow("darknet", frame);
        std::cout << "\x1B[2J\x1B[H";
        cv::waitKey(1);
    }
}
