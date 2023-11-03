#include <cmath>
#include <math.h>
#include <string>
#include <iostream>
#include <mutex>
#include <sys/select.h>
#include <termios.h>
#include <vector>
#include <chrono>
#include <unistd.h>

#include <opencv2/opencv.hpp>
#include <apriltag/apriltag.h>
#include <apriltag/apriltag_pose.h>
#include <Eigen/Dense>
#include <apriltag/tag16h5.h>

int timeMS(){
	using namespace std::chrono;
	return duration_cast< milliseconds >(system_clock::now().time_since_epoch()).count();
}

int main(){
	bool doApril = false;
	apriltag_family_t* tf = tag16h5_create();
	apriltag_detector_t* td = apriltag_detector_create();
	apriltag_detector_add_family(td, tf);
	td->quad_decimate = 2.0;
	td->quad_sigma = 0.0;
	td->nthreads = 4;
	apriltag_detector_add_family_bits(td, tf, 1);

	cv::Mat video;
	cv::VideoCapture cap(0);
	cv::Mat gray;

	// the tags irl size in inches
	const double tagPhysicalSize = 6.0;

	// TODO update focalLength to be accurate to cam
	const double camFocalLength = 525.0;
	// TODO update values to be accurate to cam
	const double camCX = 0.0;
	const double camCY = 0.0;

	namedWindow("video", cv::WINDOW_AUTOSIZE);	

 	while(true){
		cap >> video;

		if(doApril){
			cv::cvtColor(video, gray, cv::COLOR_BGR2GRAY);

			image_u8_t im = {
				.width = video.cols,
				.height = video.rows,
				.stride = video.cols,
				.buf = gray.data
			};
			zarray_t* detections = apriltag_detector_detect(td, &im);

			for(int i = 0; i < zarray_size(detections); i++){
				apriltag_detection_t* det;
				zarray_get(detections, i, &det);

				if(det->hamming == 0){
					int tagId = det->id;

					double R[3][3];
					for (int i = 0; i < 3; i++) {
			    	for (int j = 0; j < 3; j++) {
        			R[i][j] = det->H->data[i*3 + j];
    				}
					}

					double yawRad = atan2(R[1][0], R[0][0]);
					// degrees is type-casted for text spacing reasons, for accuracy use radians or a double for degrees
					double yawDeg = yawRad * (180/M_PI);
//					double pitch = atan2(-R[2][0], sqrt(R[2][1] * R[2][1] + R[2][2] * R[2][2]));
//					double roll = atan2(R[2][1], R[2][2]);

					apriltag_detection_info_t info;
					info.det = det;
					info.tagsize = tagPhysicalSize;
					info.fx = camFocalLength;
					info.fy = camFocalLength;
					info.cx = camCX;
					info.cy = camCY;

					apriltag_pose_t pose;
					estimate_pose_for_tag_homography(&info, &pose);

					Eigen::Vector3d t;
					for(int i = 0; i < 3; i++){
						t(i) = pose.t->data[i];
					}

					// distance to the tag in inches
					double dist = t.norm();

					cv::circle(video, cv::Point(det->c[0], det->c[1]), 10, cv::Scalar(255, 0, 0), cv::FILLED, cv::LINE_8);
					cv::rectangle(video, cv::Point(det->p[0][0], det->p[0][1]), cv::Point(det->p[2][0], det->p[2][1]), cv::Scalar(0, 255, 255));

					cv::putText(video, ("Tag " + std::to_string(tagId) + ": " + std::to_string(dist) + " inches away").c_str(), cv::Point(det->c[0], det->c[1]), cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(255, 255, 255), 2);
				}
			}
		}
		
		cv::imshow("video", video);
		
		int key = cv::waitKey(5);

		if(key == 113){
			cv::destroyWindow("video");
			break;
		}
		else if(key == 97){
			doApril = !doApril;
			std::cout << "Turned " << (doApril ? "on " : "off ") << "april tag detection" << std::endl;
		}
	}

	apriltag_detector_destroy(td);
	tag16h5_destroy(tf);
	std::cout << "Destroyed apriltag detections" << std::endl;

	cv::destroyAllWindows();
 	return 0;
}

