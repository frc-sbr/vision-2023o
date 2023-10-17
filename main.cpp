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
#include <libfreenect/libfreenect.hpp>
#include <apriltag/apriltag.h>
#include <apriltag/tag16h5.h>

int timeMS(){
	using namespace std::chrono;
	return duration_cast< milliseconds >(system_clock::now().time_since_epoch()).count();
}

// Code based from https://github.com/zfields/kinect-opencv-face-detect/blob/master/kinect_opencv_face_detect.cpp
class MicrosoftKinect : public Freenect::FreenectDevice
{
public:
  MicrosoftKinect(
      freenect_context *_ctx,
      int _index) : Freenect::FreenectDevice(_ctx, _index),
                    _rgb_frame_available(false),
                    _depth_frame_available(false)
  {
    setVideoResolution(FREENECT_RESOLUTION_MEDIUM);

    // Load the gamma array with color values to represent 11-bit
    // (2^11 or 0 - 2047) depth data capture by the Microsoft Kinect
    // (enables later heat map visualization)
    for (unsigned int i = 0; i < 2048; ++i)
    {
      float v = i / 2048.0f;
      v = std::pow(v, 3) * 6;
      _gamma[i] = v * 6 * 256;
    }
    setLed(LED_GREEN);
    setTiltDegrees(0);
  }

  virtual ~MicrosoftKinect() override
  {
    setTiltDegrees(0);
    setLed(LED_OFF);
  }

  bool getBGRVideo(cv::Mat &bgr_image)
  {
    std::lock_guard<std::mutex> rgb_lock(_rgb_mutex);
    if (_rgb_frame_available)
    {
      cv::cvtColor(_live_rgb_feed, bgr_image, cv::COLOR_RGB2BGR);
      _rgb_frame_available = false;
      return true;
    }
    else
    {
      return false;
    }
  }

  bool getDepthHeatMap(cv::Mat &heat_map)
  {
    std::lock_guard<std::mutex> depth_lock(_depth_mutex);

    static const size_t B(0), G(1), R(2);
    if (_depth_frame_available)
    {
      // Loop through depth array data
      for (int r = 0; r < _live_depth_feed.rows; ++r)
      {
        for (int c = 0; c < _live_depth_feed.cols; ++c)
        {
          auto depth_value = _live_depth_feed.at<uint16_t>(r, c);

          // Map the depth value to _gamma values
          uint16_t heat_value = _gamma[depth_value];
          uint8_t fine_heat = static_cast<uint8_t>(heat_value & 0xFF);
          uint8_t coarse_heat = static_cast<uint8_t>(heat_value >> 8);

          // Examine the pval with the low byte removed
          switch (coarse_heat)
          {
          // white fading to red
          case 0:
            heat_map.at<cv::Vec3b>(r, c)[B] = (255 - fine_heat);
            heat_map.at<cv::Vec3b>(r, c)[G] = (255 - fine_heat);
            heat_map.at<cv::Vec3b>(r, c)[R] = 255;
            break;
          // red fading to yellow
          case 1:
            heat_map.at<cv::Vec3b>(r, c)[B] = 0;
            heat_map.at<cv::Vec3b>(r, c)[G] = fine_heat;
            heat_map.at<cv::Vec3b>(r, c)[R] = 255;
            break;
          // yellow fading to green
          case 2:
            heat_map.at<cv::Vec3b>(r, c)[B] = 0;
            heat_map.at<cv::Vec3b>(r, c)[G] = 255;
            heat_map.at<cv::Vec3b>(r, c)[R] = (255 - fine_heat);
            break;
          // green fading to cyan
          case 3:
            heat_map.at<cv::Vec3b>(r, c)[B] = fine_heat;
            heat_map.at<cv::Vec3b>(r, c)[G] = 255;
            heat_map.at<cv::Vec3b>(r, c)[R] = 0;
            break;
          // cyan fading to blue
          case 4:
            heat_map.at<cv::Vec3b>(r, c)[B] = 255;
            heat_map.at<cv::Vec3b>(r, c)[G] = (255 - fine_heat);
            heat_map.at<cv::Vec3b>(r, c)[R] = 0;
            break;
          // blue fading to magenta
          case 5:
            heat_map.at<cv::Vec3b>(r, c)[B] = 255;
            heat_map.at<cv::Vec3b>(r, c)[G] = 0;
            heat_map.at<cv::Vec3b>(r, c)[R] = fine_heat;
            break;
          // magenta fading to black
          case 6:
            heat_map.at<cv::Vec3b>(r, c)[B] = (255 - fine_heat);
            heat_map.at<cv::Vec3b>(r, c)[G] = 0;
            heat_map.at<cv::Vec3b>(r, c)[R] = (255 - fine_heat);
            break;
          // uncategorized values are rendered gray
          default:
            heat_map.at<cv::Vec3b>(r, c)[B] = 0;
            heat_map.at<cv::Vec3b>(r, c)[G] = 0;
            heat_map.at<cv::Vec3b>(r, c)[R] = 0;
            break;
          }
        }
      }
      _depth_frame_available = false;
      return true;
    }
    else
    {
      return false;
    }
  }

  int getWindowColumnAndRowCount(int &_cols, int &_rows)
  {
    // Check resolution and create image canvas
    return videoResolutionToColumnsAndRows(getVideoResolution(), _cols, _rows);
  }

private:
  uint16_t _gamma[2048];
  std::mutex _rgb_mutex;
  std::mutex _depth_mutex;
  bool _rgb_frame_available;
  bool _depth_frame_available;
  cv::Mat _live_depth_feed;
  cv::Mat _live_rgb_feed;

  int setVideoResolution(freenect_resolution _resolution)
  {
    int result;
    int cols, rows;

    setVideoFormat(FREENECT_VIDEO_RGB, _resolution);
    setDepthFormat(FREENECT_DEPTH_11BIT, _resolution);
    if ((result = videoResolutionToColumnsAndRows(_resolution, cols, rows)))
    {
      // forward error and exit
    }
    else
    {
      _live_depth_feed = cv::Mat(cv::Size(cols, rows), CV_16UC1);
      _live_rgb_feed = cv::Mat(cv::Size(cols, rows), CV_8UC3, cv::Scalar(0));
    }

    return result;
  }

  int videoResolutionToColumnsAndRows(
      freenect_resolution _resolution,
      int &_cols,
      int &_rows)
  {
    int result;

    switch (_resolution)
    {
    case FREENECT_RESOLUTION_LOW:
      _cols = 320;
      _rows = 240;
      result = 0;
      break;
    case FREENECT_RESOLUTION_MEDIUM:
      _cols = 640;
      _rows = 480;
      result = 0;
      break;
    case FREENECT_RESOLUTION_HIGH:
      _cols = 1280;
      _rows = 1024;
      result = 0;
      break;
    default:
      std::cerr << "Unrecognized Resolution ( " << _resolution << ")" << std::endl;
      result = -1;
    }

    return result;
  }

  // Do not call directly (even in child)
  virtual void VideoCallback(
      void *_rgb,
      uint32_t timestamp) override
  {
    (void)timestamp;
    std::lock_guard<std::mutex> rgb_lock(_rgb_mutex);

    // Load data into CV compatible `Mat` (matrix) object
    _live_rgb_feed.data = static_cast<uint8_t *>(_rgb);
    _rgb_frame_available = true;
  };

  // Do not call directly (even in child)
  virtual void DepthCallback(
      void *_depth,
      uint32_t timestamp) override
  {
    (void)timestamp;
    std::lock_guard<std::mutex> depth_lock(_depth_mutex);

    // Load data into CV compatible `Mat` (matrix) object
    _live_depth_feed.data = static_cast<uint8_t *>(_depth);
    _depth_frame_available = true;
  }
};

int main(){
	int startTime = timeMS();

	bool doApril = false;
	apriltag_family_t* tf = tag16h5_create();
	apriltag_detector_t* td = apriltag_detector_create();
	apriltag_detector_add_family(td, tf);
	td->quad_decimate = 2.0;
	td->quad_sigma = 0.0;
	td->nthreads = 4;
	apriltag_detector_add_family_bits(td, tf, 1);

	int winCols,winRows;

	double tiltDegrees = 0.0;

	Freenect::Freenect freenect;
	MicrosoftKinect &kinect = freenect.createDevice<MicrosoftKinect>(0);

	if(kinect.getWindowColumnAndRowCount(winCols,winRows)){
		return 1;
	}

	cv::Mat video(cv::Size(winCols,winRows), CV_8UC3, cv::Scalar(0));
	cv::Mat gray;
	cv::Mat depthHeatMap(cv::Size(winCols,winRows), CV_8UC3);

//	namedWindow("depth_heat_map", cv::WINDOW_AUTOSIZE);	
	namedWindow("video", cv::WINDOW_AUTOSIZE);	

//	kinect.startDepth();
	kinect.startVideo();

 	while(true){
//		kinect.getDepthHeatMap(depthHeatMap);
//		cv::imshow("depth_heat_map", depthHeatMap);

		kinect.getBGRVideo(video);
		
		if(doApril){
			cv::cvtColor(video, gray, cv::COLOR_BGR2GRAY);

			image_u8_t im = {
				.width = winCols,
				.height = winRows,
				.stride = winCols,
				.buf = gray.data
			};
			zarray_t* detections = apriltag_detector_detect(td, &im);

			for(int i = 0; i < zarray_size(detections); i++){
				apriltag_detection_t* det;
				zarray_get(detections, i, &det);

				if(det->hamming == 0){
					double R[3][3];
					for (int i = 0; i < 3; i++) {
			    	for (int j = 0; j < 3; j++) {
        			R[i][j] = det->H->data[i*3 + j];
    				}
					}

					double yaw = atan2(R[1][0], R[0][0]);
					// degrees is type-casted for text spacing reasons, for accuracy use radians or a double for degrees
					int yawDeg = yaw * (180/M_PI);
//					double pitch = atan2(-R[2][0], sqrt(R[2][1] * R[2][1] + R[2][2] * R[2][2]));
//					double roll = atan2(R[2][1], R[2][2]);

					cv::circle(video, cv::Point(det->c[0], det->c[1]), 10, cv::Scalar(255, 0, 0), cv::FILLED, cv::LINE_8);
					cv::rectangle(video, cv::Point(det->p[0][0], det->p[0][1]), cv::Point(det->p[2][0], det->p[2][1]), cv::Scalar(0, 255, 255));
					// couldnt fit yaw in radians and degrees so i just picked degrees
					cv::putText(video, ("Tag: " + std::to_string(det->id) + " Yaw: " + std::to_string(yawDeg) + " deg").c_str(), cv::Point(det->c[0], det->c[1]), cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(255, 255, 255), 2);
				}
			}
		}
		
		cv::imshow("video", video);

		int key = cv::waitKey(5);

		if(key == 113){
//			kinect.stopDepth();
//			cv::destroyWindow("depth_heat_map");

			kinect.stopVideo();
			cv::destroyWindow("video");
			break;
		}
		else if(key == 97){
			doApril = !doApril;
			std::cout << "Turned " << (doApril ? "on " : "off ") << "april tag detection" << std::endl;
		}
		else if(key == 61){ // + (well technically =)
			tiltDegrees += 2;
			if(tiltDegrees > 20){
				tiltDegrees = 20;
			}
			std::cout << "Tilt set to: " << tiltDegrees << std::endl;
			kinect.setTiltDegrees(tiltDegrees);
		}
		else if(key == 45){ // -
			tiltDegrees -= 2;
			if(tiltDegrees < -20){
				tiltDegrees = -20;
			}
			std::cout << "Tilt set to: " << tiltDegrees << std::endl;
			kinect.setTiltDegrees(tiltDegrees);
		}
	}

	apriltag_detector_destroy(td);
	tag16h5_destroy(tf);
	std::cout << "Destroyed apriltag detections" << std::endl;

	cv::destroyAllWindows();

	kinect.setTiltDegrees(0);
	std::cout << "Reset tilt to 0" << std::endl;
 	return 0;
}

