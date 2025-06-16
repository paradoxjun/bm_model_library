#include <fstream>
#include <string.h>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include "json.hpp"

#include "model_func.hpp"
// #include "bytetrack/bytetrack.h"
// #include "models/yolov8_det.hpp"

using json = nlohmann::json;
using namespace std;


/*-------------------------------------------
                  Main Function
-------------------------------------------*/
#ifdef __cplusplus
extern "C" {          // 确保函数名称不会在导出时被修饰
#endif

BYTETracker* init_bytetrack_model(bytetrack_params track_params){
    return new BYTETracker(track_params);
}


STracks inference_person_bytetrack_model(BYTETracker bytetrack, object_detect_result_list result, bool enable_logger=false){
    STracks output_stracks;
    bytetrack.update(output_stracks, result);
    if (enable_logger) {
      std::cout << "output_stracks size: " << output_stracks.size() << std::endl;
      for (auto& track_box : output_stracks) {
        int track_id = track_box->track_id;
        int box_x1 = track_box->tlwh[0];
        int box_y1 = track_box->tlwh[1];
        int box_w = track_box->tlwh[2];
        int box_h = track_box->tlwh[3];
        int frame_id = track_box->frame_id;
        int class_id = track_box->class_id;
        std::string text = "track_id:" + std::to_string(track_id);
        std::cout << "frame_id:" << frame_id << " track_id:" << track_id << " class_id:" << class_id << " box:" << box_x1 << "," << box_y1 << "," << box_w << "," << box_h << std::endl;
        }
    }    
    return output_stracks;
}


#ifdef __cplusplus
}
#endif