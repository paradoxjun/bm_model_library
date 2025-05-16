
model_name=$1
model_def=$2
quant=$3  # F32 or F16
mlir=$1".mlir"
test_result=$1"_top_outputs.npz"

test_input=$1"_in_"$3".npz"
model=$1"_1684x_"$3".bmodel"

#onnx to mlir,yolov8
#python python/tools/model_transform.py --model_name yolov8n --model_def /workspace/model/best.onnx --input_shapes [[1,3,640,640]] --mean 0.0,0.0,0.0 --scale 0.0039216,0.0039216,0.0039216 --keep_aspect_ratio --pixel_format rgb --mlir yolov8n.mlir --test_input /workspace/model/1.jpg --test_result yolov8n_top_outputs.npz
# python python/tools/model_transform.py --model_name ${model_name} --model_def ${model_def} --input_shapes [[1,3,640,640]] --mean 0.0,0.0,0.0 --scale 0.0039216,0.0039216,0.0039216 --keep_aspect_ratio --pixel_format rgb --mlir ${mlir} --test_input /workspace/model/1.jpg --test_result ${test_result}

# mlir to fp16 bm_model,yolov8
#python python/tools/model_deploy.py --mlir yolov8n.mlir --quantize F16 --chip bm1684x --test_input yolov8n_in_f32.npz --test_reference yolov8n_top_outputs.npz --tolerance 0.99,0.99 --model yolov8n_1684x_f16.bmodel
# python python/tools/model_deploy.py --mlir ${mlir} --quantize F16 --chip bm1684x --test_input ${test_input} --test_reference ${test_result} --tolerance 0.99,0.99 --model ${model}

#  resnet50,onnx to mlir, model_def is onnx file
python python/tools/model_transform.py --model_name ${model_name} --model_def ${model_def} --input_shapes [[1,3,96,96]] --pixel_format bgr --mlir ${mlir}
python python/tools/model_deploy.py --mlir ${mlir} --quantize ${quant} --chip bm1684x --tolerance 0.99,0.99 --model ${model}