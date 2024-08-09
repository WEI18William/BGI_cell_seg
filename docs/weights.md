## Release

- Cross Point Detector
  - [points_detect_yolov5obb_SSDNA_20220513_pytorch.onnx](https://pan.genomics.cn/ucdisk/api/2.0/share/link/download?shareEventId=share_202331691548476_d9f05ef0d23a4a069ffcbff5a49ac860&nodeId=8a80804a867c36b90186e43cb7e4478a&code=)
- Clarity Eval
  - [clarity_eval_mobilev3small05064_DAPI_20230202_pytorch.onnx](https://pan.genomics.cn/ucdisk/api/2.0/share/link/download?shareEventId=share_202331611295523_28e42b24e57844768733f13eab17c1e4&nodeId=8a80804a867c36b90186e43c9b034785&code=)
- YOLO Tissue Segment
  - [tissueseg_yolo_SH_20230131_th.onnx](https://pan.genomics.cn/ucdisk/api/2.0/share/link/download?shareEventId=share_20233117301767_10ff2d6b54c94351a4654eb8c9bd3ddb&nodeId=8a808043867c394401869c7d01282e29)
- BCDU Tissue Segment
  - [tissueseg_bcdu_S_240618_tf.onnx](https://pan.genomics.cn/ucdisk/api/2.0/share/link/download?shareEventId=share_20233117301767_10ff2d6b54c94351a4654eb8c9bd3ddb&nodeId=8a808043867c394401869c7d1f062e38)
  - [tissueseg_bcdu_H_221101_tf.onnx](https://pan.genomics.cn/ucdisk/api/2.0/share/link/download?shareEventId=share_20233117301767_10ff2d6b54c94351a4654eb8c9bd3ddb&nodeId=8a808043867c394401869c7debdd2e57)
  - [tissueseg_bcdu_rna_220909_tf.onnx](https://pan.genomics.cn/ucdisk/api/2.0/share/link/download?shareEventId=share_20233117301767_10ff2d6b54c94351a4654eb8c9bd3ddb&nodeId=8a808043867c394401869c7d87842e4b)
- BCDU Cell Segment
    - [cellseg_bcdu_SHDI_221008_tf.onnx](https://pan.genomics.cn/ucdisk/api/2.0/share/link/download?shareEventId=share_20233117301767_10ff2d6b54c94351a4654eb8c9bd3ddb&nodeId=8a808043867c394401869c7d10192e32)
    
    
[Meaning of weight's name? ](manage.md)

You can auto download all these weights in one shot. You do not have to manually click each one.

```python
import cellbin.dnn.weights as cdw
all_weights = cdw.weights  # this return a dict. key is the name of the weight, value is the link of the weight 
save_dir = "a_directory_where_you_want_to_save"
names = all_weights.keys()  # get all model name. You can print and select some of them as needed
cdw.auto_download_weights(
    save_dir=save_dir,
    names=names
)
```

[comment]: <> (<br>)

[comment]: <> (<table>)

[comment]: <> (  <tr>)

[comment]: <> (    <th>DNN Inference Framework</th><th>Weights File </th><th>Version</th><th>Release Date</th>)

[comment]: <> (  </tr>)

[comment]: <> (  <tr>   )

[comment]: <> (    <td rowspan="6"><center>ONNX &#40;Open Neural Network Exchange&#41;</center></td><td> Cross Point Detector </td><td>2022-06-30</td><td>V0.0.1</td>)

[comment]: <> (  </tr>)

[comment]: <> (  <tr>)

[comment]: <> (    <td> Cross Point Filter </td><td>2022-06-30</td><td>V0.0.1</td>)

[comment]: <> (  </tr>)

[comment]: <> (  <tr>)

[comment]: <> (    <td> Clarity Evaler </td><td>2022-06-30</td><td>V0.0.1</td>)

[comment]: <> (  </tr>)

[comment]: <> (  <tr>)

[comment]: <> (    <td> YOLO Tissue Segmentor </td><td>2022-06-30</td><td>V0.0.1</td>)

[comment]: <> (  </tr>)

[comment]: <> (  <tr>)

[comment]: <> (    <td> BCDU Tissue Segmentor </td><td>2022-06-30</td><td>V0.0.1</td>)

[comment]: <> (  </tr>)

[comment]: <> (  <tr>)

[comment]: <> (    <td> BCDU Cell Segmentor  </td><td>2022-06-30</td><td>V0.0.1</td>)

[comment]: <> (  </tr>)


[comment]: <> (  <tr>   )

[comment]: <> (    <td rowspan="6"><center> *TensorRT &#40;NVIDIA&#41; </center></td><td> Cross Point Detector </td><td>2022-06-30</td><td>V0.0.1</td>)

[comment]: <> (  </tr>)

[comment]: <> (  <tr>)

[comment]: <> (    <td> Cross Point Filter </td><td>2022-06-30</td><td>V0.0.1</td>)

[comment]: <> (  </tr>)

[comment]: <> (  <tr>)

[comment]: <> (    <td> Clarity Evaler </td><td>2022-06-30</td><td>V0.0.1</td>)

[comment]: <> (  </tr>)

[comment]: <> (  <tr>)

[comment]: <> (    <td> YOLO Tissue Segmentor </td><td>2022-06-30</td><td>V0.0.1</td>)

[comment]: <> (  </tr>)

[comment]: <> (  <tr>)

[comment]: <> (    <td> BCDU Tissue Segmentor </td><td>2022-06-30</td><td>V0.0.1</td>)

[comment]: <> (  </tr>)

[comment]: <> (  <tr>)

[comment]: <> (    <td> BCDU Cell Segmentor  </td><td>2022-06-30</td><td>V0.0.1</td>)

[comment]: <> (  </tr>)

[comment]: <> (</table>)