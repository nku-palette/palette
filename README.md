# palette项目
palette是一个包含物体检测和分类能力的在线服务工程。前期开发以识别小宠物为目标，提供一个前端界面，从前端可以传图片、或者传视频流进行实时监测，检测图片中或者视频流中是否有小宠物，如果有，给出检测结果，如果是视频流，进行实时跟踪。

# 工程结构说明
* conf 配置文件
* docker docker镜像工具
* front_end 前端工程
* log 工程日志工具
* models 检测/分类模型工程
* service 将模型加载并启动API服务的serving工程，该目录下又有两个子目录，分别为api_service和model_service，这两块分别是对外提供restful服务接口以及将模型加载至内存进行inference服务的工程。
* tools 通用工具包

# 详细功能设计
TODO @Luoadore 补充
## 前端基本功能模块
### 需求概述
* 【基本需求】用户检测感兴趣图片或者视频中是否包含小动物。若包含，检测出小动物并框出，标明小动物的类别并给出可能性打分。
* 【扩展需求】用户可登陆，可保存结果令其满意的检测结果（为自己的小宠物建相册等）。视频还可做实时宠物状态检测器，快速检测跟踪。 

### Palette结构图

### 模块概述
- 小动物图片检测

  功能：实现图片中小宠物的检测功能

  性能：为用户提供图片检测界面。
  
  输入：一张或者多张图片。
  
  输出：无小宠物，输出原图片及语句提示；有小宠物，输出将原图片中所有可能是小动物的框，框上还有名称及可能性得分。
- 小动物视频检测
  
  功能：实现视频流中小宠物的检测功能

  性能：为用户提供视频流检测界面。
  
  输入：一段视频。
  
  输出：无小宠物，输出原视频及语句提示；有小宠物，输出将原视频中所有可能是小动物的框，框上还有名称及可能性得分，并持续跟踪整个视频中出现的小动物。
- 摄像头实时监测小宠物是否有异常行为

  功能：实现摄像头在线拍摄下实时监控小宠物，如有异常远程发出警告功能

  性能：为用户提供可实时观看摄像头拍摄画面的界面，同时发生异常时有提示或弹窗（后续再想仔细的）。

  输入：无。

  输出：用户点击观看时输出为实时画面，有异常时输出可能出现的异常提示窗口。

### 整体页面
简洁大方。
首页分为两部分，整体背景之上大标题。
![index](https://raw.githubusercontent.com/nku-palette/palette/master/design/index.png)
向下滑动鼠标，出现选择按钮，分别为photo和vedio对应的小宠物检测。
![index2](https://raw.githubusercontent.com/nku-palette/palette/master/design/index2.png)

### 视频流检测页面
按钮1：点击选择本地图片

按钮2：上传

按钮3:点击预测按钮

大窗口：输出图片

小窗口：结果

### 图片检测页面
按钮1：点击选择本地视频

按钮2：上传

按钮3:点击预测按钮

大窗口：输出视频

小窗口：结果

### 摄像头实时监测页面
大窗口：用来实时输出画面。

按钮：点击观看。

## 前端与后端服务的API接口设计

### 图片预测接口
Request

POST /palette/pictures HTTP/1.1

Host: api.luoadore.org

Accpet: application/json

Content-Type: application/jpg,png,etc

Content-Length:

{

'id':

'data':

}
### 图片预测结果返回接口
Response

HTTP/1.1 200 OK

Date:

Content-Type: application/json

Access-Control-Max-picture: (要是有《硅谷》那个压缩算法就不用定义了耶)

Cache-Control:

{

"id":

'data':

'categroy':

'score':

'non-animal': (如果没有小动物输出的信息，但创建这个是不是有点浪费)

}
### 视频流检测接口



### 报错接口

## 后端服务与模型服务API接口设计

TODO



Edit By [MaHua](http://mahua.jser.me)
