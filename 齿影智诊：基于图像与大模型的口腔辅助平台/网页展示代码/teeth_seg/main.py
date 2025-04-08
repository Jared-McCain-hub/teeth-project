from flask import Flask,request,render_template, jsonify, url_for, Response
import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import shutil
import base64
import requests
import io
import json

import torch
from torch.utils.data  import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from skimage.segmentation import watershed
from skimage.measure import label
from skimage.color import label2rgb
from skimage import measure

# 添加qwen2.5模块路径
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
qwen_path = os.path.join(parent_dir, "齿影智诊：基于图像与大模型的口腔辅助平台")
if qwen_path not in sys.path:
    sys.path.append(qwen_path)

# 导入OpenAI客户端
from openai import OpenAI

app = Flask(__name__)

# 确保静态目录存在
os.makedirs(os.path.join(app.root_path, 'static'), exist_ok=True)
os.makedirs(os.path.join(app.root_path, 'static/images'), exist_ok=True)

# 检查视频文件是否存在，如果不存在则创建一个空视频文件
video_path = os.path.join(app.root_path, 'static/yayayaya.mp4')
if not os.path.exists(video_path):
    # 创建一个空文件，实际使用时应该提供一个真实的视频文件
    with open(video_path, 'w') as f:
        pass

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
tx_X = transforms.Compose([
                           transforms.Resize((512, 512)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                           ])

ALLOWED_EXTENSIONS = set([
    "png","jpg","JPG","PNG", "bmp"
])

def is_allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

####model

class res_conv(nn.Module):
    def __init__(self, input_channels, output_channels, down=True):
        super(res_conv, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(output_channels),
                                   nn.LeakyReLU(inplace = True),
                                   nn.Dropout(0.1),
                                 )
        self.conv2 = nn.Sequential(nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(output_channels),
                                   nn.LeakyReLU(inplace = True),
                                   nn.Dropout(0.1),
                                  )
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)+x1
        return x2

class start_conv(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(start_conv, self).__init__()
        self.conv = res_conv(input_channels, output_channels)

    def forward(self, x):
        x = self.conv(x)
        return x


class down_conv(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(down_conv, self).__init__()
        self.conv = nn.Sequential(nn.MaxPool2d(2),
                                  res_conv(input_channels, output_channels),)
    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(up_conv, self).__init__()
        self.up = nn.ConvTranspose2d(input_channels//2, input_channels//2, kernel_size=2, stride=2)
        self.conv = res_conv(input_channels, output_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff1 = x2.shape[2]-x1.shape[2]
        diff2 = x2.shape[3]-x1.shape[3]
        x1 = F.pad(x1, pad=(diff1//2, diff1-diff1//2, diff2//2, diff2-diff2//2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class stop_conv(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(stop_conv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(input_channels, output_channels, kernel_size=1),
                                 nn.Sigmoid())

    def forward(self, x):
        x = self.conv(x)
        return x

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.inc = start_conv(1, 64)
        self.down1 = down_conv(64, 128)
        self.down2 = down_conv(128, 256)
        self.down3 = down_conv(256, 512)
        self.down4 = down_conv(512, 512)
        self.up1 = up_conv(1024, 256)
        self.up2 = up_conv(512, 128)
        self.up3 = up_conv(256, 64)
        self.up4 = up_conv(128, 64)
        self.outc = stop_conv(64, 1)

    def forward(self, x):
        xin = self.inc(x)
        xd1 = self.down1(xin)
        xd2 = self.down2(xd1)
        xd3 = self.down3(xd2)
        xd4 = self.down4(xd3)
        xu1 = self.up1(xd4, xd3)
        xu2 = self.up2(xu1, xd2)
        xu3 = self.up3(xu2, xd1)
        xu4 = self.up4(xu3, xin)
        out = self.outc(xu4)
        return out


def ConnectedComp(img):
	# Load in image, convert to gray scale, and Otsu's threshold
	kernel =(np.ones((3,3), dtype=np.float32))
	# print(img.dtype)
	image=cv2.resize(img.astype(np.float32),(3040,1280))
	image=cv2.morphologyEx(image, cv2.MORPH_OPEN,kernel)    # 首先对输入的彩色图像进行开运算(先腐蚀后膨胀)以去除噪声

	# sharpen=np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
	# image=cv2.filter2D(image,-1,sharpen)

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)     # 将图像转为灰度图并进行二值化，使用大津算法确定阈值。
	grayy = (gray*255*10).astype(np.uint8)
	thresh = cv2.threshold(grayy, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

	erosion = cv2.erode(thresh,kernel,iterations=3) #,iterations=2
	#gradient, aka the contours
	gradient = cv2.morphologyEx(erosion, cv2.MORPH_GRADIENT, kernel)    #对腐蚀后的图像进行梯度运算，得到目标物体的轮廓。

	ret,markers=cv2.connectedComponents(erosion,connectivity=8)      # 使用OpenCV提供的connectedComponents函数计算出图像中的联通组件。
	new = watershed(erosion,markers,mask=thresh)   # 使用分水岭算法将图像分割成连接的分量，并生成RGB图像，其中每个分量由不同的颜色表示。
	RGB = label2rgb(new, bg_label=0)    # 使用label2rgb函数为各个连通区域着不同颜色。

	return erosion,gradient,RGB   # 返回二值化后的图像、梯度图和连通组件着色的结果图。


def im_converterX(tensor):
    image = tensor.cpu().clone().detach().numpy()    #克隆张量
    image = image.transpose(1,2,0)    # Pytorch中为[Channels, H, W]，而plt.imshow()中则是[H, W, Channels]，所以交换一下通道
    image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))    # 反转一下transforms.Normalize（）的过程
    image = image.clip(0, 1)    # clip函数，将小于0的数字变为0，将大于1的数字变为1，归一化
    return image
def im_converterY(tensor):
    image = tensor.cpu().clone().detach().numpy()
    image = image.transpose(1,2,0)
    image = image * np.array((1, 1, 1))
    image = image.clip(0, 1)
    return image

def load_model():
	model = Unet()
	model_path = os.path.join(os.path.dirname(__file__), 'best_unet_051722_v1.pth')
	model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), False)
	model = model.to(device)
	return model

def get_result(img_path):
    src=Image.open(img_path)
    Xs=tx_X(src)
    #breakpoint()
    imgx=im_converterX(Xs)    # 使用im_converterX将输出数据从Tensor类型转换为图像格式
    imgx=cv2.resize(imgx, (3040,1280), interpolation = cv2.INTER_AREA)


    Xs = Xs[None,:].to(device)
    output_img=im_converterY(model(Xs)[0])
    output_img=cv2.resize(output_img, (3040,1280), interpolation = cv2.INTER_AREA)      # 将输出结果调整至(3040, 1280)大小的图像output_img，并采用INTER_AREA方式进行插值
    erosion,gradient,RGB=ConnectedComp(output_img)   # 得到三个通道的二值图像
    save_path=os.path.dirname(img_path)
    cv2.imwrite(save_path+"/"+os.path.basename(img_path).split(".")[0]+"_mask.jpg",output_img*255)
    cv2.imwrite(save_path+"/"+os.path.basename(img_path).split(".")[0]+"_rgb.jpg",RGB*255)
    res=[img_path,save_path+"/"+os.path.basename(img_path).split(".")[0]+"_mask.jpg",save_path+"/"+os.path.basename(img_path).split(".")[0]+"_rgb.jpg"]
    return res
    #res包含了原始图像、mask和rgb三个文件的路径信息。

# 调用Qwen2.5大模型分析图像
def analyze_with_qwen(image_path):
    try:
        # 创建OpenAI客户端
        client = OpenAI(
            base_url='https://api-inference.modelscope.cn/v1/',
            api_key='d15696b8-7e0d-46bb-8492-5af7947b9bc0',  # ModelScope Token
        )
        
        # 读取图像并转为base64
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # 构建提示
        prompt = """这是一张牙齿X光分割图像。请分析这张图像，指出可能存在的牙齿问题。

分析内容应包括：

【问题分析】
1. 龋齿情况：观察是否存在明显的黑色阴影区域，表明牙体组织被破坏。
2. 缺失情况：检查是否有牙齿缺失或拔除。
3. 畸形情况：观察牙齿排列是否存在不规则或异常。
4. 牙周疾病：通过牙齿根部周围的骨质密度判断是否存在牙周病。

请用专业且易懂的语言描述发现的问题。"""
        
        # 调用模型
        response = client.chat.completions.create(
            model='Qwen/Qwen2.5-VL-7B-Instruct',  # ModelScope Model-Id
            messages=[{
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': prompt},
                    {'type': 'image_url', 'image_url': {'url': f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }],
            temperature=0.7,
            max_tokens=1024
        )
        
        # 返回模型回复
        return response.choices[0].message.content
    except Exception as e:
        app.logger.error(f"调用Qwen2.5模型时出错: {str(e)}")
        return f"分析过程中发生错误: {str(e)}"

# 添加图像分析API路由
@app.route('/analyze_image')
def analyze_image():
    try:
        image_path = request.args.get('path')
        if not image_path:
            return jsonify({"success": False, "error": "未提供图像路径"})
        
        # 获取完整的图像路径
        if image_path.startswith('/static'):
            image_path = os.path.join(app.root_path, image_path[1:])
        
        # 检查文件是否存在
        if not os.path.exists(image_path):
            return jsonify({"success": False, "error": f"图像文件不存在: {image_path}"})
        
        # 调用大模型进行分析
        analysis_result = analyze_with_qwen(image_path)
        
        # 格式化结果为HTML
        formatted_result = analysis_result.replace('\n', '<br>')
        
        # 返回分析结果
        return jsonify({"success": True, "result": formatted_result})
    except Exception as e:
        app.logger.error(f"分析图像时出错: {str(e)}")
        return jsonify({"success": False, "error": str(e)})

# 添加获取解决方案API路由
@app.route('/get_solution')
def get_solution():
    try:
        image_path = request.args.get('path')
        if not image_path:
            return jsonify({"success": False, "error": "未提供图像路径"})
        
        # 获取完整的图像路径
        if image_path.startswith('/static'):
            image_path = os.path.join(app.root_path, image_path[1:])
        
        # 检查文件是否存在
        if not os.path.exists(image_path):
            return jsonify({"success": False, "error": f"图像文件不存在: {image_path}"})
        
        # 调用大模型获取解决方案
        solution_result = get_solution_with_model(image_path)
        
        # 格式化结果为HTML
        formatted_result = solution_result.replace('\n', '<br>')
        
        # 返回解决方案
        return jsonify({"success": True, "result": formatted_result})
    except Exception as e:
        app.logger.error(f"获取解决方案时出错: {str(e)}")
        return jsonify({"success": False, "error": str(e)})

# 使用DeepSeek模型获取解决方案
def get_solution_with_model(image_path):
    try:
        # 创建OpenAI客户端
        client = OpenAI(
            base_url='https://api-inference.modelscope.cn/v1/',
            api_key='d15696b8-7e0d-46bb-8492-5af7947b9bc0',  # ModelScope Token
        )
        
        # 读取图像并转为base64
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # 构建提示
        prompt = """这是一张牙齿X光分割图像。请根据图像分析结果，提供详细的治疗方案和解决建议。

【具体治疗方案】
1. 牙齿排列不齐：说明是否需要正畸治疗（如佩戴牙套）来矫正牙齿位置。
2. 龋齿：如果发现龋齿迹象，建议进行补牙手术以修复受损部分。
3. 牙周病：如果存在牙龈炎或牙周炎迹象，建议进行牙周治疗，包括洁牙和使用抗生素等。
4. 牙齿损伤：如果发现牙齿断裂或磨损情况，建议进行根管治疗或其他修复性治疗。

【日常护理建议】
1. 定期刷牙：每天至少刷牙两次，每次至少两分钟，使用含氟牙膏。
2. 使用牙线：每天使用牙线清理牙缝，保持口腔清洁。
3. 健康饮食：减少糖分摄入，多吃富含维生素的食物，增强牙齿和牙龈的抵抗力。
4. 定期检查：每半年到一年进行一次口腔检查，及时发现并处理问题。

【预防措施】
1. 定期维护：定期进行专业洁牙，去除牙菌斑和牙石。
2. 使用漱口水：使用含氟漱口水可以帮助预防蛀牙。
3. 戒烟限酒：吸烟和过量饮酒会增加患牙周病的风险。
4. 避免咬硬物：不要用牙齿开瓶盖或咬硬物，以免造成牙齿损伤。

【需要就医的紧急情况】
1. 剧烈疼痛：如果出现剧烈的牙齿疼痛，可能是由感染或其他严重问题引起的，应立即就医。
2. 牙齿折断：如果牙齿突然折断，应尽快就医，以便及时进行修复。
3. 牙龈出血：如果牙龈经常出血，可能是牙周病的早期症状，应及时就医。
4. 面部肿胀：如果面部出现肿胀，可能是感染扩散的表现，需要立即就医。

请注意，以上建议仅供参考，具体的治疗方案和护理措施需要根据个人的具体情况由专业牙医制定。"""
        
        # 调用模型
        response = client.chat.completions.create(
            model='Qwen/Qwen2.5-VL-7B-Instruct',  # ModelScope Model-Id
            messages=[{
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': prompt},
                    {'type': 'image_url', 'image_url': {'url': f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }],
            temperature=0.7,
            max_tokens=1024
        )
        
        # 返回模型回复
        return response.choices[0].message.content
    except Exception as e:
        app.logger.error(f"调用模型获取解决方案时出错: {str(e)}")
        return f"生成解决方案时发生错误: {str(e)}"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == "POST":
        try:
            f = request.files['file']
            if not f:
                return render_template("upload.html", error="请选择要上传的图片")
                
            if not (f and is_allowed_file(f.filename)):
                return render_template("upload.html", error="请检查上传的图片类型，仅限于png、PNG、jpg、JPG、bmp")
                
            basepath = os.path.dirname(__file__)
            upload_path = os.path.join(basepath, "static/images", f.filename)
            f.save(upload_path)
            
            try:
                res = get_result(upload_path)
                save_path = "/static/images"
                kwargs = {
                    "path1": save_path+"/"+os.path.basename(res[0]),
                    "path2": save_path+"/"+os.path.basename(res[1]),
                    "path3": save_path+"/"+os.path.basename(res[2])
                }
                
                # 直接在服务器端获取AI分析结果
                try:
                    ai_result = analyze_with_qwen(res[1])  # 分析分割结果图像
                    kwargs["ai_result"] = ai_result.replace('\n', '<br>')
                    
                    # 获取解决方案
                    solution_result = get_solution_with_model(res[1])
                    kwargs["solution_result"] = solution_result.replace('\n', '<br>')
                except Exception as ae:
                    app.logger.error(f"预加载AI分析时出错: {str(ae)}")
                    kwargs["ai_result"] = None
                    kwargs["solution_result"] = None
                
                return render_template("upload_ok.html", **kwargs)
            except Exception as e:
                app.logger.error(f"处理图片时出错: {str(e)}")
                return render_template("upload.html", error="处理图片时出错，请重试")
        except Exception as e:
            app.logger.error(f"上传图片时出错: {str(e)}")
            return render_template("upload.html", error="上传图片时出错，请重试")
            
    return render_template("upload.html")

@app.errorhandler(404)
def page_not_found(e):
    return render_template("upload.html", error="请求的页面不存在"), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template("upload.html", error="服务器内部错误，请稍后重试"), 500

@app.route("/medical_assistant")
def medical_assistant():
    """医疗助手页面路由"""
    return render_template("medical_assistant.html")

@app.route("/settings")
def settings():
    """系统设置页面路由"""
    return render_template("settings.html")

@app.route("/history")
def history():
    """历史记录页面路由"""
    return render_template("history.html")

@app.route("/help")
def help():
    """帮助中心页面路由"""
    return render_template("help.html")

@app.route("/chat", methods=["POST"])
def chat():
    """处理医疗助手聊天请求"""
    try:
        data = request.json
        message = data.get("message")
        context = data.get("context", "常规咨询")  # 获取上下文信息，默认为常规咨询
        analysis_content = data.get("analysis_content", "")  # 获取分析报告内容
        solution_content = data.get("solution_content", "")  # 获取解决方案内容
        
        if not message:
            return jsonify({"success": False, "error": "消息不能为空"})
        
        # 使用自定义的API密钥和URL
        key = 'sk-0b795981f6de4e5691640669220423dc'
        api_url = "https://api.deepseek.com"
        
        # 创建OpenAI客户端
        client = OpenAI(
            base_url=api_url,
            api_key=key
        )
        
        # 根据不同的上下文构建提示词
        system_prompt = "您是一位专业的牙科医疗助手，具有丰富的口腔医学知识"
        
        if context == "基于X光片分析结果的问诊":
            # 如果有分析内容和解决方案内容，则将其包含在提示中
            report_context = ""
            if analysis_content:
                report_context += f"\n\n分析报告内容：\n{analysis_content}"
            if solution_content:
                report_context += f"\n\n解决方案内容：\n{solution_content}"
                
            user_prompt = f"""您是一位专业的牙科医疗助手，负责解答患者关于牙齿X光片分析结果的问题。

患者已经看到了自己牙齿X光片的AI分析结果和解决方案建议，现在可能有更多具体问题。
请基于下面提供的分析报告和解决方案内容，针对患者的问题提供更详细、个性化的建议。
您的回答必须直接基于这些分析和解决方案内容，不要提供与分析不符的信息。
回答应当清晰、专业，同时富有同理心，语气应当平和友好。
{report_context}

患者问题: {message}

请记住：
1. 仅基于提供的分析报告和解决方案内容回答问题
2. 给出实用的建议和解释，但不要超出已有信息的范围
3. 在必要时建议患者咨询专业牙医
4. 不做确定性诊断，强调您的建议仅供参考
5. 如果患者问题与提供的报告内容无关，请引导患者询问与报告相关的问题"""
        else:
            user_prompt = f"""您是一位专业的牙科医疗助手，具有医学知识和专业素养。请用专业且友好的语气回答以下牙科相关问题。
            提供准确的医学信息，但要强调您仅提供一般性建议，不能替代专业医师的诊断和治疗。
            
用户问题: {message}"""
        
        # 调用模型生成回复（非流式）
        response = client.chat.completions.create(
            model='deepseek-chat',  # 使用适合的DeepSeek模型
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=1024
        )
        
        # 返回响应
        return jsonify({
            "success": True,
            "response": response.choices[0].message.content,
            "streaming": False  # 非流式标记
        })
    except Exception as e:
        app.logger.error(f"处理聊天请求时出错: {str(e)}")
        return jsonify({"success": False, "error": str(e)})

@app.route("/chat_stream", methods=["POST"])
def chat_stream():
    """处理医疗助手聊天请求（流式响应）"""
    def generate():
        try:
            data = request.json
            message = data.get("message")
            context = data.get("context", "常规咨询")
            analysis_content = data.get("analysis_content", "")
            solution_content = data.get("solution_content", "")
            
            if not message:
                yield f"data: {json.dumps({'success': False, 'error': '消息不能为空'})}\n\n"
                return
            
            # 使用自定义的API密钥和URL
            key = 'sk-0b795981f6de4e5691640669220423dc'
            api_url = "https://api.deepseek.com"
            
            # 创建OpenAI客户端
            client = OpenAI(
                base_url=api_url,
                api_key=key
            )
            
            # 根据不同的上下文构建提示词
            system_prompt = "您是一位专业的牙科医疗助手，具有丰富的口腔医学知识"
            
            if context == "基于X光片分析结果的问诊":
                # 如果有分析内容和解决方案内容，则将其包含在提示中
                report_context = ""
                if analysis_content:
                    report_context += f"\n\n分析报告内容：\n{analysis_content}"
                if solution_content:
                    report_context += f"\n\n解决方案内容：\n{solution_content}"
                    
                user_prompt = f"""您是一位专业的牙科医疗助手，负责解答患者关于牙齿X光片分析结果的问题。

患者已经看到了自己牙齿X光片的AI分析结果和解决方案建议，现在可能有更多具体问题。
请基于下面提供的分析报告和解决方案内容，针对患者的问题提供更详细、个性化的建议。
您的回答必须直接基于这些分析和解决方案内容，不要提供与分析不符的信息。
回答应当清晰、专业，同时富有同理心，语气应当平和友好。
{report_context}

患者问题: {message}

请记住：
1. 仅基于提供的分析报告和解决方案内容回答问题
2. 给出实用的建议和解释，但不要超出已有信息的范围
3. 在必要时建议患者咨询专业牙医
4. 不做确定性诊断，强调您的建议仅供参考
5. 如果患者问题与提供的报告内容无关，请引导患者询问与报告相关的问题"""
            else:
                user_prompt = f"""您是一位专业的牙科医疗助手，具有医学知识和专业素养。请用专业且友好的语气回答以下牙科相关问题。
                提供准确的医学信息，但要强调您仅提供一般性建议，不能替代专业医师的诊断和治疗。
                
用户问题: {message}"""
            
            # 调用模型生成流式回复
            stream = client.chat.completions.create(
                model='deepseek-chat',
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1024,
                stream=True  # 启用流式生成
            )
            
            # 发送开始标记
            yield f"data: {json.dumps({'success': True, 'type': 'start'})}\n\n"
            
            # 流式返回每个生成的文本片段
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    yield f"data: {json.dumps({'success': True, 'type': 'chunk', 'content': content})}\n\n"
            
            # 发送结束标记
            yield f"data: {json.dumps({'success': True, 'type': 'end'})}\n\n"
            
        except Exception as e:
            app.logger.error(f"处理流式聊天请求时出错: {str(e)}")
            yield f"data: {json.dumps({'success': False, 'error': str(e)})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

if __name__=="__main__":
	model=load_model()
	app.run(host="0.0.0.0",port=5001,debug=True)