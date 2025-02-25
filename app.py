import os
import cv2
import numpy as np
from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
import zipfile
import io
import base64
from skimage.restoration import denoise_wavelet

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['PROCESSED_FOLDER'] = 'processed/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

# ================== 图像处理函数扩展 ==================
def visible_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l_clahe = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l_clahe, a, b)), cv2.COLOR_LAB2BGR)

def visible_histogram(img):
    return cv2.cvtColor(cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)), cv2.COLOR_GRAY2BGR)

def visible_gamma(img, gamma=1.5):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

def infrared_median(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.medianBlur(gray, 5)

def infrared_gaussian(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5,5), 0)

def sar_nlmeans(img):
    return cv2.fastNlMeansDenoising(img, h=15)

def sar_wavelet(img):
    denoised = denoise_wavelet(img, channel_axis=-1, rescale_sigma=True)
    return (denoised * 255).astype(np.uint8)

# ================== 处理路由 ==================
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# 修改处理路由中的图像读取部分
@app.route('/process', methods=['POST'])
def process_images():
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for modality in ['visible', 'infrared', 'sar']:
            files = request.files.getlist(modality)
            algorithm = request.form.get(f'{modality}_algorithm')
            
            for file in files:
                if file.filename == '': continue
                filename = secure_filename(file.filename)
                
                # 修正图像解码部分
                file_data = file.read()  # 先读取文件数据
                np_array = np.frombuffer(file_data, np.uint8)  # 指定数据类型
                img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)  # 正确参数传递
                
                # 应用处理算法
                if modality == 'visible':
                    if algorithm == 'clahe':
                        processed = visible_clahe(img)
                    elif algorithm == 'histogram':
                        processed = visible_histogram(img)
                    elif algorithm == 'gamma':
                        processed = visible_gamma(img)
                elif modality == 'infrared':
                    if algorithm == 'median':
                        processed = infrared_median(img)
                    elif algorithm == 'gaussian':
                        processed = infrared_gaussian(img)
                elif modality == 'sar':
                    if algorithm == 'nlmeans':
                        processed = sar_nlmeans(img)
                    elif algorithm == 'wavelet':
                        processed = sar_wavelet(img)
                
                # 保存处理结果
                _, buffer = cv2.imencode('.jpg', processed)
                zip_file.writestr(f"{modality}/{filename}", buffer.tobytes())
    
    zip_buffer.seek(0)
    return send_file(zip_buffer, mimetype='application/zip', download_name='processed_images.zip')

if __name__ == '__main__':
    app.run(debug=True)