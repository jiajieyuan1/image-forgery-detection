"""
Flask API 服务
提供REST接口用于图像伪造检测
"""

import torch
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
import sys
import io
from PIL import Image
import base64

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from src.models import ModelFactory
from src.inference import InferenceEngine
from config import DEPLOYMENT, MODELS_DIR
from utils.metrics import ExplanationGenerator


app = Flask(__name__)
CORS(app)

# 全局变量
model = None
inference_engine = None


def load_model():
    """加载模型"""
    global model, inference_engine
    
    model_path = MODELS_DIR / 'best_model.pth'
    if not model_path.exists():
        model_path = MODELS_DIR / 'final_model.pth'
    
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    print(f"加载模型: {model_path}")
    
    # 创建模型
    model = ModelFactory.create_model('efficientnet', num_classes=2, pretrained=False)
    
    # 加载权重
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    # 创建推理引擎
    inference_engine = InferenceEngine(model, device=device)
    
    print("模型加载完成！")


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    })


@app.route('/predict', methods=['POST'])
def predict():
    """预测接口"""
    try:
        # 检查模型是否加载
        if inference_engine is None:
            return jsonify({'error': '模型未加载'}), 500
        
        # 获取上传的文件
        if 'file' not in request.files:
            return jsonify({'error': '未找到文件'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': '文件名为空'}), 400
        
        # 检查文件格式
        allowed_formats = set(DEPLOYMENT['supported_formats'])
        file_ext = file.filename.split('.')[-1].lower()
        if file_ext not in allowed_formats:
            return jsonify({'error': f'不支持的文件格式: {file_ext}'}), 400
        
        # 加载图像
        img_data = file.read()
        
        # 检查文件大小
        if len(img_data) > DEPLOYMENT['max_upload_size_mb'] * 1024 * 1024:
            return jsonify({'error': '文件过大'}), 400
        
        # 保存临时文件
        temp_path = Path('/tmp') / f'temp_{np.random.randint(10000)}.jpg'
        with open(temp_path, 'wb') as f:
            f.write(img_data)
        
        try:
            # 推理
            result = inference_engine.infer_single_image(str(temp_path))
            
            # 生成解释
            explanation = ExplanationGenerator.generate_prediction_explanation(result)
            result['explanation'] = explanation
            
            return jsonify({
                'success': True,
                'result': result
            })
        
        finally:
            # 删除临时文件
            if temp_path.exists():
                temp_path.unlink()
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """批量预测接口"""
    try:
        if inference_engine is None:
            return jsonify({'error': '模型未加载'}), 500
        
        if 'files' not in request.files:
            return jsonify({'error': '未找到文件'}), 400
        
        files = request.files.getlist('files')
        results = []
        
        for file in files:
            if file.filename == '':
                continue
            
            try:
                # 加载图像
                img_data = file.read()
                temp_path = Path('/tmp') / f'temp_{np.random.randint(100000)}.jpg'
                
                with open(temp_path, 'wb') as f:
                    f.write(img_data)
                
                try:
                    result = inference_engine.infer_single_image(str(temp_path))
                    result['image_name'] = file.filename
                    results.append(result)
                finally:
                    if temp_path.exists():
                        temp_path.unlink()
            
            except Exception as e:
                results.append({
                    'image_name': file.filename,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'total': len(files),
            'results': results
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/model_info', methods=['GET'])
def model_info():
    """获取模型信息"""
    return jsonify({
        'model_name': 'EfficientNet-b4 Forgery Detector',
        'num_classes': 2,
        'classes': ['Real', 'Fake'],
        'input_size': [256, 256],
        'supported_formats': DEPLOYMENT['supported_formats'],
        'max_upload_size_mb': DEPLOYMENT['max_upload_size_mb']
    })


@app.route('/predict_base64', methods=['POST'])
def predict_base64():
    """Base64图像预测接口"""
    try:
        if inference_engine is None:
            return jsonify({'error': '模型未加载'}), 500
        
        data = request.json
        if 'image' not in data:
            return jsonify({'error': '未找到图像数据'}), 400
        
        # 解码Base64
        img_data = base64.b64decode(data['image'])
        temp_path = Path('/tmp') / f'temp_{np.random.randint(100000)}.jpg'
        
        with open(temp_path, 'wb') as f:
            f.write(img_data)
        
        try:
            result = inference_engine.infer_single_image(str(temp_path))
            return jsonify({
                'success': True,
                'result': result
            })
        finally:
            if temp_path.exists():
                temp_path.unlink()
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/v1/detect', methods=['POST'])
def detect_v1():
    """API v1 检测端点"""
    # 与 /predict 相同
    return predict()


# 错误处理
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': '端点不存在'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': '服务器内部错误'}), 500


def main():
    """启动API服务"""
    print("=" * 50)
    print("图像伪造检测 API 服务")
    print("=" * 50)
    
    try:
        load_model()
    except Exception as e:
        print(f"错误: {e}")
        print("请先训练模型并保存到 models/ 目录")
        return
    
    print(f"\n启动服务在 {DEPLOYMENT['flask_host']}:{DEPLOYMENT['flask_port']}")
    print(f"文档: http://{DEPLOYMENT['flask_host']}:{DEPLOYMENT['flask_port']}/model_info")
    
    app.run(
        host=DEPLOYMENT['flask_host'],
        port=DEPLOYMENT['flask_port'],
        debug=False,
        threaded=True
    )


if __name__ == '__main__':
    main()
