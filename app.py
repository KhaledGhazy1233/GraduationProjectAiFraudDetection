from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import json
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# 1. تحميل الموديلات والـ Scaler
try:
    rf_model = joblib.load('rf_model.pkl')
    xgb_model = joblib.load('xgb_model.pkl')
    ann_model = load_model('fraud_model_ann.h5')
    scaler = joblib.load('scaler.pkl')
    
    with open('gui_test_cases.json', 'r', encoding='utf-8') as f:
        test_cases = json.load(f)
    print("✅ تم تحميل جميع الملفات بنجاح")
except Exception as e:
    print(f"❌ خطأ في تحميل الملفات: {e}")

# 2. مصفوفة تحليل السلوك الذكي (تفسير النتائج للمستخدم)
behavior_logs = [
    "النمط طبيعي: العميل يقوم بعملية معتادة من موقعه الجغرافي المسجل، والتوقيت متوافق مع نشاطه اليومي.",
    "النمط طبيعي: عملية شرائية صغيرة من متجر موثوق، قيم الـ V تظهر استقراراً في البصمة الرقمية.",
    "تحذير سلوكي: تم رصد محاولة دخول من موقع جغرافي بعيد في وقت متأخر من الليل مع تغيير مفاجئ في الجهاز المستخدم.",
    "خطر مرتفع: نمط السحب يظهر محاولة لتفريغ الرصيد بسرعة (Velocity Attack)، قيم الـ V تظهر انحرافاً حاداً عن سلوك البشر.",
    "النمط طبيعي: العميل يستخدم ميزة الدفع اللاتلامسي في متجر معتاد، لا يوجد أي شذوذ رقمي.",
    "تنبيه: محاولة استخدام الكارت في دولة تختلف عن موقع الهاتف المحمول، النمط الرقمي يشير لعملية مسربة.",
    "النمط طبيعي: تحديث دوري للبيانات يتبعه عملية شرائية متوسطة، السلوك يبدو روتينياً.",
    "خطر: القيم الرقمية تشير إلى استخدام برامج آلية (Bot) لإتمام العملية، النمط غير بشري تماماً."
]

@app.route('/')
def index():
    # نرسل الحالات لتعمل كقاعدة بيانات عشوائية في الواجهة
    return render_template('index.html', cases=test_cases)

@app.route('/predict_all', methods=['POST'])
def predict_all():
    try:
        data = request.get_json()
        features_dict = data['features']
        pattern_index = data.get('pattern_index', 0) # استلام رقم النمط المختار عشوائياً

        # 3. معالجة البيانات وترتيب الأعمدة (V1 إلى V28 ثم Amount)
        columns = [f'V{i}' for i in range(1, 29)] + ['Amount']
        ordered_values = [features_dict[col] for col in columns]
        
        df_input = pd.DataFrame([ordered_values], columns=columns)
        
        # 4. عمل Scaling للمبلغ فقط بناءً على تدريب الموديل
        df_input['Amount'] = scaler.transform(df_input[['Amount']])

        # 5. استخراج التوقعات من الموديلات الثلاثة
        # Random Forest
        rf_p = int(rf_model.predict(df_input)[0])
        
        # XGBoost
        xgb_p = int(xgb_model.predict(df_input)[0])
        
        # Neural Network (ANN)
        ann_prob = float(ann_model.predict(df_input, verbose=0)[0][0])
        ann_p = 1 if ann_prob > 0.5 else 0

        # 6. القرار النهائي (تصويت الأغلبية)
        final_score = rf_p + xgb_p + ann_p
        final_decision = "Fraud" if final_score >= 2 else "Safe"

        # 7. جلب التفسير السلوكي المناسب
        analysis = behavior_logs[pattern_index] if pattern_index < len(behavior_logs) else "تحليل النمط غير متوفر."

        # إرجاع النتائج شاملة التحليل والنسب
        return jsonify({
            'rf': 'Fraud' if rf_p == 1 else 'Safe',
            'xgb': 'Fraud' if xgb_p == 1 else 'Safe',
            'ann': 'Fraud' if ann_p == 1 else 'Safe',
            'final': final_decision,
            'behavior_analysis': analysis
        })

    except Exception as e:
        print(f"⚠️ خطأ أثناء المعالجة: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)