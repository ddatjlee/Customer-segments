import os
import io
import base64
from flask import Flask, request, render_template, session
import pandas as pd
import matplotlib.pyplot as plt
from customer_segmentation_model import CustomerSegmentationModel
from sklearn.cluster import KMeans

# Khởi tạo Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Tạo thư mục lưu trữ tạm thời
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload_file', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file:
        # Lưu tệp vào thư mục tạm thời
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        session['uploaded_file_path'] = file_path  # Lưu đường dẫn tệp vào session

        # Đọc tệp để xử lý biểu đồ Elbow
        data = pd.read_excel(file_path, sheet_name='Online Retail')
        distortions = []
        segmentation_model = CustomerSegmentationModel()
        customer_metrics = segmentation_model.preprocess_data(data)
        features = customer_metrics[['Frequency', 'Monetary', 'Recency']]
        data_scaled = segmentation_model.scaler.fit_transform(features)

        max_k = 10  # Số cụm tối đa
        for k_value in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=k_value, random_state=42)
            kmeans.fit(data_scaled)
            distortions.append(kmeans.inertia_)

        # Vẽ biểu đồ Elbow
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, max_k + 1), distortions, marker='o')
        plt.title('Kết quả Elbow')
        plt.xlabel('Số cụm (k)')
        plt.ylabel('Sai số (Inertia)')
        plt.grid(True)
        plt.tight_layout()

        img_elbow = io.BytesIO()
        plt.savefig(img_elbow, format='png')
        img_elbow.seek(0)
        plot_url_elbow = base64.b64encode(img_elbow.getvalue()).decode()

        return render_template('index.html', plot_url_elbow=plot_url_elbow)

    return "Vui lòng upload file Excel hợp lệ."


@app.route('/train_and_cluster', methods=['POST'])
def train_and_cluster():
    k = int(request.form.get('k', 4))
    file_path = session.get('uploaded_file_path')  # Lấy đường dẫn tệp từ session
    if file_path and os.path.exists(file_path):
        # Đọc lại tệp
        data = pd.read_excel(file_path, sheet_name='Online Retail')

        # Huấn luyện mô hình
        segmentation_model = CustomerSegmentationModel(n_clusters=k)
        customer_segments = segmentation_model.fit(data)

        # Vẽ biểu đồ phân cụm
        plt.figure(figsize=(10, 6))
        clusters = customer_segments['Cluster']
        plt.scatter(customer_segments['Frequency'], customer_segments['Monetary'], c=clusters, cmap='viridis')
        plt.title('Kết quả biểu đồ phân loại khách hàng')
        plt.xlabel('Tần suất mua hàng')
        plt.ylabel('Tổng chi tiêu')
        plt.colorbar(label='Cụm')
        plt.tight_layout()

        img_cluster = io.BytesIO()
        plt.savefig(img_cluster, format='png')
        img_cluster.seek(0)
        plot_url_cluster = base64.b64encode(img_cluster.getvalue()).decode()

        result_html = customer_segments.to_html(classes='table table-striped table-bordered', index=False)

        return render_template('index.html', table=result_html, plot_url_cluster=plot_url_cluster)

    return "Không tìm thấy tệp đã tải lên. Vui lòng thử lại."


if __name__ == '__main__':
    app.run(debug=True)
