import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, request, render_template
import pandas as pd
from customer_segmentation_model import CustomerSegmentationModel
from sklearn.cluster import KMeans

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train_and_elbow', methods=['POST'])
def train_and_elbow():
    file = request.files['file']
    if file:
        k = int(request.form.get('k', 4))  # Lấy giá trị k từ form
        max_k = int(request.form.get('max_k', 10))  # Giá trị max_k từ form
        data = pd.read_excel(file, sheet_name='Online Retail')

        # Kiểm tra các cột cần thiết
        required_columns = ['CustomerID', 'InvoiceNo', 'Quantity', 'UnitPrice', 'InvoiceDate']
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            return f"Lỗi: Dữ liệu thiếu các cột cần thiết: {', '.join(missing_columns)}"

        # Khởi tạo mô hình phân cụm
        segmentation_model = CustomerSegmentationModel(n_clusters=k)
        customer_segments = segmentation_model.fit(data)

        # In cấu trúc dữ liệu
        print("Cấu trúc dữ liệu sau phân cụm:")
        print(customer_segments.head())
        print("Tên cột hiện tại:", list(customer_segments.columns))

        # Đổi tên cột
        if len(customer_segments.columns) == 5:
             customer_segments.columns = ['Số ID khách hàng', 'Tần suất mua hàng', 'Tổng chi tiêu', 'Độ gần đây (ngày)', 'Cụm']
        else:
             return f"Lỗi: Số lượng cột trong DataFrame không khớp với dự kiến. Các cột hiện tại là: {list(customer_segments.columns)}"

        # ===== Biểu đồ phân cụm =====
        plt.figure(figsize=(10, 6))
        clusters = customer_segments['Cụm']
        plt.scatter(customer_segments['Tần suất mua hàng'], customer_segments['Tổng chi tiêu'], c=clusters, cmap='viridis')
        plt.title('Phân Loại khách hàng')
        plt.xlabel('Tần suất mua hàng')
        plt.ylabel('Tổng chi tiêu')
        plt.colorbar(label='Cụm')
        plt.tight_layout()

        # Lưu biểu đồ phân cụm
        img_cluster = io.BytesIO()
        plt.savefig(img_cluster, format='png')
        img_cluster.seek(0)
        plot_url_cluster = base64.b64encode(img_cluster.getvalue()).decode()

        # ===== Biểu đồ Elbow =====
        distortions = []
        customer_metrics = segmentation_model.preprocess_data(data)
        features = customer_metrics[['Frequency', 'Monetary', 'Recency']]
        data_scaled = segmentation_model.scaler.fit_transform(features)

        for k_value in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=k_value, random_state=42)
            kmeans.fit(data_scaled)
            distortions.append(kmeans.inertia_)

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, max_k + 1), distortions, marker='o')
        plt.title('Phương pháp Elbow')
        plt.xlabel('Số cụm (k)')
        plt.ylabel('Sai số (Inertia)')
        plt.grid(True)
        plt.tight_layout()

        # Lưu biểu đồ Elbow
        img_elbow = io.BytesIO()
        plt.savefig(img_elbow, format='png')
        img_elbow.seek(0)
        plot_url_elbow = base64.b64encode(img_elbow.getvalue()).decode()

        # Kết quả phân cụm (bảng HTML)
        result_html = customer_segments.to_html(classes='table table-striped', border=0, index=False)

        return render_template('index.html', table=result_html, plot_url_cluster=plot_url_cluster, plot_url_elbow=plot_url_elbow)

    return "Vui lòng upload file Excel hợp lệ."



if __name__ == '__main__':
    app.run(debug=True)
