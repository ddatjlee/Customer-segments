import pandas as pd
from customer_segmentation_model import CustomerSegmentationModel

# Đọc dữ liệu từ file
file_path = 'D:/CNTT/Python/AI/Online Retail.xlsx'
data = pd.read_excel(file_path, sheet_name='Online Retail')

# Khởi tạo mô hình phân cụm
segmentation_model = CustomerSegmentationModel(n_clusters=4)

# Huấn luyện mô hình
customer_segments = segmentation_model.fit(data)

# Hiển thị kết quả phân cụm
print(customer_segments)

# Dự đoán với dữ liệu mới
import numpy as np
new_data = np.array([[10, 500, 20], [5, 100, 60]])  # Example new customer behavior
clusters = segmentation_model.predict(new_data)
print("Dự đoán cụm:", clusters)

# Lấy tọa độ trung tâm các cụm
cluster_centers = segmentation_model.get_cluster_centers()
print("Tọa độ trung tâm cụm:", cluster_centers)
